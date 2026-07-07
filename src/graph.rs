use std::fmt::Write as _;
use std::io::Write;

use rsomics_common::{Result, RsomicsError};

use crate::io::{Graph, Groups};
use crate::mst::min_spanning_tree;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Model {
    /// Default scanpy model. Directed graph from the kNN **distances** pattern;
    /// confidence = actual inter-cluster edges over the random-null expectation,
    /// clamped to 1.
    V1_2,
    /// Legacy model. Undirected graph from the **connectivities** pattern;
    /// confidence = inter-cluster edge count over a √(n_neighbors²·nᵢ·nⱼ)
    /// geometric-mean normaliser.
    V1_0,
}

/// The PAGA result: a symmetric clusters×clusters connectivity matrix and the
/// directed (upper-triangular) maximum-confidence spanning tree.
pub struct Paga {
    pub k: usize,
    /// Row-major `k×k`, symmetric, zero diagonal. scanpy `paga/connectivities`.
    pub connectivities: Vec<f64>,
    /// Row-major `k×k`. scanpy `paga/connectivities_tree`: for each spanning-tree
    /// edge `(i, j)` with `i < j`, holds `connectivities[i, j]`; all other cells 0.
    pub tree: Vec<f64>,
}

impl Paga {
    /// Compute PAGA connectivities for the given model.
    ///
    /// # Errors
    /// Errors when `v1.0` is requested without `n_neighbors`.
    pub fn compute(
        graph: &Graph,
        groups: &Groups,
        model: Model,
        n_neighbors: Option<usize>,
    ) -> Result<Paga> {
        let k = groups.categories.len();
        let (connectivities, tree) = match model {
            Model::V1_2 => {
                let conn = connectivities_v1_2(graph, groups);
                // v1.2 tree: MST on the scaled connectivities themselves.
                let tree = connectivities_tree(&conn, &conn, k);
                (conn, tree)
            }
            Model::V1_0 => {
                let nn = n_neighbors.ok_or_else(|| {
                    RsomicsError::InvalidInput("v1.0 model requires n_neighbors".into())
                })?;
                let (conn, inter_es) = connectivities_v1_0(graph, groups, nn);
                // v1.0 tree: MST on the raw inter-cluster edge counts, not the
                // scaled connectivities — the geometric-mean normaliser is per-pair
                // so the two can rank edges differently.
                let tree = connectivities_tree(&inter_es, &conn, k);
                (conn, tree)
            }
        };
        Ok(Paga {
            k,
            connectivities,
            tree,
        })
    }

    /// Write the connectivity matrix as a labelled dense TSV: an empty top-left
    /// cell, the category names as the header, then one row per cluster.
    ///
    /// # Errors
    /// Propagates write errors.
    pub fn write_matrix<W: Write>(&self, out: W, categories: &[String]) -> Result<()> {
        write_labelled(out, &self.connectivities, self.k, categories)
    }

    /// Write the connectivities tree in the same labelled dense layout.
    ///
    /// # Errors
    /// Propagates write errors.
    pub fn write_tree<W: Write>(&self, out: W, categories: &[String]) -> Result<()> {
        write_labelled(out, &self.tree, self.k, categories)
    }
}

/// scanpy `_compute_connectivities_v1_2`. The kNN distance graph is read as a
/// **directed** graph (one edge per stored `(s, t)`), with every edge weight 1.
/// For cluster pair `(i, j)`:
///   `inter[i][j]` = directed edges from a cell in i to a cell in j (i ≠ j);
///   `es[i]`       = intra-cluster directed edges in i, plus outgoing inter edges;
///   `v`           = `inter[i][j] + inter[j][i]` (symmetrised edge count);
///   `expected`    = `(es[i]·nⱼ + es[j]·nᵢ) / (n − 1)`;
///   `conn[i][j]`  = `min(v / expected, 1)`, or 1 when `expected == 0`.
fn connectivities_v1_2(graph: &Graph, groups: &Groups) -> Vec<f64> {
    let k = groups.categories.len();
    let n = groups.codes.len();
    let codes = &groups.codes;

    let mut es_inner = vec![0u64; k];
    let mut inter = vec![0u64; k * k];
    for e in 0..graph.src.len() {
        let ci = codes[graph.src[e] as usize] as usize;
        let cj = codes[graph.dst[e] as usize] as usize;
        if ci == cj {
            es_inner[ci] += 1;
        } else {
            inter[ci * k + cj] += 1;
        }
    }

    // es[i] = intra-cluster edges + all outgoing inter-cluster edges.
    let mut es = vec![0f64; k];
    for i in 0..k {
        let out: u64 = (0..k).map(|j| inter[i * k + j]).sum();
        es[i] = (es_inner[i] + out) as f64;
    }
    let ns: Vec<f64> = groups.sizes.iter().map(|&s| s as f64).collect();
    let denom = (n - 1) as f64;

    let mut conn = vec![0f64; k * k];
    for i in 0..k {
        for j in 0..k {
            if i == j {
                continue;
            }
            let v = (inter[i * k + j] + inter[j * k + i]) as f64;
            if v == 0.0 {
                continue;
            }
            let expected = (es[i] * ns[j] + es[j] * ns[i]) / denom;
            let scaled = if expected != 0.0 { v / expected } else { 1.0 };
            conn[i * k + j] = scaled.min(1.0);
        }
    }
    conn
}

/// scanpy `_compute_connectivities_v1_0`. The connectivities graph is read as an
/// **undirected** graph; `inter_es[i][j]` is the summed undirected inter-cluster
/// edge weight halved — equivalently `(inter[i][j] + inter[j][i]) / 2` over the
/// directed stored entries. Then
///   `conn[i][j] = inter_es[i][j] / √(n_neighbors² · nᵢ · nⱼ)`.
///
/// Returns both the scaled connectivities and the raw `inter_es` edge-count
/// matrix; the latter feeds the v1.0 tree MST (scanpy `_get_connectivities_tree_v1_0`).
fn connectivities_v1_0(graph: &Graph, groups: &Groups, n_neighbors: usize) -> (Vec<f64>, Vec<f64>) {
    let k = groups.categories.len();
    let codes = &groups.codes;

    let mut inter = vec![0u64; k * k];
    for e in 0..graph.src.len() {
        let ci = codes[graph.src[e] as usize] as usize;
        let cj = codes[graph.dst[e] as usize] as usize;
        if ci != cj {
            inter[ci * k + cj] += 1;
        }
    }

    let ns: Vec<f64> = groups.sizes.iter().map(|&s| s as f64).collect();
    let nn_sq = (n_neighbors * n_neighbors) as f64;

    let mut inter_es = vec![0f64; k * k];
    let mut conn = vec![0f64; k * k];
    for i in 0..k {
        for j in 0..k {
            if i == j {
                continue;
            }
            let v = (inter[i * k + j] + inter[j * k + i]) as f64 / 2.0;
            inter_es[i * k + j] = v;
            if v == 0.0 {
                continue;
            }
            let geom = (nn_sq * ns[i] * ns[j]).sqrt();
            conn[i * k + j] = if geom != 0.0 { v / geom } else { 1.0 };
        }
    }
    (conn, inter_es)
}

/// scanpy `_get_connectivities_tree_v1_x`. Take the minimum spanning tree of the
/// symmetric graph whose edge weights are `1 / mst_weights[i][j]` (so the tree
/// maximises the total of `mst_weights`), then store the `store` value on each
/// tree edge oriented `i < j` — matching scipy's upper-triangular MST output.
///
/// v1.2 runs the MST on the scaled connectivities (`mst_weights == store`); v1.0
/// runs it on the raw inter-cluster edge counts while storing the connectivities,
/// which can pick a different edge set.
fn connectivities_tree(mst_weights: &[f64], store: &[f64], k: usize) -> Vec<f64> {
    let edges = min_spanning_tree(mst_weights, k);
    let mut tree = vec![0f64; k * k];
    for (i, j) in edges {
        let (a, b) = if i < j { (i, j) } else { (j, i) };
        tree[a * k + b] = store[a * k + b];
    }
    tree
}

fn write_labelled<W: Write>(
    mut out: W,
    mat: &[f64],
    k: usize,
    categories: &[String],
) -> Result<()> {
    let mut header = String::new();
    for c in categories {
        header.push('\t');
        header.push_str(c);
    }
    writeln!(out, "{header}").map_err(RsomicsError::Io)?;

    let mut line = String::new();
    for i in 0..k {
        line.clear();
        line.push_str(&categories[i]);
        for j in 0..k {
            line.push('\t');
            push_g17(&mut line, mat[i * k + j]);
        }
        writeln!(out, "{line}").map_err(RsomicsError::Io)?;
    }
    Ok(())
}

/// Shortest decimal that round-trips the f64 at 17 significant digits, so a
/// value-level diff against scanpy's float64 matrix is exact.
fn push_g17(buf: &mut String, x: f64) {
    if x == 0.0 {
        buf.push('0');
        return;
    }
    let _ = write!(buf, "{x:.17e}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{Graph, Groups};

    // Two clusters {0,1} and {2,3}; directed distance edges among them.
    fn fixture() -> (Graph, Groups) {
        let g = "# 4\t4\t6\n0\t1\t0.5\n1\t0\t0.5\n2\t3\t0.5\n3\t2\t0.5\n0\t2\t0.9\n2\t0\t0.9\n";
        let labels = "cell\tc\nc0\tA\nc1\tA\nc2\tB\nc3\tB\n";
        (
            Graph::parse_triplets(g.as_bytes()).unwrap(),
            Groups::parse(labels.as_bytes()).unwrap(),
        )
    }

    #[test]
    fn v1_2_symmetric_zero_diagonal() {
        let (g, gr) = fixture();
        let conn = connectivities_v1_2(&g, &gr);
        assert_eq!(conn.len(), 4);
        assert_eq!(conn[0], 0.0, "diagonal A,A");
        assert_eq!(conn[3], 0.0, "diagonal B,B");
        assert_eq!(conn[1], conn[2], "off-diagonal symmetric");
        assert!(conn[1] > 0.0);
    }

    // Hand-computed v1.2 check on the fixture.
    // es_inner = [2, 2]; inter[A->B]=1 (0->2), inter[B->A]=1 (2->0).
    // es[A] = 2 + 1 = 3, es[B] = 2 + 1 = 3; ns=[2,2], n=4, n-1=3.
    // v = inter[A,B]+inter[B,A] = 2.
    // expected = (es[A]*ns[B] + es[B]*ns[A]) / 3 = (3*2 + 3*2)/3 = 4.
    // conn = min(2/4, 1) = 0.5.
    #[test]
    fn v1_2_exact_value() {
        let (g, gr) = fixture();
        let conn = connectivities_v1_2(&g, &gr);
        assert!((conn[1] - 0.5).abs() < 1e-12, "got {}", conn[1]);
    }

    #[test]
    fn v1_2_clamped_to_one() {
        // Dense bidirectional bridge → ratio above 1 must clamp.
        let g =
            "# 4\t4\t8\n0\t2\t1\n2\t0\t1\n1\t3\t1\n3\t1\t1\n0\t3\t1\n3\t0\t1\n1\t2\t1\n2\t1\t1\n";
        let labels = "cell\tc\nc0\tA\nc1\tA\nc2\tB\nc3\tB\n";
        let graph = Graph::parse_triplets(g.as_bytes()).unwrap();
        let groups = Groups::parse(labels.as_bytes()).unwrap();
        let conn = connectivities_v1_2(&graph, &groups);
        assert!(conn[1] <= 1.0 + 1e-12);
        assert_eq!(conn[1], 1.0);
    }

    // v1.0 on the fixture (treat the directed edges as the connectivities graph):
    // inter[A,B]=1, inter[B,A]=1 → inter_es = (1+1)/2 = 1.
    // geom = sqrt(nn^2 * 2 * 2) with nn=2 → sqrt(4*4) = 4. conn = 1/4 = 0.25.
    #[test]
    fn v1_0_exact_value() {
        let (g, gr) = fixture();
        let (conn, _inter_es) = connectivities_v1_0(&g, &gr, 2);
        assert!((conn[1] - 0.25).abs() < 1e-12, "got {}", conn[1]);
        assert_eq!(conn[0], 0.0);
    }

    #[test]
    fn tree_upper_triangular() {
        // Chain A-B-C connectivities → MST keeps both edges, stored i<j.
        let conn = vec![
            0.0, 0.8, 0.1, //
            0.8, 0.0, 0.5, //
            0.1, 0.5, 0.0,
        ];
        let tree = connectivities_tree(&conn, &conn, 3);
        let at = |i: usize, j: usize| tree[i * 3 + j];
        // Edges (0,1) and (1,2) maximise confidence; (0,2)=0.1 dropped.
        assert_eq!(at(0, 1), 0.8);
        assert_eq!(at(1, 2), 0.5);
        assert_eq!(at(0, 2), 0.0);
        // strictly upper triangular
        assert_eq!(at(1, 0), 0.0);
        assert_eq!(at(2, 1), 0.0);
    }

    // v1.0 tree runs the MST on the RAW inter-cluster edge counts, not the scaled
    // connectivities, so it can drop a different edge. Clusters A={0,1} B={2,3}
    // C={4..11}, n_neighbors=3: inter_es A-B=3, A-C=4, B-C=4 → conn A-B=0.5,
    // A-C=B-C=1/3. MST of 1/inter_es keeps A-C,B-C (drops the lowest-count A-B),
    // whereas MST of 1/conn would keep the highest-conn A-B. scanpy v1.0 gives
    // {A-C, B-C}.
    #[test]
    fn v1_0_tree_uses_raw_counts() {
        let mut g = String::from("# 12\t12\t22\n");
        let uedges = [
            (0, 2),
            (0, 3),
            (1, 2),
            (0, 4),
            (0, 5),
            (1, 6),
            (1, 7),
            (2, 8),
            (2, 9),
            (3, 10),
            (3, 11),
        ];
        for (a, b) in uedges {
            let _ = writeln!(g, "{a}\t{b}\t1");
            let _ = writeln!(g, "{b}\t{a}\t1");
        }
        let mut labels = String::from("cell\tcluster\n");
        for (i, l) in ["A", "A", "B", "B", "C", "C", "C", "C", "C", "C", "C", "C"]
            .iter()
            .enumerate()
        {
            let _ = writeln!(labels, "cell{i}\t{l}");
        }
        let graph = Graph::parse_triplets(g.as_bytes()).unwrap();
        let groups = Groups::parse(labels.as_bytes()).unwrap();
        let paga = Paga::compute(&graph, &groups, Model::V1_0, Some(3)).unwrap();
        let at = |i: usize, j: usize| paga.tree[i * 3 + j];
        let third = 1.0_f64 / 3.0;
        assert!((at(0, 2) - third).abs() < 1e-15, "A-C {}", at(0, 2));
        assert!((at(1, 2) - third).abs() < 1e-15, "B-C {}", at(1, 2));
        assert_eq!(at(0, 1), 0.0, "A-B must be dropped");
    }
}
