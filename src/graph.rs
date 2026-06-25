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
        let connectivities = match model {
            Model::V1_2 => connectivities_v1_2(graph, groups),
            Model::V1_0 => {
                let nn = n_neighbors.ok_or_else(|| {
                    RsomicsError::InvalidInput("v1.0 model requires n_neighbors".into())
                })?;
                connectivities_v1_0(graph, groups, nn)
            }
        };
        let k = groups.categories.len();
        let tree = connectivities_tree(&connectivities, k);
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
fn connectivities_v1_0(graph: &Graph, groups: &Groups, n_neighbors: usize) -> Vec<f64> {
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

    let mut conn = vec![0f64; k * k];
    for i in 0..k {
        for j in 0..k {
            if i == j {
                continue;
            }
            let v = (inter[i * k + j] + inter[j * k + i]) as f64 / 2.0;
            if v == 0.0 {
                continue;
            }
            let geom = (nn_sq * ns[i] * ns[j]).sqrt();
            conn[i * k + j] = if geom != 0.0 { v / geom } else { 1.0 };
        }
    }
    conn
}

/// scanpy `_get_connectivities_tree_v1_2`. Invert the nonzero connectivities,
/// take the minimum spanning tree of that symmetric graph (so the tree maximises
/// total confidence), and re-store the original `connectivities` value on each
/// tree edge oriented `i < j` — matching scipy's upper-triangular MST output.
fn connectivities_tree(conn: &[f64], k: usize) -> Vec<f64> {
    let edges = min_spanning_tree(conn, k);
    let mut tree = vec![0f64; k * k];
    for (i, j) in edges {
        let (a, b) = if i < j { (i, j) } else { (j, i) };
        tree[a * k + b] = conn[a * k + b];
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
        let conn = connectivities_v1_0(&g, &gr, 2);
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
        let tree = connectivities_tree(&conn, 3);
        let at = |i: usize, j: usize| tree[i * 3 + j];
        // Edges (0,1) and (1,2) maximise confidence; (0,2)=0.1 dropped.
        assert_eq!(at(0, 1), 0.8);
        assert_eq!(at(1, 2), 0.5);
        assert_eq!(at(0, 2), 0.0);
        // strictly upper triangular
        assert_eq!(at(1, 0), 0.0);
        assert_eq!(at(2, 1), 0.0);
    }
}
