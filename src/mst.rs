/// Minimum spanning forest of the graph whose edge weights are `1 / conn[i][j]`
/// over the nonzero entries of the symmetric `k×k` connectivity matrix. Returns
/// the chosen edges as `(i, j)` pairs.
///
/// This mirrors `scipy.sparse.csgraph.minimum_spanning_tree`: minimising the sum
/// of inverse confidences maximises total confidence, and on a disconnected
/// graph scipy returns a spanning tree per connected component (a forest), which
/// Prim's algorithm restarted from every unvisited vertex reproduces.
pub fn min_spanning_tree(conn: &[f64], k: usize) -> Vec<(usize, usize)> {
    let weight = |i: usize, j: usize| {
        let c = conn[i * k + j];
        if c == 0.0 { f64::INFINITY } else { 1.0 / c }
    };

    let mut in_tree = vec![false; k];
    let mut edges = Vec::new();

    for start in 0..k {
        if in_tree[start] {
            continue;
        }
        in_tree[start] = true;
        // best[v] = (min inverse-weight edge from the tree to v, the tree endpoint).
        let mut best_w = vec![f64::INFINITY; k];
        let mut best_from = vec![usize::MAX; k];
        let mut frontier_has = false;
        for v in 0..k {
            if !in_tree[v] {
                let w = weight(start, v);
                if w < best_w[v] {
                    best_w[v] = w;
                    best_from[v] = start;
                    frontier_has |= w.is_finite();
                }
            }
        }

        loop {
            // Pick the unvisited vertex reachable from the tree with smallest
            // inverse weight (lowest index breaks ties — deterministic).
            let mut pick = usize::MAX;
            let mut pick_w = f64::INFINITY;
            for v in 0..k {
                if !in_tree[v] && best_w[v] < pick_w {
                    pick_w = best_w[v];
                    pick = v;
                }
            }
            if pick == usize::MAX || !frontier_has {
                break;
            }
            in_tree[pick] = true;
            edges.push((best_from[pick], pick));
            frontier_has = false;
            for v in 0..k {
                if !in_tree[v] {
                    let w = weight(pick, v);
                    if w < best_w[v] {
                        best_w[v] = w;
                        best_from[v] = pick;
                    }
                    frontier_has |= best_w[v].is_finite();
                }
            }
        }
    }
    edges
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge_set(mut e: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
        for p in &mut e {
            if p.0 > p.1 {
                *p = (p.1, p.0);
            }
        }
        e.sort_unstable();
        e
    }

    #[test]
    fn picks_high_confidence_edges() {
        // Triangle: confidences 0.9 (0-1), 0.8 (1-2), 0.1 (0-2).
        // Max-confidence spanning tree drops the weakest edge (0-2).
        let conn = vec![
            0.0, 0.9, 0.1, //
            0.9, 0.0, 0.8, //
            0.1, 0.8, 0.0,
        ];
        let edges = edge_set(min_spanning_tree(&conn, 3));
        assert_eq!(edges, vec![(0, 1), (1, 2)]);
    }

    #[test]
    fn spanning_forest_on_disconnected_graph() {
        // Two components: {0,1} and {2,3}, no cross edges.
        let conn = vec![
            0.0, 0.5, 0.0, 0.0, //
            0.5, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.7, //
            0.0, 0.0, 0.7, 0.0,
        ];
        let edges = edge_set(min_spanning_tree(&conn, 4));
        assert_eq!(edges, vec![(0, 1), (2, 3)]);
    }

    #[test]
    fn isolated_vertex_yields_no_edge() {
        let conn = vec![
            0.0, 0.5, 0.0, //
            0.5, 0.0, 0.0, //
            0.0, 0.0, 0.0,
        ];
        let edges = edge_set(min_spanning_tree(&conn, 3));
        assert_eq!(edges, vec![(0, 1)]);
    }
}
