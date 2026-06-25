use std::io::{BufRead, Write};

use rsomics_common::{Result, RsomicsError};

mod graph;
mod io;
mod mst;

pub use graph::{Model, Paga};
pub use io::{Graph, Groups};

/// Run PAGA: parse the kNN neighbors graph triplets and the per-cell cluster
/// labels, compute the cluster×cluster connectivity matrix, and write it (plus
/// the connectivities tree when requested).
///
/// # Errors
/// Propagates parse, dimension-mismatch, and write errors.
pub fn run<Rg: BufRead, Rl: BufRead, Wc: Write, Wt: Write>(
    graph_reader: Rg,
    groups_reader: Rl,
    model: Model,
    n_neighbors: Option<usize>,
    conn_out: Wc,
    tree_out: Option<Wt>,
) -> Result<()> {
    let graph = Graph::parse_triplets(graph_reader)?;
    let groups = Groups::parse(groups_reader)?;

    if graph.n != groups.codes.len() {
        return Err(RsomicsError::InvalidInput(format!(
            "graph has {} cells but groups has {} labels",
            graph.n,
            groups.codes.len()
        )));
    }
    if model == Model::V1_0 && n_neighbors.is_none() {
        return Err(RsomicsError::InvalidInput(
            "--model v1.0 needs --n-neighbors (the kNN graph's n_neighbors, as in sc.pp.neighbors)"
                .into(),
        ));
    }

    let paga = Paga::compute(&graph, &groups, model, n_neighbors)?;
    paga.write_matrix(conn_out, &groups.categories)?;
    if let Some(w) = tree_out {
        paga.write_tree(w, &groups.categories)?;
    }
    Ok(())
}
