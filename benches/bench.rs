use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use rsomics_sc_paga::{Graph, Groups, Model, Paga};

/// Synthesise a chain-of-blobs kNN distances graph with `k` clusters of `per`
/// cells each and `nn` directed out-edges per cell, biased toward same-cluster
/// and adjacent-cluster targets — the structure PAGA actually contracts.
fn synth(k: usize, per: usize, nn: usize) -> (String, String) {
    let n = k * per;
    let mut g = format!("# {n}\t{n}\t{}\n", n * nn);
    let mut groups = String::from("cell\tcluster\n");
    let mut state: u64 = 0x1234_5678;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for i in 0..n {
        let c = i / per;
        groups.push_str(&format!("cell{i}\tC{c}\n"));
        for _ in 0..nn {
            // mostly intra-cluster; occasionally jump to a neighbouring cluster.
            let r = next();
            let tc = if r % 5 == 0 && c + 1 < k {
                c + 1
            } else if r % 7 == 0 && c > 0 {
                c - 1
            } else {
                c
            };
            let j = tc * per + (next() as usize % per);
            g.push_str(&format!("{i}\t{j}\t1.0\n"));
        }
    }
    (g, groups)
}

fn bench_paga(c: &mut Criterion) {
    let (graph_txt, groups_txt) = synth(20, 1000, 15);
    let graph = Graph::parse_triplets(graph_txt.as_bytes()).unwrap();
    let groups = Groups::parse(groups_txt.as_bytes()).unwrap();

    c.bench_function("paga_v1_2_20clusters_20k_cells", |b| {
        b.iter(|| {
            let p =
                Paga::compute(black_box(&graph), black_box(&groups), Model::V1_2, None).unwrap();
            black_box(p.connectivities.len())
        });
    });
}

criterion_group!(benches, bench_paga);
criterion_main!(benches);
