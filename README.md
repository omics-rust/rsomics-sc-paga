# rsomics-sc-paga

PAGA (partition-based graph abstraction) cluster connectivity — a Rust port of
scanpy `sc.tl.paga`. Given a kNN neighbors graph and per-cell cluster labels, it
computes the clusters×clusters connectivity matrix and the `connectivities_tree`.
Output is **byte-identical** to scanpy 1.12.1 (0.0 diff — closed-form integer
edge counting), at ~100× the throughput.

```sh
cargo install rsomics-sc-paga
```

## Usage

```sh
rsomics-sc-paga --graph distances.triplet.tsv --groups labels.tsv \
    --out-connectivities paga_conn.tsv --out-tree paga_tree.tsv
```

- `--graph` — the kNN graph as a sparse triplet TSV (`i<TAB>j<TAB>value`, 0-based;
  only the nonzero pattern is used). For the default `v1.2` model feed
  `adata.obsp['distances']`; for `--model v1.0` feed `adata.obsp['connectivities']`
  and pass `--n-neighbors`.
- `--groups` — per-cell cluster labels (last tab field of each row). Cluster order
  follows pandas categorical (lexicographically sorted unique labels).

## Method (matched to scanpy 1.12.1 `scanpy/tools/_paga.py`)

`v1.2` (default): from the directed kNN pattern, per cluster pair `(i,j)`,
`v = inter_ij + inter_ji`, `expected = (es_i·n_j + es_j·n_i)/(n−1)`,
`connectivity = min(v/expected, 1)` (symmetric, zero diagonal). `v1.0`:
`inter_es / sqrt(n_neighbors²·n_i·n_j)`. The `connectivities_tree` is the
minimum-spanning-tree of the inverted connectivities (maximising confidence),
emitted upper-triangular; disconnected graphs give a spanning forest.

## Performance

20k cells, 320k edges, 8 clusters (mac arm64), vs scanpy 1.12.1:

| | ours | scanpy | ratio |
|---|---|---|---|
| full process | 37.3 ms | 4.06 s | ~109× faster |
| compute only | ~1.12 ms | ~207 ms | ~185× faster |

Connectivity, tree, and both models are bit-identical to scanpy. The committed
golden (`tests/golden/`) is checked in CI via `tests/compat.rs` — no scanpy needed
at test time.

## Origin

Independent Rust reimplementation of scanpy `sc.tl.paga` connectivity, informed by
the scanpy source (BSD-3-Clause, a permissive license that allows reading and
citing): `scanpy/tools/_paga.py` (`_compute_connectivities_v1_2`,
`_get_connectivities_tree_v1_2`). Method: Wolf et al., *PAGA: graph abstraction
reconciles clustering with trajectory inference through a topology preserving map
of single cells*, Genome Biology 2019, [doi:10.1186/s13059-019-1663-x](https://doi.org/10.1186/s13059-019-1663-x).
Black-box tested against scanpy 1.12.1.

License: MIT OR Apache-2.0.
Upstream credit: [scanpy](https://github.com/scverse/scanpy) (BSD-3-Clause).
