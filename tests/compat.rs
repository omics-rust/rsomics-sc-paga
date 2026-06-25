//! Differential compat against scanpy 1.12.1 `sc.tl.paga` (igraph 0.11). The
//! committed golden under `tests/golden/` was captured from that exact upstream
//! and always runs in CI without scanpy present. When `RSOMICS_SCANPY_PY` points
//! at a scanpy interpreter, a live differential also runs; absent it, the live
//! half loud-skips.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_rsomics-sc-paga")
}

fn golden(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(name)
}

/// Dense numeric matrix from scanpy `np.savetxt` (no header, no row labels).
fn read_numeric(path: &Path) -> Vec<Vec<f64>> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.split('\t')
                .map(|s| s.trim().parse::<f64>().unwrap())
                .collect()
        })
        .collect()
}

/// Our labelled output (header row, first column = category name).
fn read_labelled(path: &Path) -> Vec<Vec<f64>> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.split('\t')
                .skip(1)
                .map(|s| s.trim().parse::<f64>().unwrap())
                .collect()
        })
        .collect()
}

fn max_abs(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    a.iter()
        .zip(b)
        .flat_map(|(ra, rb)| ra.iter().zip(rb).map(|(&x, &y)| (x - y).abs()))
        .fold(0.0_f64, f64::max)
}

fn max_rel(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    a.iter()
        .zip(b)
        .flat_map(|(ra, rb)| {
            ra.iter()
                .zip(rb)
                .map(|(&x, &y)| (x - y).abs() / y.abs().max(1e-12))
        })
        .fold(0.0_f64, f64::max)
}

fn run_ours(
    graph: &Path,
    groups: &Path,
    model: &str,
    n_neighbors: Option<usize>,
    out_dir: &Path,
) -> (PathBuf, PathBuf) {
    let conn = out_dir.join(format!("conn_{model}.tsv"));
    let tree = out_dir.join(format!("tree_{model}.tsv"));
    let mut cmd = Command::new(bin());
    cmd.arg("--graph")
        .arg(graph)
        .arg("--groups")
        .arg(groups)
        .args(["--model", model])
        .arg("--out-connectivities")
        .arg(&conn)
        .arg("--out-tree")
        .arg(&tree);
    if let Some(nn) = n_neighbors {
        cmd.args(["--n-neighbors", &nn.to_string()]);
    }
    let status = cmd.status().unwrap();
    assert!(status.success(), "rsomics-sc-paga exited non-zero");
    (conn, tree)
}

#[test]
fn matches_committed_scanpy_golden_v1_2() {
    let tmp = std::env::temp_dir().join("rsomics_sc_paga_golden_v12");
    std::fs::create_dir_all(&tmp).unwrap();
    let (conn, tree) = run_ours(
        &golden("distances.tsv"),
        &golden("groups.tsv"),
        "v1.2",
        None,
        &tmp,
    );

    let ours_conn = read_labelled(&conn);
    let gold_conn = read_numeric(&golden("scanpy_conn_v1.2.tsv"));
    let abs = max_abs(&ours_conn, &gold_conn);
    let rel = max_rel(&ours_conn, &gold_conn);
    assert!(abs < 1e-9, "v1.2 connectivities max abs err {abs:e}");
    assert!(rel < 1e-9, "v1.2 connectivities max rel err {rel:e}");

    let ours_tree = read_labelled(&tree);
    let gold_tree = read_numeric(&golden("scanpy_tree_v1.2.tsv"));
    let tabs = max_abs(&ours_tree, &gold_tree);
    assert!(tabs < 1e-9, "v1.2 connectivities_tree max abs err {tabs:e}");
}

#[test]
fn matches_committed_scanpy_golden_v1_0() {
    let tmp = std::env::temp_dir().join("rsomics_sc_paga_golden_v10");
    std::fs::create_dir_all(&tmp).unwrap();
    let (conn, _tree) = run_ours(
        &golden("connectivities.tsv"),
        &golden("groups.tsv"),
        "v1.0",
        Some(15),
        &tmp,
    );

    let ours_conn = read_labelled(&conn);
    let gold_conn = read_numeric(&golden("scanpy_conn_v1.0.tsv"));
    let abs = max_abs(&ours_conn, &gold_conn);
    let rel = max_rel(&ours_conn, &gold_conn);
    assert!(abs < 1e-9, "v1.0 connectivities max abs err {abs:e}");
    assert!(rel < 1e-9, "v1.0 connectivities max rel err {rel:e}");
}

#[test]
fn live_scanpy_differential() {
    let Ok(py) = std::env::var("RSOMICS_SCANPY_PY") else {
        eprintln!("SKIP live_scanpy_differential: set RSOMICS_SCANPY_PY to a scanpy python");
        return;
    };

    let tmp = std::env::temp_dir().join("rsomics_sc_paga_live");
    std::fs::create_dir_all(&tmp).unwrap();
    let dist = tmp.join("dist.tsv");
    let groups = tmp.join("groups.tsv");
    let sc_conn = tmp.join("sc_conn.tsv");

    let script = tmp.join("oracle.py");
    let mut f = std::fs::File::create(&script).unwrap();
    write!(
        f,
        r##"
import numpy as np, scanpy as sc, anndata as ad
rng = np.random.default_rng(3)
n_per, k, d = 200, 6, 40
centers = np.zeros((k, d))
for c in range(k):
    centers[c, 0] = c * 3.0
    centers[c, 1:4] = rng.standard_normal(3) * 1.5
X = np.zeros((n_per * k, d)); labels = np.zeros(n_per * k, dtype=int)
for c in range(k):
    X[c*n_per:(c+1)*n_per] = centers[c] + rng.standard_normal((n_per, d))
    labels[c*n_per:(c+1)*n_per] = c
perm = rng.permutation(n_per * k); X = X[perm]; labels = labels[perm]
a = ad.AnnData(X.astype(np.float32))
a.obs['cluster'] = [f"C{{l}}" for l in labels]
a.obs['cluster'] = a.obs['cluster'].astype('category')
sc.pp.neighbors(a, n_neighbors=15, use_rep='X', random_state=0)
dist = a.obsp['distances'].tocoo()
with open(r"{dist}", "w") as fh:
    fh.write(f"# {{a.n_obs}}\t{{a.n_obs}}\t{{dist.nnz}}\n")
    for i, j, v in zip(dist.row, dist.col, dist.data):
        fh.write(f"{{i}}\t{{j}}\t{{repr(float(v))}}\n")
with open(r"{groups}", "w") as fh:
    fh.write("cell\tcluster\n")
    for i in range(a.n_obs):
        fh.write(f"cell{{i}}\tC{{labels[i]}}\n")
sc.tl.paga(a, groups='cluster')
np.savetxt(r"{conn}", a.uns['paga']['connectivities'].toarray(), delimiter="\t")
"##,
        dist = dist.display(),
        groups = groups.display(),
        conn = sc_conn.display(),
    )
    .unwrap();

    let ok = Command::new(&py).arg(&script).status().unwrap();
    assert!(ok.success(), "scanpy paga oracle failed");

    let (conn, _tree) = run_ours(&dist, &groups, "v1.2", None, &tmp);
    let ours = read_labelled(&conn);
    let gold = read_numeric(&sc_conn);
    let abs = max_abs(&ours, &gold);
    let rel = max_rel(&ours, &gold);
    assert!(abs < 1e-9, "live v1.2 max abs err {abs:e}");
    assert!(rel < 1e-9, "live v1.2 max rel err {rel:e}");
}
