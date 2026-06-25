use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

use clap::Parser;
use rsomics_common::{CommonFlags, Result, RsomicsError, Tool, ToolMeta};
use rsomics_help::{Example, FlagSpec, HelpSpec, Origin, Section};

use rsomics_sc_paga::{Model, run};

pub const META: ToolMeta = ToolMeta {
    name: env!("CARGO_PKG_NAME"),
    version: env!("CARGO_PKG_VERSION"),
};

#[derive(Parser, Debug)]
#[command(name = "rsomics-sc-paga", version, about, long_about = None, disable_help_flag = true)]
pub struct Cli {
    /// kNN neighbors graph as a sparse `i  j  value` triplet TSV (value ignored).
    /// For `--model v1.2` pass the directed distances graph
    /// (`adata.obsp['distances']`); for `v1.0` the connectivities graph.
    /// Reads stdin when "-".
    #[arg(long, value_name = "graph.tsv", default_value = "-")]
    graph: PathBuf,

    /// Per-cell cluster labels TSV: a header row, then one row per cell whose
    /// last column is the label (cell IDs in earlier columns are ignored).
    #[arg(long, value_name = "groups.tsv")]
    groups: PathBuf,

    /// PAGA connectivity model: `v1.2` (scanpy default, distances graph) or
    /// `v1.0` (legacy, connectivities graph + --n-neighbors).
    #[arg(long, value_name = "v1.2|v1.0", default_value = "v1.2")]
    model: String,

    /// kNN `n_neighbors` (as in sc.pp.neighbors). Required for `--model v1.0`,
    /// unused for v1.2.
    #[arg(long, value_name = "N")]
    n_neighbors: Option<usize>,

    /// Cluster×cluster connectivity matrix output ("-" for stdout).
    #[arg(long, default_value = "-")]
    out_connectivities: String,

    /// Connectivities-tree (maximum-confidence spanning tree) output. Omit to
    /// skip writing the tree.
    #[arg(long)]
    out_tree: Option<String>,

    #[command(flatten)]
    pub common: CommonFlags,
}

impl Tool for Cli {
    fn meta() -> ToolMeta {
        META
    }
    fn common(&self) -> &CommonFlags {
        &self.common
    }

    fn execute(self) -> Result<()> {
        let model = match self.model.as_str() {
            "v1.2" => Model::V1_2,
            "v1.0" => Model::V1_0,
            other => {
                return Err(RsomicsError::InvalidInput(format!(
                    "unknown --model '{other}' (expected v1.2 or v1.0)"
                )));
            }
        };

        let graph_reader = open_source(&self.graph)?;
        let groups_reader = open_source(&self.groups)?;
        let conn_out = open_sink(&self.out_connectivities)?;
        let tree_out = match self.out_tree.as_deref() {
            Some(path) => Some(open_sink(path)?),
            None => None,
        };

        run(
            graph_reader,
            groups_reader,
            model,
            self.n_neighbors,
            conn_out,
            tree_out,
        )
    }
}

fn open_source(path: &PathBuf) -> Result<Box<dyn BufRead>> {
    if path.as_os_str() == "-" {
        Ok(Box::new(BufReader::new(std::io::stdin().lock())))
    } else {
        Ok(Box::new(BufReader::new(File::open(path).map_err(|e| {
            RsomicsError::InvalidInput(format!("{}: {e}", path.display()))
        })?)))
    }
}

fn open_sink(path: &str) -> Result<Box<dyn Write>> {
    if path == "-" {
        Ok(Box::new(BufWriter::new(std::io::stdout().lock())))
    } else {
        Ok(Box::new(BufWriter::new(
            File::create(path).map_err(RsomicsError::Io)?,
        )))
    }
}

pub static HELP: HelpSpec = HelpSpec {
    name: env!("CARGO_PKG_NAME"),
    version: env!("CARGO_PKG_VERSION"),
    tagline: "PAGA cluster-connectivity matrix from a kNN graph + labels (scanpy sc.tl.paga).",
    origin: Some(Origin {
        upstream: "scanpy sc.tl.paga",
        upstream_license: "BSD-3-Clause",
        our_license: "MIT OR Apache-2.0",
        paper_doi: Some("10.1186/s13059-019-1663-x"),
    }),
    usage_lines: &[
        "--graph distances.tsv --groups groups.tsv [--model v1.2|v1.0] [--n-neighbors N] [--out-connectivities conn.tsv] [--out-tree tree.tsv]",
    ],
    sections: &[Section {
        title: "OPTIONS",
        flags: &[
            FlagSpec {
                short: None,
                long: "graph",
                aliases: &[],
                value: Some("<graph.tsv>"),
                type_hint: Some("path"),
                required: false,
                default: Some("-"),
                description: "kNN neighbors graph as a sparse i/j/value triplet TSV (value ignored).",
                why_default: None,
            },
            FlagSpec {
                short: None,
                long: "groups",
                aliases: &[],
                value: Some("<groups.tsv>"),
                type_hint: Some("path"),
                required: true,
                default: None,
                description: "Per-cell cluster labels TSV (header + one label-tailed row per cell).",
                why_default: None,
            },
            FlagSpec {
                short: None,
                long: "model",
                aliases: &[],
                value: Some("<v1.2|v1.0>"),
                type_hint: None,
                required: false,
                default: Some("v1.2"),
                description: "Connectivity model: v1.2 (distances graph) or v1.0 (connectivities graph).",
                why_default: Some("scanpy's default is v1.2"),
            },
            FlagSpec {
                short: None,
                long: "n-neighbors",
                aliases: &[],
                value: Some("<N>"),
                type_hint: Some("usize"),
                required: false,
                default: None,
                description: "kNN n_neighbors; required for v1.0, unused for v1.2.",
                why_default: None,
            },
            FlagSpec {
                short: None,
                long: "out-connectivities",
                aliases: &[],
                value: Some("<path>"),
                type_hint: Some("String"),
                required: false,
                default: Some("-"),
                description: "Cluster×cluster connectivity matrix output (- for stdout).",
                why_default: None,
            },
            FlagSpec {
                short: None,
                long: "out-tree",
                aliases: &[],
                value: Some("<path>"),
                type_hint: Some("String"),
                required: false,
                default: None,
                description: "Connectivities-tree (max-confidence spanning tree) output.",
                why_default: None,
            },
        ],
    }],
    examples: &[
        Example {
            description: "Default v1.2 PAGA from a distances graph",
            command: "rsomics-sc-paga --graph distances.tsv --groups clusters.tsv --out-connectivities paga.tsv --out-tree tree.tsv",
        },
        Example {
            description: "Legacy v1.0 model from a connectivities graph",
            command: "rsomics-sc-paga --graph connectivities.tsv --groups clusters.tsv --model v1.0 --n-neighbors 15",
        },
    ],
    json_result_schema_doc: None,
};

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_debug_assert() {
        Cli::command().debug_assert();
    }
}
