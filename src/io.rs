use std::collections::BTreeSet;
use std::io::BufRead;

use rsomics_common::{Result, RsomicsError};

/// The sparsity pattern of a kNN neighbors graph, parsed from an `i  j  value`
/// triplet TSV. PAGA only uses which `(i, j)` cell-pairs are connected — the
/// edge values are discarded (scanpy does `ones.data = np.ones(...)`), so this
/// records the directed edge endpoints alone.
///
/// For `--model v1.2` feed the directed kNN **distances** graph
/// (`adata.obsp['distances']`); for `--model v1.0` feed the symmetric
/// **connectivities** graph (`adata.obsp['connectivities']`).
pub struct Graph {
    pub n: usize,
    pub src: Vec<u32>,
    pub dst: Vec<u32>,
}

impl Graph {
    /// Parse a sparse triplet TSV. A leading `# rows cols nnz` comment is
    /// optional; any `#`-prefixed or blank line is skipped. Each data line is
    /// `row<TAB>col<TAB>value`; the value column is read but ignored. Indices
    /// are 0-based. `n` is taken from the header dim when present, else from the
    /// largest index seen.
    ///
    /// # Errors
    /// Errors on a non-integer index, a missing column, or an index that is
    /// inconsistent with a declared header dimension.
    pub fn parse_triplets<R: BufRead>(reader: R) -> Result<Graph> {
        let mut declared_n: Option<usize> = None;
        let mut src = Vec::new();
        let mut dst = Vec::new();
        let mut max_idx = 0usize;

        for line in reader.lines() {
            let line = line.map_err(RsomicsError::Io)?;
            let t = line.trim();
            if t.is_empty() {
                continue;
            }
            if let Some(rest) = t.strip_prefix('#') {
                let mut it = rest.split_whitespace();
                if let (Some(r), Some(c)) = (it.next(), it.next())
                    && let (Ok(r), Ok(c)) = (r.parse::<usize>(), c.parse::<usize>())
                {
                    if r != c {
                        return Err(RsomicsError::InvalidInput(format!(
                            "neighbors graph must be square; header says {r}×{c}"
                        )));
                    }
                    declared_n = Some(r);
                }
                continue;
            }
            let mut f = t.split('\t');
            let i: usize = f
                .next()
                .ok_or_else(|| RsomicsError::InvalidInput("triplet missing row index".into()))?
                .trim()
                .parse()
                .map_err(|_| {
                    RsomicsError::InvalidInput("triplet row index not an integer".into())
                })?;
            let j: usize = f
                .next()
                .ok_or_else(|| RsomicsError::InvalidInput("triplet missing col index".into()))?
                .trim()
                .parse()
                .map_err(|_| {
                    RsomicsError::InvalidInput("triplet col index not an integer".into())
                })?;
            max_idx = max_idx.max(i).max(j);
            src.push(
                u32::try_from(i).map_err(|_| {
                    RsomicsError::InvalidInput("cell index exceeds u32 range".into())
                })?,
            );
            dst.push(
                u32::try_from(j).map_err(|_| {
                    RsomicsError::InvalidInput("cell index exceeds u32 range".into())
                })?,
            );
        }

        let n = match declared_n {
            Some(d) => {
                if max_idx >= d && !src.is_empty() {
                    return Err(RsomicsError::InvalidInput(format!(
                        "triplet index {max_idx} out of range for declared dimension {d}"
                    )));
                }
                d
            }
            None => max_idx + 1,
        };
        Ok(Graph { n, src, dst })
    }
}

/// Per-cell cluster labels, factorised to integer codes the way scanpy does:
/// the categories are the sorted-unique label strings (pandas' default
/// `astype('category')` ordering), and `codes[i]` is the category index of cell
/// `i`. PAGA's matrix rows/columns follow this category order.
pub struct Groups {
    pub categories: Vec<String>,
    pub codes: Vec<u32>,
    pub sizes: Vec<usize>,
}

impl Groups {
    /// Parse a per-cell label TSV: a header row, then one row per cell whose
    /// **last** tab-separated field is the cluster label (earlier fields, e.g. a
    /// cell ID, are ignored). Categories are the lexicographically sorted unique
    /// labels, matching pandas `Categorical` default ordering.
    ///
    /// # Errors
    /// Errors when there is no data row.
    pub fn parse<R: BufRead>(reader: R) -> Result<Groups> {
        let mut labels = Vec::new();
        let mut lines = reader.lines();
        // header
        loop {
            match lines.next() {
                Some(l) => {
                    let l = l.map_err(RsomicsError::Io)?;
                    if l.trim().is_empty() || l.starts_with('#') {
                        continue;
                    }
                    break;
                }
                None => return Err(RsomicsError::InvalidInput("empty groups file".into())),
            }
        }
        for line in lines {
            let line = line.map_err(RsomicsError::Io)?;
            if line.trim().is_empty() || line.starts_with('#') {
                continue;
            }
            let label = line.rsplit('\t').next().unwrap_or("").trim().to_string();
            labels.push(label);
        }
        if labels.is_empty() {
            return Err(RsomicsError::InvalidInput(
                "groups file has no cell rows".into(),
            ));
        }

        let cat_set: BTreeSet<&str> = labels.iter().map(String::as_str).collect();
        let categories: Vec<String> = cat_set.iter().map(|s| (*s).to_string()).collect();
        let code_of = |s: &str| categories.iter().position(|c| c == s).unwrap();

        let codes: Vec<u32> = labels
            .iter()
            .map(|l| u32::try_from(code_of(l)).unwrap())
            .collect();
        let mut sizes = vec![0usize; categories.len()];
        for &c in &codes {
            sizes[c as usize] += 1;
        }
        Ok(Groups {
            categories,
            codes,
            sizes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triplet_header_sets_dim() {
        let g = "# 5\t5\t3\n0\t1\t0.2\n1\t2\t0.3\n4\t0\t0.4\n";
        let graph = Graph::parse_triplets(g.as_bytes()).unwrap();
        assert_eq!(graph.n, 5);
        assert_eq!(graph.src, [0, 1, 4]);
        assert_eq!(graph.dst, [1, 2, 0]);
    }

    #[test]
    fn triplet_dim_inferred_when_no_header() {
        let g = "0\t1\t0.2\n2\t0\t0.3\n";
        let graph = Graph::parse_triplets(g.as_bytes()).unwrap();
        assert_eq!(graph.n, 3);
    }

    #[test]
    fn triplet_index_past_declared_dim_errors() {
        let g = "# 2\t2\t1\n0\t5\t0.2\n";
        assert!(Graph::parse_triplets(g.as_bytes()).is_err());
    }

    #[test]
    fn groups_sorted_categories_and_codes() {
        let labels = "cell\tcluster\nc0\tB\nc1\tA\nc2\tB\nc3\tC\n";
        let g = Groups::parse(labels.as_bytes()).unwrap();
        assert_eq!(g.categories, ["A", "B", "C"]);
        assert_eq!(g.codes, [1, 0, 1, 2]);
        assert_eq!(g.sizes, [1, 2, 1]);
    }

    #[test]
    fn groups_uses_last_column() {
        let labels = "id\tx\tcluster\nc0\tfoo\tT2\nc1\tbar\tT1\n";
        let g = Groups::parse(labels.as_bytes()).unwrap();
        assert_eq!(g.categories, ["T1", "T2"]);
        assert_eq!(g.codes, [1, 0]);
    }

    #[test]
    fn groups_empty_errors() {
        assert!(Groups::parse("header_only\n".as_bytes()).is_err());
    }
}
