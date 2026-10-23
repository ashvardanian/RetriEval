//! Structured error types for dataset parsing and brute-force ground truth.
//!
//! Backends still return `Result<_, String>` — their failure messages are
//! engine-specific strings with no shared shape worth matching on.

use std::path::PathBuf;

use numkong as nk;

// #region Dataset errors

/// Parsing a BigANN-style `.fbin` / `.u8bin` / `.i8bin` / `.b1bin` / `.ibin`
/// file failed.
#[derive(Debug, thiserror::Error)]
pub enum DatasetError {
    /// File shorter than the 8-byte `<rows: u32 LE><dimensions: u32 LE>` header.
    #[error("{path:?}: {kind} header too small (got {got} bytes, need at least 8)")]
    HeaderTooSmall {
        path: PathBuf,
        kind: &'static str,
        got: u64,
    },

    /// Filename extension not one of the supported formats.
    #[error("{path:?}: unsupported file extension `.{extension}` (expected fbin/u8bin/i8bin/b1bin)")]
    UnsupportedExtension { path: PathBuf, extension: String },

    /// File body shorter than the `rows * dimensions * scalar_size` declared in
    /// the header.
    #[error("{path:?}: body too small — expected {expected} bytes for {rows}x{dimensions}, got {got}")]
    BodyTooSmall {
        path: PathBuf,
        expected: u64,
        rows: u32,
        dimensions: u32,
        got: u64,
    },

    /// Matryoshka truncation request invalid: 0, larger than the file's
    /// native dimensionality, or — for `b1bin` — not a multiple of 8.
    #[error("invalid truncation: requested {requested} dimensions (native {native}; B1x8 must be 8-aligned)")]
    InvalidTruncation { requested: usize, native: usize },

    /// A glob expansion produced shards with mismatched dimensionality or
    /// scalar format. All shards in one logical dataset must agree.
    #[error("{path:?}: shard width {got_width} doesn't match earlier shards' width {expected_width}")]
    ShardMismatch {
        path: PathBuf,
        expected_width: u32,
        got_width: u32,
    },
}

// #endregion

// #region Ground-truth errors

/// Input-validation or execution failure inside brute-force top-K ground
/// truth computation.
#[derive(Debug, thiserror::Error)]
pub enum GroundTruthError {
    /// Queries and base disagree on the vector width.
    #[error("dimension mismatch: queries.dimensions={queries}, base.dimensions={base}")]
    DimensionMismatch { queries: usize, base: usize },

    /// The ground-truth output view has a different number of rows than the
    /// query set — one row of neighbors is expected per query.
    #[error("row-count mismatch: ground_truth.rows={ground_truth_rows}, queries.rows={query_rows}")]
    RowCountMismatch {
        ground_truth_rows: usize,
        query_rows: usize,
    },

    /// `top_k` larger than the base set.
    #[error("top_k={top_k} exceeds base_count={base_count}")]
    TopKTooLarge { top_k: usize, base_count: usize },

    /// More base rows than the `u32` neighbor ids of `.ibin` can name.
    #[error("base_count={base_count} exceeds the 2^32 rows `u32` neighbor ids can name")]
    IndexOverflow { base_count: usize },

    /// ForkUnion failed to spawn the pool or dispatch to it, usually from OOM or permissions on
    /// constrained systems.
    #[error("ForkUnion pool: {0}")]
    ThreadPool(String),

    /// Shape / allocation / kernel failure inside NumKong.
    #[error("numkong tensor error: {0:?}")]
    Tensor(#[from] nk::Error),
}

// #endregion
