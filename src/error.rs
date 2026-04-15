//! Structured error types for the retrieval library.
//!
//! Replaces the earlier `Result<_, String>` / `Result<_, Box<dyn Error>>`
//! patterns so callers can pattern-match on variants, attach context, or
//! translate errors into JSON shapes for reporting.
//!
//! Five enums, one per concern:
//!
//! - [`DatasetError`] — parsing `.fbin` / `.u8bin` / `.i8bin` / `.b1bin` /
//!   `.ibin` files.
//! - [`GroundTruthError`] — brute-force top-K computation (shape, stride,
//!   thread-pool spawn, NumKong tensor errors).
//! - [`BackendError`] — USearch, FAISS, Qdrant, Redis, Weaviate, LanceDB,
//!   cuVS engine-level failures (unknown metric/dtype/op).
//! - [`PerfCountersError`] — Linux `perf_event_open` permission /
//!   unsupported-target paths.
//! - [`DownloadError`] — Parquet shard fetch, HTTP status, schema mismatch.
//!   Feature-gated on `download` so non-download builds don't pull reqwest.
//!
//! Binary-level `main()` functions continue to return
//! `Result<(), Box<dyn Error>>` — those are terminal sinks and every enum
//! here implements `Error`, so they box automatically via `?`.

use std::io;
use std::path::PathBuf;

// #region Dataset errors

/// Parsing a BigANN-style `.fbin` / `.u8bin` / `.i8bin` / `.b1bin` / `.ibin`
/// file failed.
#[derive(Debug, thiserror::Error)]
pub enum DatasetError {
    /// File shorter than the 8-byte `<rows: u32 LE><dims: u32 LE>` header.
    #[error("{path:?}: {kind} header too small (got {got} bytes, need at least 8)")]
    HeaderTooSmall {
        path: PathBuf,
        kind: &'static str,
        got: u64,
    },

    /// Filename extension not one of the supported formats.
    #[error("{path:?}: unsupported file extension `.{extension}` (expected fbin/u8bin/i8bin/b1bin)")]
    UnsupportedExtension { path: PathBuf, extension: String },

    /// File body shorter than the `rows * dims * scalar_size` declared in
    /// the header.
    #[error("{path:?}: body too small — expected {expected} bytes for {rows}x{dims}, got {got}")]
    BodyTooSmall {
        path: PathBuf,
        expected: u64,
        rows: u32,
        dims: u32,
        got: u64,
    },

    #[error(transparent)]
    Io(#[from] io::Error),
}

// #endregion

// #region Ground-truth errors

/// Input-validation or execution failure inside brute-force top-K ground
/// truth computation.
#[derive(Debug, thiserror::Error)]
pub enum GroundTruthError {
    /// One of the three input views has non-contiguous rows.
    #[error("{which} view must have contiguous rows")]
    NonContiguousView { which: &'static str },

    /// NumKong's `u1x8` storage requires the logical dimension to be a
    /// multiple of 8. Fingerprints like MACCS (166 logical bits) must be
    /// rounded up to 168 by the caller before packing.
    #[error("bits_per_vector={bits} must be a multiple of 8 for u1x8 storage")]
    BitsPerVectorMisaligned { bits: usize },

    /// Queries and base disagree on the vector width.
    #[error("dimension mismatch: queries.dimensions={queries}, base.dimensions={base}")]
    DimensionMismatch { queries: usize, base: usize },

    /// `top_k` larger than the base set.
    #[error("top_k={top_k} exceeds base_count={base_count}")]
    TopKTooLarge { top_k: usize, base_count: usize },

    /// A view's row stride doesn't match `bytes_per_vector`.
    #[error("{which} row stride {got} != bytes_per_vector {expected}")]
    StrideMismatch {
        which: &'static str,
        got: usize,
        expected: usize,
    },

    /// `fork_union::ThreadPool::try_spawn` failed (usually OOM or
    /// permission-related on constrained systems).
    #[error("fork_union::ThreadPool::try_spawn: {0}")]
    ThreadPool(String),

    /// Shape / allocation / kernel failure inside NumKong.
    #[error("numkong tensor error: {0:?}")]
    Tensor(numkong::TensorError),
}

impl From<numkong::TensorError> for GroundTruthError {
    fn from(error: numkong::TensorError) -> Self {
        Self::Tensor(error)
    }
}

// #endregion

// #region Backend errors

/// Errors raised by the backend engines (USearch, FAISS, etc.) and the
/// small amount of wrapper code that drives them.
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    /// Unknown distance-metric string (e.g. `--metric foo`).
    #[error("{backend}: unknown metric `{value}`")]
    UnknownMetric { backend: &'static str, value: String },

    /// Unknown scalar-type string (e.g. `--dtype foo`).
    #[error("{backend}: unknown dtype `{value}`")]
    UnknownDtype { backend: &'static str, value: String },

    /// Engine-internal operation failure — `add`, `search`, `create`, etc.
    #[error("{backend}: {operation}: {message}")]
    OpFailed {
        backend: &'static str,
        operation: &'static str,
        message: String,
    },
}

// #endregion

// #region Perf-counter errors

/// Linux hardware performance-counter wiring failures.
#[derive(Debug, thiserror::Error)]
pub enum PerfCountersError {
    /// No online CPU accepted a group — usually means permissions or the
    /// host has ≤0 online CPUs (should never happen in practice).
    #[error("no online CPUs accepted a perf-counter group")]
    NoGroupsAccepted,

    /// Either `target_os != linux` or the `perf-counters` feature is off.
    #[error("perf-counters feature disabled or target_os != linux")]
    Unsupported,

    #[error(transparent)]
    Io(#[from] io::Error),
}

// #endregion

// #region Download errors

/// Network, parquet, or schema failures inside the dataset download binaries.
#[cfg(feature = "download")]
#[derive(Debug, thiserror::Error)]
pub enum DownloadError {
    /// HTTP GET returned a non-success status.
    #[error("GET {url} → {status}")]
    HttpStatus {
        url: String,
        status: reqwest::StatusCode,
    },

    /// Parquet column we need isn't in the shard's schema.
    #[error("parquet schema missing column `{column}` — wrong source?")]
    SchemaMissingColumn { column: String },

    /// Parquet column is present but has a type we can't consume.
    #[error("parquet column `{column}` has unexpected type {ty}")]
    ColumnUnexpectedType { column: String, ty: String },

    /// All shards downloaded but zero usable rows extracted.
    #[error("no rows downloaded — check --source / --url-prefix")]
    NoRowsDownloaded,

    #[error(transparent)]
    Reqwest(#[from] reqwest::Error),

    #[error(transparent)]
    Parquet(#[from] parquet::errors::ParquetError),

    #[error(transparent)]
    Io(#[from] io::Error),

    #[error(transparent)]
    Dataset(#[from] DatasetError),

    #[error(transparent)]
    GroundTruth(#[from] GroundTruthError),
}

// #endregion
