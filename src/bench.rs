//! Shared benchmark infrastructure for vector search engines.
//!
//! This is the library root. Backend binaries (`usearch.rs`, `faiss.rs`, etc.)
//! import from here and provide their own `main()`.

pub mod dataset;
#[cfg(feature = "tier2")]
pub mod docker;
#[cfg(feature = "download")]
pub mod download;
pub mod error;
pub mod eval;
#[cfg(any(feature = "generate", feature = "download"))]
pub mod ground_truth;
pub mod output;
#[cfg(any(feature = "generate", feature = "download"))]
pub mod packed_distance;
pub mod perf_counters;

use std::{
    alloc::{Allocator, System},
    collections::HashMap,
    fmt,
    hash::{BuildHasher, Hasher},
    num::{NonZeroU16, NonZeroUsize},
    ops::Range,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use forkunion::{self as fu, ThreadPool, Topology};
use indicatif::{ProgressBar, ProgressStyle};
use numkong::{self as nk, MatrixSpan, MatrixView};
use serde_json::Value;

pub use crate::{
    dataset::{Dataset, GroundTruth, Keys, Permutation},
    error::{DatasetError, GroundTruthError},
    output::{config_hash, write_report, ConfigReport, DatasetInfo, StepAddEntry, StepEntry, StepSearchEntry},
};

// #region Core types

/// Vector key type used throughout the benchmark.
pub type Key = u32;

/// Distance/similarity value returned by search operations.
pub type Distance = f32;

// #region Vector types

/// A batch of row-major vectors, one matrix row per vector; `B1x8` rows hold `dimensions` bits.
#[derive(Clone, Copy)]
pub enum Vectors<'a> {
    F32(MatrixView<'a, f32>),
    I8(MatrixView<'a, i8>),
    U8(MatrixView<'a, u8>),
    B1x8(MatrixView<'a, nk::u1x8>),
}

/// A batch's values end to end, the layout files and wire formats take.
pub enum VectorSlice<'a> {
    F32(&'a [f32]),
    I8(&'a [i8]),
    U8(&'a [u8]),
    /// Binary vectors: 1-bit values packed 8 per byte.
    B1x8(&'a [u8]),
}

impl<'a> Vectors<'a> {
    /// Shapes `values` into rows `dimensions` wide, counted in bits for `B1x8`.
    pub fn new(values: VectorSlice<'a>, dimensions: usize) -> Self {
        match values {
            VectorSlice::F32(values) => Self::F32(matrix_view(values, values.len() / dimensions, dimensions)),
            VectorSlice::I8(values) => Self::I8(matrix_view(values, values.len() / dimensions, dimensions)),
            VectorSlice::U8(values) => Self::U8(matrix_view(values, values.len() / dimensions, dimensions)),
            VectorSlice::B1x8(bytes) => Self::B1x8(binary_view(bytes, bytes.len() / (dimensions / 8), dimensions)),
        }
    }

    fn shape(&self) -> &[usize] {
        match self {
            Self::F32(view) => view.shape(),
            Self::I8(view) => view.shape(),
            Self::U8(view) => view.shape(),
            Self::B1x8(view) => view.shape(),
        }
    }

    pub fn len(&self) -> usize {
        self.shape()[0]
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Values per row, counted in bits for `B1x8`.
    pub fn dimensions(&self) -> usize {
        self.shape()[1]
    }

    /// The rows end to end.
    pub fn values(&self) -> VectorSlice<'_> {
        const WHOLE_ROWS: &str = "a batch holds whole contiguous rows";
        match self {
            Self::F32(view) => VectorSlice::F32(view.as_packed_slice().expect(WHOLE_ROWS)),
            Self::I8(view) => VectorSlice::I8(view.as_packed_slice().expect(WHOLE_ROWS)),
            Self::U8(view) => VectorSlice::U8(view.as_packed_slice().expect(WHOLE_ROWS)),
            // SAFETY: `u1x8` is a transparent byte.
            Self::B1x8(view) => {
                VectorSlice::B1x8(unsafe { pod_slice_as_bytes(view.as_packed_slice().expect(WHOLE_ROWS)) })
            }
        }
    }

    /// The values as `f32`, cast into `scratch` unless they already are.
    pub fn to_f32_in<'s, A: Allocator>(&'s self, scratch: &'s mut Vec<f32, A>) -> Result<&'s [f32], &'static str> {
        match self.values() {
            VectorSlice::F32(values) => Ok(values),
            values => cast_in(&values, scratch),
        }
    }
}

/// Casts every value of `source` into `target`, resized to fit, with NumKong's conversions.
fn cast_in<'a, T: nk::CastDType + Copy + Default, A: Allocator>(
    source: &VectorSlice<'_>,
    target: &'a mut Vec<T, A>,
) -> Result<&'a [T], &'static str> {
    let cast = match source {
        VectorSlice::F32(values) => cast_values(values, target),
        VectorSlice::I8(values) => cast_values(values, target),
        VectorSlice::U8(values) => cast_values(values, target),
        VectorSlice::B1x8(_) => return Err("Packed binary input cannot be cast to a dense type without a bit metric"),
    };
    cast.expect("matching conversion lengths");
    Ok(target)
}

fn cast_values<Source: nk::CastDType, T: nk::CastDType + Copy + Default, A: Allocator>(
    values: &[Source],
    target: &mut Vec<T, A>,
) -> Result<(), nk::Error> {
    target.clear();
    target.resize(values.len(), T::default());
    nk::cast(values, target)
}

/// A backend's wire buffer in its stored scalar type: input already in that type passes through as
/// bytes, and any other input is cast into the buffer.
pub enum CastScratch {
    F64(Vec<f64, System>),
    F32(Vec<f32, System>),
    BF16(Vec<nk::bf16, System>),
    F16(Vec<nk::f16, System>),
    I8(Vec<i8, System>),
    U8(Vec<u8, System>),
}

impl CastScratch {
    /// The bytes of `vectors` in this buffer's scalar type.
    pub fn encode<'a>(&'a mut self, vectors: &'a Vectors<'_>) -> Result<&'a [u8], &'static str> {
        let source = vectors.values();
        // SAFETY: every scalar type here is POD.
        Ok(match (self, source) {
            (Self::F32(_), VectorSlice::F32(values)) => unsafe { pod_slice_as_bytes(values) },
            (Self::I8(_), VectorSlice::I8(values)) => unsafe { pod_slice_as_bytes(values) },
            (Self::U8(_), VectorSlice::U8(values)) => values,
            (Self::F64(target), source) => unsafe { pod_slice_as_bytes(cast_in(&source, target)?) },
            (Self::F32(target), source) => unsafe { pod_slice_as_bytes(cast_in(&source, target)?) },
            (Self::BF16(target), source) => unsafe { pod_slice_as_bytes(cast_in(&source, target)?) },
            (Self::F16(target), source) => unsafe { pod_slice_as_bytes(cast_in(&source, target)?) },
            (Self::I8(target), source) => unsafe { pod_slice_as_bytes(cast_in(&source, target)?) },
            (Self::U8(target), source) => unsafe { pod_slice_as_bytes(cast_in(&source, target)?) },
        })
    }
}

/// A `[rows, bits]` view of bit-packed rows; `bits` must be a whole number of bytes.
pub fn binary_view(bytes: &[u8], rows: usize, bits: usize) -> MatrixView<'_, nk::u1x8> {
    assert!(
        bits.is_multiple_of(8) && bytes.len() == rows * (bits / 8),
        "{} bytes do not hold {rows} rows of {bits} bits",
        bytes.len()
    );
    // SAFETY: the bytes hold `rows` packed rows of `bits / 8` bytes, and `u1x8` is a transparent byte.
    unsafe { MatrixView::from_raw_parts(bytes.as_ptr().cast(), &[rows, bits], &[(bits / 8) as isize, 1]) }
        .expect("whole packed rows are a valid view")
}

/// A `[rows, columns]` view of row-major values.
pub fn matrix_view<T: nk::StorageElement>(values: &[T], rows: usize, columns: usize) -> MatrixView<'_, T> {
    assert_eq!(
        values.len(),
        rows * columns,
        "values do not hold {rows} rows of {columns}"
    );
    let element = std::mem::size_of::<T>() as isize;
    // SAFETY: the values hold `rows` rows of `columns` elements end to end.
    unsafe {
        MatrixView::from_raw_parts(
            values.as_ptr(),
            &[rows, columns],
            &[columns as isize * element, element],
        )
    }
    .expect("whole rows are a valid view")
}

/// Get the current process resident set size (RSS) in bytes.
pub fn process_rss_bytes() -> u64 {
    let process_id = sysinfo::get_current_pid().expect("get current pid");
    let mut system = sysinfo::System::new();
    system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[process_id]), true);
    system.process(process_id).map(|process| process.memory()).unwrap_or(0)
}

// #region Backend trait

/// Common trait for all vector search backends.
pub trait Backend: Send {
    fn metadata(&self) -> HashMap<String, Value>;
    fn add(&mut self, keys: &[Key], vectors: Vectors) -> Result<(), String>;
    /// Answers every row of `queries` into the matching row of `hits`.
    fn search(&mut self, queries: Vectors, hits: Hits<'_>) -> Result<(), String>;
    fn memory_bytes(&self) -> usize;

    /// How many vectors the backend holds, when it can say. A loaded index may
    /// cover a different slice of the base than this run's flags imply, and
    /// self-recall queries base rows assuming each one is indexed.
    fn indexed_count(&self) -> Option<usize> {
        None
    }

    /// Persist the index at `path`.
    fn save(&self, _path: &str) -> Result<(), String> {
        Err("--index: save not yet implemented for this backend".into())
    }
}

// #region Index configuration

/// The hyper-parameters every graph backend understands, in the shared
/// vocabulary. Backends take this by value and add their own engine-specific
/// knobs alongside; naming the fields is what stops `metric` and `data_type`,
/// or the run of `usize` graph knobs, from being transposed at a call site.
#[derive(Clone, Copy)]
pub struct IndexConfig<DataType, Metric> {
    pub dimensions: usize,
    pub data_type: DataType,
    pub metric: Metric,
    pub connectivity: usize,
    pub expansion_add: usize,
    pub expansion_search: usize,
}

/// Where a config's index comes from and goes to, decided once from `--index`.
#[derive(Clone, Copy)]
pub enum IndexFile<'a> {
    None,
    Save(&'a str),
    Load(&'a str),
}

impl<'a> IndexFile<'a> {
    /// Loads `path` when it exists and saves to it otherwise; a path serves a single config,
    /// since several would write to or read from the same file.
    pub fn new(path: Option<&'a str>, configs: usize) -> Self {
        match path {
            None => Self::None,
            Some(_) if configs > 1 => bail(&format!(
                "--index requires a single config; got {configs} configs from the sweep"
            )),
            Some(path) if Path::new(path).exists() => Self::Load(path),
            Some(path) => Self::Save(path),
        }
    }
}

impl fmt::Display for IndexFile<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => formatter.write_str("none"),
            Self::Save(path) => write!(formatter, "save to {path}"),
            Self::Load(path) => write!(formatter, "load {path}"),
        }
    }
}

// #region Result buffers

/// Every query's answers for one pass: a `[queries, count]` key matrix, its distances, and how
/// many hits each row holds.
pub struct SearchBuffers {
    keys: nk::Matrix<Key>,
    distances: nk::Matrix<Distance>,
    counts: Vec<usize, System>,
}

impl SearchBuffers {
    pub fn new(queries: usize, count: usize) -> Result<Self, String> {
        let error = |error| format!("result matrices for {queries} queries of {count} hits: {error:?}");
        let mut counts = Vec::with_capacity_in(queries, System);
        counts.resize(queries, 0);
        Ok(Self {
            keys: nk::Matrix::full(&[queries, count], Key::MAX).map_err(error)?,
            distances: nk::Matrix::full(&[queries, count], Distance::INFINITY).map_err(error)?,
            counts,
        })
    }

    /// The rows of one batch, for a backend to answer into.
    pub fn hits(&mut self, rows: Range<usize>) -> Hits<'_> {
        const WHOLE_ROWS: &str = "a batch is a run of whole rows";
        Hits {
            keys: self.keys.slice_mut((rows.clone(), ..)).expect(WHOLE_ROWS),
            distances: self.distances.slice_mut((rows.clone(), ..)).expect(WHOLE_ROWS),
            counts: &mut self.counts[rows],
        }
    }

    /// Each query's keys, distances and hit count, in query order.
    pub fn rows(&self) -> impl Iterator<Item = (&[Key], &[Distance], usize)> {
        let count = self.keys.shape()[1];
        self.keys
            .as_slice()
            .chunks_exact(count)
            .zip(self.distances.as_slice().chunks_exact(count))
            .zip(&self.counts)
            .map(|((keys, distances), &found)| (keys, distances, found))
    }

    /// Each query's found keys, best first.
    pub fn found(&self) -> impl Iterator<Item = &[Key]> {
        self.rows().map(|(keys, _, found)| &keys[..found])
    }
}

/// One batch's rows of a [`SearchBuffers`].
pub struct Hits<'a> {
    keys: MatrixSpan<'a, Key>,
    distances: MatrixSpan<'a, Distance>,
    counts: &'a mut [usize],
}

impl Hits<'_> {
    /// Hits requested per query.
    pub fn count(&self) -> usize {
        self.keys.shape()[1]
    }

    /// Each query's row, in query order.
    pub fn rows(&mut self) -> impl Iterator<Item = HitRow<'_>> {
        const WHOLE_ROWS: &str = "a batch is a run of whole rows";
        let count = self.count();
        let keys = self.keys.as_contiguous_slice_mut().expect(WHOLE_ROWS);
        let distances = self.distances.as_contiguous_slice_mut().expect(WHOLE_ROWS);
        keys.chunks_exact_mut(count)
            .zip(distances.chunks_exact_mut(count))
            .zip(self.counts.iter_mut())
            .map(|((keys, distances), found)| HitRow { keys, distances, found })
    }
}

/// One query's row of a [`Hits`] batch.
pub struct HitRow<'a> {
    keys: &'a mut [Key],
    distances: &'a mut [Distance],
    found: &'a mut usize,
}

impl HitRow<'_> {
    /// Writes the query's hits best first, pads the rest with `Key::MAX` and infinity, and counts
    /// them. `eval` scores only the counted prefix, so backends drop their engine's misses rather
    /// than yield sentinels; more hits than slots is an error.
    pub fn write(self, hits: impl IntoIterator<Item = Result<(Key, Distance), String>>) -> Result<(), String> {
        let mut found = 0;
        for hit in hits {
            let (key, distance) = hit?;
            if found == self.keys.len() {
                return Err(format!("more than the {found} hits requested"));
            }
            self.keys[found] = key;
            self.distances[found] = distance;
            found += 1;
        }
        self.keys[found..].fill(Key::MAX);
        self.distances[found..].fill(Distance::INFINITY);
        *self.found = found;
        Ok(())
    }
}

// #region Utilities

/// Reinterpret a `&[T]` as the raw bytes that back it, without copying.
///
/// # Safety
///
/// `T` must be "plain old data": no padding, no destructor, no invalid bit patterns.
pub unsafe fn pod_slice_as_bytes<T: Copy>(slice: &[T]) -> &[u8] {
    std::slice::from_raw_parts(slice.as_ptr().cast::<u8>(), std::mem::size_of_val(slice))
}

/// Format a number with thousand separators: 1234567 → "1,234,567"
pub fn format_thousands(number: u64) -> String {
    let digits = number.to_string();
    let mut result = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, character) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            result.push(',');
        }
        result.push(character);
    }
    result
}

/// Splits `range` into runs of `step`, the last one possibly shorter.
pub fn row_ranges(range: Range<usize>, step: usize) -> impl Iterator<Item = Range<usize>> {
    let end = range.end;
    range.step_by(step).map(move |start| start..(start + step).min(end))
}

// #region Common CLI args

/// Shared CLI arguments for all backends.
#[derive(clap::Args, Debug)]
pub struct CommonArgs {
    /// Base vectors to index (.fbin, .u8bin, .i8bin, .b1bin)
    #[arg(long, env = "RETRIEVAL_BASE_VECTORS")]
    pub base_vectors: PathBuf,

    /// Optional keys for the base vectors (.i32bin). One per base vector.
    #[arg(long, env = "RETRIEVAL_BASE_KEYS")]
    pub base_keys: Option<PathBuf>,

    /// Query vectors to search with
    #[arg(long, env = "RETRIEVAL_QUERY_VECTORS")]
    pub query_vectors: PathBuf,

    /// Ground-truth neighbors for those queries (.ibin)
    #[arg(long, env = "RETRIEVAL_QUERY_NEIGHBORS")]
    pub query_neighbors: PathBuf,

    /// Neighbors to request per query — the k every metric is taken at.
    /// Defaults to the ground-truth file's width, and may not exceed it.
    #[arg(long, env = "RETRIEVAL_TOP_K", value_parser = parse_count_flag)]
    pub top_k: Option<usize>,

    /// Order in which base vectors are inserted: `shuffled` by `--seed`, or the file's `original` order
    #[arg(long, env = "RETRIEVAL_INSERTION_ORDER", value_enum, default_value = "shuffled")]
    pub insertion_order: InsertionOrder,

    /// Seed of the shuffled insertion order, or `random` to draw one; 42 when unset
    #[arg(long, env = "RETRIEVAL_SEED", value_parser = parse_seed_flag)]
    pub seed: Option<Seed>,

    /// Number of measurement steps (dataset is split into this many equal parts)
    #[arg(long, env = "RETRIEVAL_STEPS", default_value_t = 10, value_parser = parse_count_flag)]
    pub steps: usize,

    /// Vectors per add() call for each of the backend's worker threads
    #[arg(long, env = "RETRIEVAL_INSERTS_PER_CORE", default_value_t = 2048, value_parser = parse_count_flag)]
    pub inserts_per_core: usize,

    /// Queries per search() call for each of the backend's worker threads
    #[arg(long, env = "RETRIEVAL_QUERIES_PER_CORE", default_value_t = 2048, value_parser = parse_count_flag)]
    pub queries_per_core: usize,

    /// Output directory for JSON result files
    #[arg(long, env = "RETRIEVAL_OUTPUT")]
    pub output: Option<PathBuf>,

    /// Cap the number of base vectors used (for calibration on a slice of a larger file).
    /// Queries and ground truth are unaffected; only the add/permutation range shrinks.
    #[arg(long, env = "RETRIEVAL_MAX_BASE_VECTORS", value_parser = parse_count_flag)]
    pub max_base_vectors: Option<usize>,

    /// Replay indexed vectors as their own queries and report the fraction that
    /// retrieve themselves within the top-`--self-search-top-k` ("self-recall").
    /// Runs once, after the last insertion, and needs no ground truth — a vector
    /// in the index is its own nearest neighbor. Because it can sweep the whole
    /// base rather than a short query file, it is also the only phase that
    /// measures throughput under sustained load.
    #[arg(long, env = "RETRIEVAL_SELF_SEARCH", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new())]
    pub self_search: bool,

    /// Neighbors requested per query during the self-recall sweep (the `k` in
    /// self-recall@k). No effect without `--self-search`.
    #[arg(long, env = "RETRIEVAL_SELF_SEARCH_TOP_K", default_value_t = 10, value_parser = parse_count_flag)]
    pub self_search_top_k: usize,

    /// How many base vectors the self-recall sweep replays. A bare integer ≥ 1
    /// is an absolute count; a value with a decimal point (≤ 1.0) is a fraction
    /// of the base, so `1.0` means all and `1` means a single vector. Omitted →
    /// all. A full sweep is the point on GPU backends, but 100M round trips
    /// through a server-style backend is not a measurement anyone waits for.
    ///
    /// The subset is the *leading* N rows of the base file, not a random draw —
    /// contiguity is what keeps the query slices zero-copy, so the throughput
    /// figure stays clean. On a base whose row order carries structure (sorted,
    /// clustered, or concatenated shards) a partial sweep is therefore not a
    /// representative sample; only the default full sweep is unbiased.
    #[arg(long, env = "RETRIEVAL_SELF_SEARCH_SAMPLE", value_name = "N|FRACTION", value_parser = parse_self_search_sample, requires = "self_search")]
    pub self_search_sample: Option<SelfSearchSample>,

    /// Matryoshka-style embedding-dimension truncations to evaluate
    /// (comma-separated). Empty → use the file's native dimensions. Each value must
    /// be ≤ the native dimensions; for `.b1bin` files each must be a multiple of 8.
    #[arg(long, env = "RETRIEVAL_DIMS", value_delimiter = ',', value_parser = parse_count_flag)]
    pub dims: Vec<usize>,
}

/// Order in which base vectors are inserted.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum InsertionOrder {
    Shuffled,
    Original,
}

impl fmt::Display for InsertionOrder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        spell_value(self, formatter)
    }
}

/// Seed of the shuffled insertion order when `--seed` is unset.
const DEFAULT_SEED: Seed = Seed(42);

impl CommonArgs {
    /// Resolve `--dims` into a sweep list. Empty CLI input expands to a
    /// single-element list at the file's native dimensions, so binaries can iterate
    /// uniformly without special-casing the no-truncation path.
    pub fn dimensions_sweep(&self, native: usize) -> Vec<usize, System> {
        let mut dimensions = Vec::new_in(System);
        dimensions.extend_from_slice(if self.dims.is_empty() {
            std::slice::from_ref(&native)
        } else {
            &self.dims
        });
        dimensions
    }
}

// #region BenchState

/// The `--self-search` pass: neighbors per replayed vector, and how many vectors replay.
#[derive(Clone, Copy)]
pub struct SelfSearch {
    pub top_k: usize,
    pub sample: Option<SelfSearchSample>,
}

impl SelfSearch {
    /// Vectors to replay out of the `vectors_indexed` an index holds.
    fn queries(self, vectors_indexed: usize) -> usize {
        resolve_self_search_sample(self.sample, vectors_indexed)
    }
}

/// Pre-loaded benchmark state. Call `BenchState::load()` once, then `run()` per configuration.
pub struct BenchState {
    dataset: Dataset,
    /// Effective base-vector count; equals `dataset.rows()` unless `--max-base-vectors` capped it.
    total_vectors: usize,
    keys: Keys,
    query_dataset: Dataset,
    ground_truth: GroundTruth,
    permutation: Permutation,
    /// Resolved `--top-k`: the search width and the k every metric uses.
    top_k: usize,
    steps: usize,
    inserts_per_core: usize,
    queries_per_core: usize,
    self_search: Option<SelfSearch>,
    output_dir: Option<PathBuf>,
    dataset_info: DatasetInfo,
    search_buffers: SearchBuffers,
    key_scratch: Vec<Key, System>,
    /// Rows copied for a batch that cannot borrow the mapped file: a gather, a truncation or a
    /// shard boundary.
    row_scratch: Vec<u8, System>,
}

impl BenchState {
    pub fn load(args: &CommonArgs) -> Result<Self, Box<dyn std::error::Error>> {
        let seed = match (args.insertion_order, args.seed) {
            (InsertionOrder::Shuffled, seed) => Some(seed.unwrap_or(DEFAULT_SEED)),
            (InsertionOrder::Original, None) => None,
            (InsertionOrder::Original, Some(_)) => {
                return Err("--seed applies only to --insertion-order shuffled".into())
            }
        };

        if let Some(directory) = &args.output {
            if directory
                .extension()
                .is_some_and(|extension| extension == "json" || extension == "jsonl")
            {
                return Err(format!(
                    "--output should be a directory, not a file: {}. \
                     Each config produces its own JSON file inside this directory.",
                    directory.display()
                )
                .into());
            }
            std::fs::create_dir_all(directory)?;
        }

        eprintln!("Loading dataset: {}", args.base_vectors.display());
        let dataset = Dataset::load(&args.base_vectors)?;
        let file_vectors = dataset.rows();
        let total_vectors = args.max_base_vectors.map_or(file_vectors, |cap| cap.min(file_vectors));
        let dimensions = dataset.dimensions();
        if total_vectors < file_vectors {
            eprintln!(
                "  {} vectors in file, capping to {} (--max-base-vectors)",
                format_thousands(file_vectors as u64),
                format_thousands(total_vectors as u64),
            );
            eprintln!(
                "  note: ground truth still refers to all {} rows, so raw recall is bounded by \
                 ~{:.0}%. The `*_normalized` fields divide that back out, assuming the indexed \
                 slice is a uniform sample of the base — treat them as estimates, and don't \
                 compare capped runs against uncapped ones.",
                format_thousands(file_vectors as u64),
                100.0 * total_vectors as f64 / file_vectors as f64,
            );
        } else {
            eprintln!(
                "  {} vectors, {} dimensions",
                format_thousands(total_vectors as u64),
                dimensions
            );
        }

        let keys = match &args.base_keys {
            Some(path) => {
                eprintln!("Loading keys: {}", path.display());
                let keys = Keys::load(path)?;
                eprintln!("  {} keys", format_thousands(keys.rows() as u64));
                // The add loop reads one key per base row, so a short file would read past its mapping.
                if keys.rows() < total_vectors {
                    return Err(format!(
                        "--base-keys has {} entries but {} base vectors are in play; \
                         one key per vector is required",
                        format_thousands(keys.rows() as u64),
                        format_thousands(total_vectors as u64),
                    )
                    .into());
                }
                keys
            }
            None => Keys::Sequential { count: total_vectors },
        };

        eprintln!("Loading queries: {}", args.query_vectors.display());
        let query_dataset = Dataset::load(&args.query_vectors)?;
        let query_count = query_dataset.rows();
        eprintln!(
            "  {} queries, {} dimensions",
            format_thousands(query_count as u64),
            query_dataset.dimensions()
        );

        for &truncated in &args.dims {
            dataset.check_dimensions(truncated)?;
            query_dataset.check_dimensions(truncated)?;
        }

        eprintln!("Loading ground truth: {}", args.query_neighbors.display());
        let ground_truth = GroundTruth::load(&args.query_neighbors)?;
        eprintln!(
            "  {} queries, {} neighbors each",
            format_thousands(ground_truth.queries() as u64),
            ground_truth.neighbors_per_query(),
        );

        let ground_truth_width = ground_truth.neighbors_per_query();
        let top_k = args.top_k.unwrap_or(ground_truth_width);
        if top_k > ground_truth_width {
            return Err(
                format!("--top-k {top_k} exceeds the ground truth's {ground_truth_width} neighbors per query").into(),
            );
        }

        let self_search = args.self_search.then_some(SelfSearch {
            top_k: args.self_search_top_k,
            sample: args.self_search_sample,
        });

        eprintln!("- Insertion order: {}", args.insertion_order);
        if let Some(seed) = seed {
            eprintln!("- Seed: {seed}");
        }
        eprintln!("- Top k: {top_k}");
        eprintln!("- Dims: {}", spell_list(args.dimensions_sweep(dimensions)));
        eprintln!("- Steps: {}", args.steps);
        eprintln!("- Inserts per core: {}", args.inserts_per_core);
        eprintln!("- Queries per core: {}", args.queries_per_core);
        match args.max_base_vectors {
            Some(count) => eprintln!("- Max base vectors: {count}"),
            None => eprintln!("- Max base vectors: all"),
        }
        match self_search {
            Some(SelfSearch { top_k, sample }) => {
                eprintln!("- Self-search top k: {top_k}");
                match sample {
                    Some(sample) => eprintln!("- Self-search sample: {sample}"),
                    None => eprintln!("- Self-search sample: all"),
                }
            }
            None => eprintln!("- Self-search: false"),
        }
        match &args.output {
            Some(path) => eprintln!("- Output: {}", path.display()),
            None => eprintln!("- Output: none"),
        }

        let permutation = match seed {
            None => Permutation::Identity,
            Some(seed) => Permutation::shuffled(total_vectors, seed),
        };

        let dataset_info = DatasetInfo {
            base_vectors_path: args.base_vectors.display().to_string(),
            query_vectors_path: args.query_vectors.display().to_string(),
            query_neighbors_path: args.query_neighbors.display().to_string(),
            vectors_count: file_vectors,
            queries_count: query_count,
            dimensions,
            neighbors_per_query: ground_truth_width,
        };

        Ok(Self {
            total_vectors,
            permutation,
            top_k,
            steps: args.steps,
            inserts_per_core: args.inserts_per_core,
            queries_per_core: args.queries_per_core,
            self_search,
            output_dir: args.output.clone(),
            dataset_info,
            search_buffers: SearchBuffers::new(query_count, top_k)?,
            key_scratch: Vec::new_in(System),
            row_scratch: Vec::new_in(System),
            dataset,
            keys,
            query_dataset,
            ground_truth,
        })
    }

    /// Native dimensions of the base dataset; truncation is per config.
    pub fn dimensions(&self) -> usize {
        self.dataset.dimensions()
    }
}

// #region Benchmark loop

type BenchResult<T> = Result<T, Box<dyn std::error::Error>>;

/// Items per second, zero before any time has passed.
fn per_second(count: usize, elapsed: f64) -> u64 {
    if elapsed > 0.0 {
        (count as f64 / elapsed) as u64
    } else {
        0
    }
}

/// A progress bar over `total` items, drawn by `template`.
fn progress_bar(total: usize, template: &str) -> ProgressBar {
    let progress = ProgressBar::new(total as u64);
    progress.set_style(
        ProgressStyle::default_bar()
            .template(template)
            .expect("a valid progress template")
            .progress_chars("##-"),
    );
    progress
}

/// Arm perf counters before a phase; on failure disarm the whole capture so
/// later phases don't try again. Single place for the error-logging wording.
fn start_counter_capture(perf: &mut Option<perf_counters::PerfCounters>, phase: &str) {
    if let Some(counters) = perf.as_mut() {
        if let Err(error) = counters.reset_and_enable() {
            eprintln!("  perf counters: reset/enable before {phase} failed ({error})");
            *perf = None;
        }
    }
}

/// Read a sample at the end of a phase; same disarm-on-error policy.
fn stop_and_read_counters(
    perf: &mut Option<perf_counters::PerfCounters>,
    phase: &str,
) -> Option<perf_counters::CounterSample> {
    let counters = perf.as_mut()?;
    match counters.disable_and_read() {
        Ok(sample) => Some(sample),
        Err(error) => {
            eprintln!("  perf counters: read after {phase} failed ({error})");
            *perf = None;
            None
        }
    }
}

/// Inserts the base rows `step` in batches of `inserts_per_call`, at `dimensions`.
fn run_add_phase(
    backend: &mut dyn Backend,
    state: &mut BenchState,
    perf: &mut Option<perf_counters::PerfCounters>,
    step: Range<usize>,
    dimensions: usize,
    inserts_per_call: usize,
) -> BenchResult<StepAddEntry> {
    let total_vectors = state.total_vectors;
    let message = |indexed: usize, throughput: u64| {
        format!(
            "{}/{} ({} add/s)",
            format_thousands(indexed as u64),
            format_thousands(total_vectors as u64),
            format_thousands(throughput),
        )
    };
    let progress = progress_bar(total_vectors, "  add    [{elapsed_precise}] {bar:40.cyan/blue} {msg}");
    progress.set_position(step.start as u64);

    start_counter_capture(perf, "add");
    let started = Instant::now();
    for batch in row_ranges(step.clone(), inserts_per_call) {
        let (keys, vectors) = match &state.permutation {
            Permutation::Identity => (
                state.keys.slice(batch.start, batch.len(), &mut state.key_scratch),
                state
                    .dataset
                    .slice(batch.start, batch.len(), dimensions, &mut state.row_scratch),
            ),
            Permutation::Shuffled { indices, .. } => {
                let indices = &indices[batch.clone()];
                for (slot, &row) in state.key_scratch.iter_mut().zip(indices) {
                    *slot = state.keys.get(row);
                }
                (
                    &state.key_scratch[..indices.len()],
                    state
                        .dataset
                        .gather(indices.iter().copied(), dimensions, &mut state.row_scratch),
                )
            }
        };
        backend.add(keys, vectors)?;
        progress.set_position(batch.end as u64);
        progress.set_message(message(
            batch.end,
            per_second(batch.end - step.start, started.elapsed().as_secs_f64()),
        ));
    }
    let elapsed = started.elapsed().as_secs_f64();
    let counters = stop_and_read_counters(perf, "add");
    let throughput = per_second(step.len(), elapsed);
    progress.set_message(message(step.end, throughput));
    progress.finish();
    Ok(StepAddEntry {
        elapsed,
        throughput,
        counters,
    })
}

/// Runs all queries against an index holding `vectors_indexed` vectors and scores them;
/// `final_step` drops the `~` that marks metrics over a partial index.
fn run_search_phase(
    backend: &mut dyn Backend,
    state: &mut BenchState,
    perf: &mut Option<perf_counters::PerfCounters>,
    dimensions: usize,
    queries_per_call: usize,
    vectors_indexed: usize,
    final_step: bool,
) -> BenchResult<StepSearchEntry> {
    let query_count = state.query_dataset.rows();
    let count = state.top_k;
    let progress = progress_bar(query_count, "  search [{elapsed_precise}] {bar:40.green/blue} {msg}");

    start_counter_capture(perf, "search");
    let started = Instant::now();
    for batch in row_ranges(0..query_count, queries_per_call) {
        let queries = state
            .query_dataset
            .slice(batch.start, batch.len(), dimensions, &mut state.row_scratch);
        backend.search(queries, state.search_buffers.hits(batch.clone()))?;
        progress.set_position(batch.end as u64);
        progress.set_message(format!(
            "{}/{} ({} search/s)",
            format_thousands(batch.end as u64),
            format_thousands(query_count as u64),
            format_thousands(per_second(batch.end, started.elapsed().as_secs_f64())),
        ));
    }
    let elapsed = started.elapsed().as_secs_f64();
    let counters = stop_and_read_counters(perf, "search");
    let throughput = per_second(query_count, elapsed);
    let scores = eval::score(state.search_buffers.found(), &state.ground_truth, count);

    // Raw metrics, since dividing out coverage would treat the indexed prefix as a uniform sample.
    let coverage = vectors_indexed as f64 / state.dataset.rows().max(1) as f64;
    let approximate = if final_step { "" } else { "~" };
    progress.finish_with_message(format!(
        "{} search/s, {approximate}recall@1={:.4}, \
         {approximate}recall@{count}={:.4}, \
         {approximate}intersection@{count}={:.4}, \
         {approximate}NDCG@{count}={:.4} ({} vectors, {:.0}% coverage)",
        format_thousands(throughput),
        scores.recall_at_1,
        scores.recall_at_k,
        scores.intersection_at_k,
        scores.ndcg_at_k,
        format_thousands(vectors_indexed as u64),
        coverage * 100.0,
    ));

    Ok(StepSearchEntry {
        queries: query_count,
        top_k: count,
        elapsed,
        throughput,
        recall_at_1: scores.recall_at_1,
        recall_at_k: scores.recall_at_k,
        intersection_at_k: Some(scores.intersection_at_k),
        ndcg_at_k: Some(scores.ndcg_at_k),
        counters,
    })
}

/// Replays the leading `query_count` indexed vectors as their own queries: a vector in the index
/// is its own nearest neighbor, so identity is the ground truth, and any dataset can run it.
/// Batches are scored and dropped as they land, since a 100M-vector answer matrix would only
/// reduce to two counters.
fn run_self_search_phase(
    backend: &mut dyn Backend,
    state: &mut BenchState,
    perf: &mut Option<perf_counters::PerfCounters>,
    count: usize,
    query_count: usize,
    dimensions: usize,
    queries_per_call: usize,
) -> BenchResult<StepSearchEntry> {
    let batch_size = queries_per_call.min(query_count);
    let mut buffers = SearchBuffers::new(batch_size, count)?;
    let mut expected_scratch = Vec::with_capacity_in(batch_size, System);
    expected_scratch.resize(batch_size, 0 as Key);
    let progress = progress_bar(query_count, "  self   [{elapsed_precise}] {bar:40.magenta/blue} {msg}");

    start_counter_capture(perf, "self-recall");
    let started = Instant::now();
    let (mut hits_at_1, mut hits_at_k) = (0, 0);
    for batch in row_ranges(0..query_count, batch_size) {
        let expected = state.keys.slice(batch.start, batch.len(), &mut expected_scratch);
        let queries = state
            .dataset
            .slice(batch.start, batch.len(), dimensions, &mut state.row_scratch);
        backend.search(queries, buffers.hits(0..batch.len()))?;
        let (at_1, at_k) = eval::self_recall_hits(buffers.found().take(batch.len()), expected, count);
        hits_at_1 += at_1;
        hits_at_k += at_k;
        progress.set_position(batch.end as u64);
        progress.set_message(format!(
            "{}/{} ({} search/s)",
            format_thousands(batch.end as u64),
            format_thousands(query_count as u64),
            format_thousands(per_second(batch.end, started.elapsed().as_secs_f64())),
        ));
    }
    let elapsed = started.elapsed().as_secs_f64();
    let counters = stop_and_read_counters(perf, "self-recall");
    let throughput = per_second(query_count, elapsed);
    let recall_at_1 = hits_at_1 as f64 / query_count as f64;
    let recall_at_k = hits_at_k as f64 / query_count as f64;
    progress.finish_with_message(format!(
        "{} search/s, recall@1={recall_at_1:.4}, recall@{count}={recall_at_k:.4} ({} queries)",
        format_thousands(throughput),
        format_thousands(query_count as u64),
    ));

    Ok(StepSearchEntry {
        queries: query_count,
        top_k: count,
        elapsed,
        throughput,
        recall_at_1,
        recall_at_k,
        // Identity truth holds one key per query, so only `recall_at_1` is meaningful.
        intersection_at_k: None,
        ndcg_at_k: None,
        counters,
    })
}

/// Parsed form of `--self-search-sample`. A value containing a decimal point is
/// a fraction of the base (`≤ 1.0`); a bare integer is an absolute vector count.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SelfSearchSample {
    Fraction(f64),
    Absolute(usize),
}

impl fmt::Display for SelfSearchSample {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fraction(fraction) => write!(formatter, "{fraction:?}"),
            Self::Absolute(count) => write!(formatter, "{count}"),
        }
    }
}

/// Parse a `--self-search-sample` spec. The base size isn't known at CLI-parse
/// time, so resolution to a concrete count is deferred to `resolve_self_search_sample`.
fn parse_self_search_sample(spec: &str) -> Result<SelfSearchSample, String> {
    let sample = if spec.contains('.') {
        let digits = spec.bytes().all(|byte| byte.is_ascii_digit() || byte == b'.');
        digits
            .then(|| spec.parse().ok())
            .flatten()
            .filter(|fraction| *fraction > 0.0 && *fraction <= 1.0)
            .map(SelfSearchSample::Fraction)
    } else {
        parse_count(spec).map(SelfSearchSample::Absolute)
    };
    sample.ok_or_else(|| "expected a positive count like 1000 or a fraction like 0.1".into())
}

/// Resolve a sample spec against the number of indexed vectors. `None` → all.
/// A fraction rounds to at least one vector; an absolute count is clamped to
/// what is actually in the index.
fn resolve_self_search_sample(sample: Option<SelfSearchSample>, vectors_indexed: usize) -> usize {
    match sample {
        None => vectors_indexed,
        Some(SelfSearchSample::Fraction(fraction)) => ((vectors_indexed as f64 * fraction).round() as usize).max(1),
        Some(SelfSearchSample::Absolute(count)) => count.min(vectors_indexed),
    }
}

/// Assemble a `ConfigReport` and write it to `<output_dir>/<backend>-<hash>.json`.
/// No-op when the state has no output directory configured.
fn save_report(
    state: &BenchState,
    mut metadata: HashMap<String, Value>,
    dimensions: usize,
    steps: Vec<StepEntry>,
) -> BenchResult<()> {
    let Some(directory) = &state.output_dir else {
        return Ok(());
    };

    // Every harness flag that changes the measurement belongs in the hash; backends add only their own knobs.
    metadata.insert("dimensions".into(), Value::from(dimensions));
    metadata.insert("vectors_count".into(), Value::from(state.total_vectors));
    metadata.insert("seed".into(), Value::from(state.permutation.seed().map(|seed| seed.0)));
    metadata.insert("steps".into(), Value::from(state.steps));
    metadata.insert("inserts_per_core".into(), Value::from(state.inserts_per_core));
    metadata.insert("queries_per_core".into(), Value::from(state.queries_per_core));
    metadata.insert("top_k".into(), Value::from(state.top_k));
    if let Some(entry) = steps.last().and_then(|step| step.self_search.as_ref()) {
        metadata.insert("self_search_top_k".into(), Value::from(entry.top_k));
        metadata.insert("self_search_sample".into(), Value::from(entry.queries));
    }
    let backend_name = metadata.get("backend").and_then(Value::as_str).unwrap_or("unknown");
    let path = directory.join(format!("{backend_name}-{}.json", config_hash(&metadata)));

    let report = ConfigReport {
        machine: output::collect_machine_info(),
        dataset: state.dataset_info.clone(),
        config: metadata,
        steps,
    };

    write_report(&path, &report)?;
    eprintln!("  → {}", path.display());
    Ok(())
}

/// Runs one configuration and writes its report. A fresh index fills step by step and is
/// measured after each; an index loaded holding `loaded` vectors is measured once.
pub fn run(
    backend: &mut dyn Backend,
    state: &mut BenchState,
    description: &str,
    dimensions: usize,
    loaded: Option<usize>,
) -> BenchResult<()> {
    let metadata = backend.metadata();
    let cores = metadata.get("threads").and_then(Value::as_u64).unwrap_or(1) as usize;
    let inserts_per_call = state.inserts_per_core * cores;
    let queries_per_call = state.queries_per_core * cores;
    let row_bytes = state.dataset.vector_bytes().max(state.query_dataset.vector_bytes());
    state.key_scratch.resize(inserts_per_call, 0 as Key);
    state
        .row_scratch
        .resize(inserts_per_call.max(queries_per_call) * row_bytes, 0);
    eprintln!("\n── {description} ──");

    // Without CAP_PERFMON or on another OS, steps run without counters and their fields stay empty.
    let mut perf = perf_counters::PerfCounters::new()
        .inspect(|_| eprintln!("  perf counters: system-wide per-CPU capture enabled"))
        .inspect_err(|error| eprintln!("  perf counters: unavailable ({error}); running without"))
        .ok();

    let mut steps = Vec::with_capacity(state.steps);
    match loaded {
        Some(indexed) => {
            if indexed != state.total_vectors {
                eprintln!(
                    "  loaded index holds {} vectors, dataset implies {} — using the index's count",
                    format_thousands(indexed as u64),
                    format_thousands(state.total_vectors as u64),
                );
            }
            let search = run_search_phase(backend, state, &mut perf, dimensions, queries_per_call, indexed, true)?;
            steps.push(StepEntry {
                vectors_indexed: indexed,
                memory_bytes: backend.memory_bytes() as u64,
                add: None,
                ground_truth_search: Some(search),
                self_search: None,
            });
        }
        None => {
            let total_vectors = state.total_vectors;
            for step in row_ranges(0..total_vectors, total_vectors.div_ceil(state.steps)) {
                let add = run_add_phase(backend, state, &mut perf, step.clone(), dimensions, inserts_per_call)?;
                let final_step = step.end == total_vectors;
                let search = run_search_phase(
                    backend,
                    state,
                    &mut perf,
                    dimensions,
                    queries_per_call,
                    step.end,
                    final_step,
                )?;
                steps.push(StepEntry {
                    vectors_indexed: step.end,
                    memory_bytes: backend.memory_bytes() as u64,
                    add: Some(add),
                    ground_truth_search: Some(search),
                    self_search: None,
                });
            }
        }
    }

    // Runs once against the finished index, so it lands on the last step.
    let vectors_indexed = steps.last().map_or(0, |step| step.vectors_indexed);
    if let Some(self_search) = state.self_search {
        let queries = self_search.queries(vectors_indexed);
        if queries > 0 {
            let entry = run_self_search_phase(
                backend,
                state,
                &mut perf,
                self_search.top_k,
                queries,
                dimensions,
                queries_per_call,
            )?;
            if let Some(last) = steps.last_mut() {
                last.self_search = Some(entry);
            }
        }
    }

    let peak_memory = steps.iter().map(|step| step.memory_bytes).max().unwrap_or(0);
    save_report(state, metadata, dimensions, steps)?;
    eprintln!("  peak memory: {:.2} GB", peak_memory as f64 / (1u64 << 30) as f64);
    eprintln!();
    Ok(())
}

// #region Sweep driver

/// How one config in a sweep ended up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigOutcome {
    /// Backend constructed and the benchmark ran to the end.
    Ran,
    /// Backend construction failed — config is invalid for this engine (e.g. non-default ef on
    /// FAISS binary HNSW).
    Skipped,
    /// Backend ran but the benchmark loop or the save errored mid-flight.
    Failed,
}

impl ConfigOutcome {
    fn skipped(description: &str, error: impl fmt::Display) -> Self {
        eprintln!("\n── {description} — skipped ──\n  {error}");
        Self::Skipped
    }

    fn failed(description: &str, error: impl fmt::Display) -> Self {
        eprintln!("\n── {description} — failed ──\n  {error}");
        Self::Failed
    }
}

/// Runs one opened config, measuring a loaded index once and saving a fresh one when
/// `index_file` asks; failures land in the outcome rather than ending the sweep.
pub fn run_config(
    description: &str,
    index_file: IndexFile<'_>,
    backend: Result<impl Backend, impl fmt::Display>,
    state: &mut BenchState,
    dimensions: usize,
) -> ConfigOutcome {
    let mut backend = match backend {
        Ok(backend) => backend,
        Err(error) => return ConfigOutcome::skipped(description, error),
    };
    let loaded = match index_file {
        IndexFile::Load(_) => Some(backend.indexed_count().unwrap_or(state.total_vectors)),
        IndexFile::None | IndexFile::Save(_) => None,
    };
    if let Err(error) = run(&mut backend, state, description, dimensions, loaded) {
        return ConfigOutcome::failed(description, error);
    }
    let IndexFile::Save(path) = index_file else {
        return ConfigOutcome::Ran;
    };
    match backend.save(path) {
        Ok(()) => {
            eprintln!("  saved index → {path}");
            ConfigOutcome::Ran
        }
        Err(error) => ConfigOutcome::failed(description, error),
    }
}

/// Aggregate outcome counter for a sweep — record per config, report once at the end.
#[derive(Debug, Default, Clone, Copy)]
pub struct SweepSummary {
    pub ran: usize,
    pub skipped: usize,
    pub failed: usize,
}

impl SweepSummary {
    pub fn record(&mut self, outcome: ConfigOutcome) {
        match outcome {
            ConfigOutcome::Ran => self.ran += 1,
            ConfigOutcome::Skipped => self.skipped += 1,
            ConfigOutcome::Failed => self.failed += 1,
        }
    }

    /// Prints the tally and exits with status 1 when a config failed or none ran; servers must
    /// stop their containers first, since the exit skips destructors.
    pub fn finish(self) {
        eprintln!(
            "Benchmark complete: {} config(s) ran, {} skipped, {} failed.",
            self.ran, self.skipped, self.failed
        );
        if self.failed != 0 || self.ran == 0 {
            std::process::exit(1);
        }
    }
}

/// Print a CLI validation error and exit with status 1. Backend binaries call this from their `main()`
/// when `--metric`/`--data-type`/etc. parsing fails — at that point we haven't started the sweep yet, so an
/// early exit is the right shape (vs. a per-config skip).
pub fn bail(message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(1);
}

/// Parses the command line; a bad value prints `--name="value" does not parse, expected ...` and exits with 1.
pub fn parse_cli<T: clap::Parser>() -> T {
    use clap::error::{ContextKind, ContextValue, ErrorKind};

    let error = match T::try_parse() {
        Ok(cli) => return cli,
        Err(error) if !error.use_stderr() => error.exit(),
        Err(error) => error,
    };
    let context = |kind| match error.get(kind) {
        Some(ContextValue::String(text)) => text.as_str(),
        _ => "",
    };
    let expected = match (
        error.kind(),
        std::error::Error::source(&error),
        error.get(ContextKind::ValidValue),
    ) {
        (ErrorKind::ValueValidation, Some(expected), _) => expected.to_string(),
        (ErrorKind::InvalidValue, _, Some(ContextValue::Strings(values))) if !values.is_empty() => {
            format!("expected one of {}", values.join(", "))
        }
        (ErrorKind::InvalidValue, _, _) => "expected a non-empty value".into(),
        _ => {
            eprint!("{error}");
            std::process::exit(1)
        }
    };
    let flag = context(ContextKind::InvalidArg).split(' ').next().unwrap_or_default();
    eprintln!(
        "{flag}=\"{}\" does not parse, {expected}",
        context(ContextKind::InvalidValue)
    );
    std::process::exit(1)
}

/// Parses a 32-bit unsigned integer, or `random` as 32 bits from the OS entropy source.
pub fn parse_seed(text: &str) -> Option<Seed> {
    if text == "random" {
        return Some(Seed(std::hash::RandomState::new().build_hasher().finish() as u32));
    }
    let digits = !text.is_empty() && text.bytes().all(|byte| byte.is_ascii_digit());
    digits.then(|| text.parse().ok().map(Seed)).flatten()
}

/// Parses a thread count like `8`, or `0` as `all_cores`.
pub fn parse_threads(text: &str, all_cores: NonZeroUsize) -> Option<Threads> {
    match text {
        "0" => Some(Threads(all_cores)),
        _ => parse_count(text).and_then(NonZeroUsize::new).map(Threads),
    }
}

/// Parses a positive whole number in ASCII digits, like `128`; zero is `None`.
pub fn parse_count(text: &str) -> Option<usize> {
    let digits = !text.is_empty() && text.bytes().all(|byte| byte.is_ascii_digit());
    digits.then(|| text.parse().ok()).flatten().filter(|&count| count != 0)
}

/// Parses a duration like `200ms` or `10s`; a bare number, a fraction or zero is `None`.
pub fn parse_duration(text: &str) -> Option<Duration> {
    match text.strip_suffix("ms") {
        Some(count) => parse_count(count).map(|count| Duration::from_millis(count as u64)),
        None => parse_count(text.strip_suffix('s')?).map(|count| Duration::from_secs(count as u64)),
    }
}

/// Spells a duration the way `parse_duration` reads it: `1s`, `1500ms`.
pub fn spell_duration(duration: Duration) -> String {
    let milliseconds = duration.as_millis();
    match milliseconds % 1000 {
        0 => format!("{}s", milliseconds / 1000),
        _ => format!("{milliseconds}ms"),
    }
}

/// A 32-bit run seed, an integer or drawn from the OS for `random`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Seed(pub u32);

impl From<Seed> for u64 {
    fn from(seed: Seed) -> u64 {
        u64::from(seed.0)
    }
}

impl fmt::Display for Seed {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

/// A thread count; `0` in the variable resolves to every core when read, so it is never zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Threads(pub NonZeroUsize);

impl Threads {
    pub const ONE: Threads = Threads(NonZeroUsize::MIN);

    /// Every logical core the machine's topology reports, at least one.
    pub fn all_cores() -> Self {
        let cores = Topology::new().and_then(|topology| topology.logical_cores_count());
        Self(cores.ok().and_then(NonZeroUsize::new).unwrap_or(NonZeroUsize::MIN))
    }
}

/// A ForkUnion pool of `threads` workers over the machine's topology.
pub fn spawn_pool(threads: Threads) -> fu::Result<ThreadPool> {
    Topology::new().and_then(|topology| ThreadPool::spawn(&topology, threads.0.get()))
}

impl fmt::Display for Threads {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

/// SplitMix64's increment, the golden ratio in 64 bits.
const SPLITMIX64_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// SplitMix64's finalizer, a bijection that spreads every input bit over the whole output.
pub fn mix(value: u64) -> u64 {
    let value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// The key of stream `index` of `name`, two mixes away from `seed`, so neighboring seeds and indices never meet.
pub fn stream_key(seed: Seed, name: &str, index: u64) -> u64 {
    let hashed = name.bytes().fold(0xCBF2_9CE4_8422_2325, |hash: u64, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01B3)
    });
    mix(mix(u64::from(seed) ^ hashed).wrapping_add(index))
}

/// A SplitMix64 stream seeded with a `stream_key`, bit-identical to the C++ `splitmix64_t`.
pub struct SplitMix64 {
    pub state: u64,
}

impl SplitMix64 {
    /// The next 64 random bits.
    pub fn draw(&mut self) -> u64 {
        self.state = self.state.wrapping_add(SPLITMIX64_GAMMA);
        mix(self.state)
    }

    /// A draw below `bound`: the high half of a draw times `bound`, with a bias of at most `bound` over 2^64.
    pub fn below(&mut self, bound: u64) -> u64 {
        ((u128::from(self.draw()) * u128::from(bound)) >> 64) as u64
    }
}

/// A TCP port; `0` is rejected at parse time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Port(pub NonZeroU16);

impl From<Port> for u16 {
    fn from(port: Port) -> u16 {
        port.0.get()
    }
}

impl fmt::Display for Port {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

/// `parse_count` for clap.
pub fn parse_count_flag(text: &str) -> Result<usize, String> {
    parse_count(text).ok_or_else(|| "expected a positive count".into())
}

/// `parse_threads` for clap, with `0` resolved to every available core.
pub fn parse_threads_flag(text: &str) -> Result<Threads, String> {
    parse_threads(text, Threads::all_cores().0).ok_or_else(|| "expected a count, 0 for all cores".into())
}

/// `parse_seed` for clap.
pub fn parse_seed_flag(text: &str) -> Result<Seed, String> {
    parse_seed(text).ok_or_else(|| "expected an unsigned integer or random".into())
}

/// `parse_duration` for clap.
pub fn parse_duration_flag(text: &str) -> Result<Duration, String> {
    parse_duration(text).ok_or_else(|| "expected a duration like 200ms or 10s".into())
}

/// Parses a TCP port from 1 to 65535.
pub fn parse_port(text: &str) -> Result<Port, String> {
    parse_count(text)
        .and_then(|port| u16::try_from(port).ok())
        .and_then(NonZeroU16::new)
        .map(Port)
        .ok_or_else(|| "expected a port from 1 to 65535".into())
}

/// Writes a `clap::ValueEnum` value the way its flag spells it, for `Display` impls.
pub fn spell_value<T: clap::ValueEnum>(value: &T, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    let possible = value.to_possible_value().expect("every variant has a name");
    formatter.write_str(possible.get_name())
}

/// Spells a list the way a comma-separated sweep flag reads it: `16,32`.
pub fn spell_list<T: fmt::Display>(values: impl IntoIterator<Item = T>) -> String {
    let mut spelled = String::new();
    for (index, value) in values.into_iter().enumerate() {
        let separator = if index == 0 { "" } else { "," };
        fmt::Write::write_fmt(&mut spelled, format_args!("{separator}{value}")).expect("a String takes any write");
    }
    spelled
}

/// Sugar for the recurring `result.unwrap_or_else(|e| bail(&format!("{prefix}: {e}")))`
/// pattern that flows up out of `BenchState::load`, `state.check_dimensions(...)`,
/// and the various backend constructors. The error variant is rendered with
/// `Display` so anything that satisfies the trait works without ceremony.
pub trait UnwrapOrBail<T> {
    /// Unwrap the success variant or bail with `<prefix>: <error>`.
    fn unwrap_or_bail(self, prefix: &str) -> T;
}

impl<T, E: std::fmt::Display> UnwrapOrBail<T> for Result<T, E> {
    fn unwrap_or_bail(self, prefix: &str) -> T {
        match self {
            Ok(value) => value,
            Err(error) => bail(&format!("{prefix}: {error}")),
        }
    }
}

// #region Tests

#[cfg(test)]
mod tests {
    use std::alloc::System;

    use super::{
        parse_self_search_sample, resolve_self_search_sample, CastScratch, Distance, Key, SearchBuffers,
        SelfSearchSample, SplitMix64, VectorSlice, Vectors,
    };

    #[test]
    fn hit_rows_pad_count_and_refuse_overflow() {
        let mut buffers = SearchBuffers::new(2, 3).unwrap();
        {
            let mut hits = buffers.hits(0..2);
            let mut rows = hits.rows();
            rows.next().unwrap().write([Ok((7, 0.5)), Ok((9, 1.5))]).unwrap();
            let overflow = rows.next().unwrap().write((0..4).map(|key| Ok((key, 0.0))));
            assert!(overflow.is_err());
        }
        let (keys, distances, found) = buffers.rows().next().unwrap();
        assert_eq!(keys, [7, 9, Key::MAX]);
        assert_eq!(distances, [0.5, 1.5, Distance::INFINITY]);
        assert_eq!(found, 2);
        assert_eq!(buffers.found().next().unwrap(), [7, 9]);
    }

    #[test]
    fn conversions_reuse_allocations_and_pass_matching_types_through() {
        let mut scratch = Vec::new_in(System);
        let signed = Vectors::new(VectorSlice::I8(&[-2, 3]), 2);
        assert_eq!(signed.to_f32_in(&mut scratch).unwrap(), [-2.0, 3.0]);
        let pointer = scratch.as_ptr();
        let unsigned = Vectors::new(VectorSlice::U8(&[7]), 1);
        assert_eq!(unsigned.to_f32_in(&mut scratch).unwrap(), [7.0]);
        assert_eq!(pointer, scratch.as_ptr());
        let source = [2.0, 4.0];
        let floats = Vectors::new(VectorSlice::F32(&source), 2);
        assert_eq!(floats.to_f32_in(&mut scratch).unwrap().as_ptr(), source.as_ptr());
        let mut wire = CastScratch::F32(Vec::new_in(System));
        let pointer = wire
            .encode(&Vectors::new(VectorSlice::I8(&[1, -2, 3]), 3))
            .unwrap()
            .as_ptr();
        assert_eq!(
            wire.encode(&Vectors::new(VectorSlice::I8(&[4, 5]), 2))
                .unwrap()
                .as_ptr(),
            pointer
        );
        let bytes = [9u8, 8];
        let mut bytes_wire = CastScratch::U8(Vec::new_in(System));
        let unsigned = Vectors::new(VectorSlice::U8(&bytes), 2);
        let passed = bytes_wire.encode(&unsigned).unwrap();
        assert_eq!(passed.as_ptr(), bytes.as_ptr());
    }

    #[test]
    fn self_recall_sample_none_is_all() {
        assert_eq!(resolve_self_search_sample(None, 1000), 1000);
    }

    #[test]
    fn self_recall_sample_absolute_is_clamped() {
        let resolve = |spec| resolve_self_search_sample(Some(parse_self_search_sample(spec).unwrap()), 1000);
        assert_eq!(resolve("250"), 250);
        assert_eq!(resolve("5000"), 1000);
    }

    #[test]
    fn self_recall_sample_fraction_rounds_and_floors_at_one() {
        let resolve = |spec| resolve_self_search_sample(Some(parse_self_search_sample(spec).unwrap()), 1000);
        assert_eq!(resolve("0.1"), 100);
        assert_eq!(resolve("1.0"), 1000);
        // A fraction so small it would round to zero still yields one vector.
        assert_eq!(resolve("0.0001"), 1);
    }

    #[test]
    fn self_recall_sample_one_vs_one_point_zero_disambiguate() {
        // `1` is an absolute count; `1.0` is the whole base.
        assert_eq!(parse_self_search_sample("1").unwrap(), SelfSearchSample::Absolute(1));
        assert_eq!(
            parse_self_search_sample("1.0").unwrap(),
            SelfSearchSample::Fraction(1.0)
        );
        assert_eq!(SelfSearchSample::Fraction(1.0).to_string(), "1.0");
    }

    #[test]
    fn self_recall_sample_rejects_zero_signs_and_out_of_range() {
        for spec in ["0", "0.0", "1.5", "abc", "+5", "+0.5", "-0.5", "1e-3"] {
            assert!(parse_self_search_sample(spec).is_err(), "{spec}");
        }
    }

    #[test]
    fn splitmix64_matches_the_shared_test_vector() {
        assert_eq!(SplitMix64 { state: 42 }.draw(), 0xbdd7_3226_2feb_6e95);
    }
}
