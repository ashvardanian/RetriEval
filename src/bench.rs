//! Shared benchmark infrastructure for vector search engines.
//!
//! This is the library root. Backend binaries (`usearch.rs`, `faiss.rs`, etc.)
//! import from here and provide their own `main()`.

#![feature(btreemap_alloc)]

// The generator is also compiled as a standalone binary.
extern crate self as retrieval;

pub mod dataset;
#[cfg(feature = "tier2")]
pub mod docker;
pub mod error;
pub mod eval;
#[cfg(any(feature = "generate", feature = "download"))]
pub mod generate;
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
    path::PathBuf,
    time::{Duration, Instant},
};

use indicatif::{ProgressBar, ProgressStyle};
use serde_json::Value;

#[cfg(feature = "download")]
pub use crate::error::DownloadError;
pub use crate::{
    dataset::{Dataset, GroundTruth, Keys},
    error::{DatasetError, GroundTruthError, PerfCountersError},
    output::{
        collect_machine_info, config_hash, write_report, ConfigReport, DatasetInfo, MachineInfo, PhaseCounters,
        StepAddEntry, StepEntry, StepSearchEntry,
    },
};

// #region Core types

/// Vector key type used throughout the benchmark.
pub type Key = u32;

/// Distance/similarity value returned by search operations.
pub type Distance = f32;

// #region Vector types

/// Borrowed batch of row-major vectors with known dimensionality.
pub struct Vectors<'a> {
    pub data: VectorSlice<'a>,
    pub dimensions: usize,
}

pub enum VectorSlice<'a> {
    F32(&'a [f32]),
    I8(&'a [i8]),
    U8(&'a [u8]),
    /// Binary vectors: 1-bit values packed 8 per byte. Dimensions is in bits.
    B1x8(&'a [u8]),
}

impl VectorSlice<'_> {
    pub fn to_f32_in<'a, A: Allocator>(&'a self, scratch: &'a mut Vec<f32, A>) -> Result<&'a [f32], &'static str> {
        if let Self::F32(values) = self {
            return Ok(values);
        }
        scratch.clear();
        match self {
            Self::I8(values) => scratch.extend(values.iter().map(|&v| v as f32)),
            Self::U8(values) => scratch.extend(values.iter().map(|&v| v as f32)),
            Self::B1x8(_) => return Err("Packed binary input cannot be converted to dense F32 without a bit metric"),
            Self::F32(_) => unreachable!(),
        }
        Ok(scratch)
    }
}

impl Vectors<'_> {
    pub fn len(&self) -> usize {
        let dimensions = self.dimensions;
        match self.data {
            VectorSlice::F32(data) => data.len() / dimensions,
            VectorSlice::I8(data) => data.len() / dimensions,
            VectorSlice::U8(data) => data.len() / dimensions,
            VectorSlice::B1x8(data) => data.len() / dimensions.div_ceil(8),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Get the current process resident set size (RSS) in bytes.
pub fn process_rss_bytes() -> u64 {
    let pid = sysinfo::get_current_pid().expect("get current pid");
    let mut sys = sysinfo::System::new();
    sys.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
    sys.process(pid).map(|p| p.memory()).unwrap_or(0)
}

// #region Backend trait

/// Common trait for all vector search backends.
pub trait Backend: Send {
    fn description(&self) -> String;
    fn metadata(&self) -> HashMap<String, Value>;
    fn add(&mut self, keys: &[Key], vectors: Vectors) -> Result<(), String>;
    fn search(
        &mut self,
        queries: Vectors,
        count: usize,
        out_keys: &mut [Key],
        out_distances: &mut [Distance],
        out_counts: &mut [usize],
    ) -> Result<(), String>;
    fn memory_bytes(&self) -> usize;

    /// How many vectors the backend actually holds, when it can say.
    ///
    /// The harness otherwise infers this from the dataset, which is wrong on
    /// the `--index` load path: a saved index may have been built over a
    /// different slice of the base than the current run's `--max-base-vectors`
    /// implies. Self-recall depends on the answer — it queries base rows and
    /// asserts each one is in the index — so a backend that knows its own
    /// population should report it. `None` means "no idea, trust the dataset".
    fn indexed_count(&self) -> Option<usize> {
        None
    }

    /// Persist the index under `handle`. For embedded backends `handle` is a
    /// filesystem path; for server-style backends it's a collection / table /
    /// index name. Default returns Err so backends opt in by overriding.
    fn save(&self, _handle: &str) -> Result<(), String> {
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

// #region Result buffers

/// Write one query's hits into its slice of the output buffers and return how
/// many landed. Hits are compacted to the front and the tail is padded with
/// `Key::MAX` / `Distance::INFINITY`.
///
/// `eval` scores `out_keys[..found_count]`, so a backend that leaves a hole at
/// a non-terminal rank pushes real hits outside the scanned window. Backends
/// filter their engine's misses out of `hits` rather than yielding sentinels.
pub fn write_row(
    hits: impl IntoIterator<Item = (Key, Distance)>,
    out_keys: &mut [Key],
    out_distances: &mut [Distance],
) -> usize {
    let count = out_keys.len().min(out_distances.len());
    let mut found = 0;
    for (key, distance) in hits.into_iter().take(count) {
        out_keys[found] = key;
        out_distances[found] = distance;
        found += 1;
    }
    for slot in found..count {
        out_keys[slot] = Key::MAX;
        out_distances[slot] = Distance::INFINITY;
    }
    found
}

/// Reusable owned search storage; adapters receive only borrowed output slices.
pub struct SearchBuffers<A: Allocator = System> {
    pub keys: Vec<Key, A>,
    pub distances: Vec<Distance, A>,
    pub counts: Vec<usize, A>,
}
impl<A: Allocator + Clone> SearchBuffers<A> {
    pub fn new_in(allocator: A) -> Self {
        Self {
            keys: Vec::new_in(allocator.clone()),
            distances: Vec::new_in(allocator.clone()),
            counts: Vec::new_in(allocator),
        }
    }
    pub fn resize(&mut self, queries: usize, count: usize) {
        self.keys.resize(queries * count, Key::MAX);
        self.distances.resize(queries * count, Distance::INFINITY);
        self.counts.resize(queries, 0);
    }
}

// #region Utilities

/// Reinterpret a `&[T]` as the raw bytes that back it, without copying.
///
/// Used to splat flat numeric buffers (ground-truth indices, f32 base vectors,
/// etc.) into the BigANN `.fbin` / `.ibin` file format without re-encoding.
///
/// # Safety
///
/// `T` must be "plain old data" in the C/bytemuck sense: no padding, no
/// destructor, no invalid bit patterns. `u32`, `i32`, `u64`, `f32`, `f64`,
/// `u8`, `i8` all qualify. Do not use with types containing references,
/// `bool`, or anything `Drop`.
pub unsafe fn pod_slice_as_bytes<T: Copy>(slice: &[T]) -> &[u8] {
    std::slice::from_raw_parts(slice.as_ptr().cast::<u8>(), std::mem::size_of_val(slice))
}

/// Format a number with thousand separators: 1234567 → "1,234,567"
pub fn format_thousands(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (char_index, character) in s.chars().enumerate() {
        if char_index > 0 && (s.len() - char_index).is_multiple_of(3) {
            result.push(',');
        }
        result.push(character);
    }
    result
}

// #region Common CLI args

/// Shared CLI arguments for all backends.
#[derive(clap::Args, Debug)]
pub struct CommonArgs {
    /// Base vectors to index (.fbin, .u8bin, .i8bin, .b1bin)
    #[arg(long)]
    pub base_vectors: PathBuf,

    /// Optional keys for the base vectors (.i32bin). One per base vector.
    #[arg(long)]
    pub base_keys: Option<PathBuf>,

    /// Query vectors to search with
    #[arg(long)]
    pub query_vectors: PathBuf,

    /// Ground-truth neighbors for those queries (.ibin)
    #[arg(long)]
    pub query_neighbors: PathBuf,

    /// Neighbors to request per query — the k every metric is taken at.
    /// Defaults to the ground-truth file's width, and may not exceed it.
    #[arg(long, value_parser = parse_count_flag)]
    pub top_k: Option<usize>,

    /// Order in which base vectors are inserted: `shuffled` by `--seed`, or the file's `original` order
    #[arg(long, value_enum, default_value = "shuffled")]
    pub insertion_order: InsertionOrder,

    /// Seed of the shuffled insertion order, or `random` to draw one; 42 when unset
    #[arg(long, value_parser = parse_seed_flag)]
    pub seed: Option<Seed>,

    /// Number of measurement steps (dataset is split into this many equal parts)
    #[arg(long, default_value_t = 10, value_parser = parse_count_flag)]
    pub steps: usize,

    /// Vectors per backend add() call
    #[arg(long, default_value_t = 10_000, value_parser = parse_count_flag)]
    pub vectors_per_add: usize,

    /// Queries per backend search() call
    #[arg(long, default_value_t = 10_000, value_parser = parse_count_flag)]
    pub queries_per_search: usize,

    /// Output directory for JSON result files
    #[arg(long)]
    pub output: Option<PathBuf>,

    /// Cap the number of base vectors used (for calibration on a slice of a larger file).
    /// Queries and ground truth are unaffected; only the add/permutation range shrinks.
    #[arg(long, value_parser = parse_count_flag)]
    pub max_base_vectors: Option<usize>,

    /// Persisted-index handle. For embedded backends (USearch, FAISS, cuVS): a
    /// filesystem path — if it exists the backend loads it and skips the add
    /// phase, otherwise the backend builds and saves to it. For server-style
    /// backends: a collection / table / index name (not yet implemented).
    /// Requires a single-config sweep — multi-valued sweep axes are rejected
    /// at startup when `--index` is set.
    #[arg(long, value_parser = clap::builder::NonEmptyStringValueParser::new())]
    pub index: Option<String>,

    /// Replay indexed vectors as their own queries and report the fraction that
    /// retrieve themselves within the top-`--self-search-top-k` ("self-recall").
    /// Runs once, after the last insertion, and needs no ground truth — a vector
    /// in the index is its own nearest neighbor. Because it can sweep the whole
    /// base rather than a short query file, it is also the only phase that
    /// measures throughput under sustained load.
    #[arg(long, default_value_t = false)]
    pub self_search: bool,

    /// Neighbors requested per query during the self-recall sweep (the `k` in
    /// self-recall@k). No effect without `--self-search`.
    #[arg(long, default_value_t = 10, value_parser = parse_count_flag)]
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
    #[arg(long, value_name = "N|FRACTION", value_parser = parse_self_search_sample, requires = "self_search")]
    pub self_search_sample: Option<SelfSearchSample>,

    /// Matryoshka-style embedding-dimension truncations to evaluate
    /// (comma-separated). Empty → use the file's native dimensions. Each value must
    /// be ≤ the native dimensions; for `.b1bin` files each must be a multiple of 8.
    #[arg(long, value_delimiter = ',', value_parser = parse_count_flag)]
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

    /// When `--index` is set, the sweep must collapse to a single config —
    /// otherwise multiple builds would write to (or load from) the same file.
    /// Pass the cardinality of every sweep axis (`dimensions_sweep.len()`,
    /// `cli.data_type.len()`, …) and this exits with a clear message if their
    /// product exceeds one. No-op when `--index` is absent.
    pub fn ensure_single_config(&self, axis_lengths: &[usize]) {
        if self.index.is_none() {
            return;
        }
        let cardinality: usize = axis_lengths.iter().product();
        if cardinality > 1 {
            bail(&format!(
                "--index requires a single config; got {cardinality} configs from the sweep"
            ));
        }
    }
}

// #region BenchState

/// Pre-loaded benchmark state. Call `BenchState::load()` once, then `run()` per configuration.
pub struct BenchState {
    pub dataset: Dataset,
    /// Effective base-vector count; equals `dataset.rows()` unless `--max-base-vectors` capped it.
    pub total_vectors: usize,
    pub keys: Keys,
    pub query_dataset: Dataset,
    pub ground_truth: GroundTruth,
    pub perm: dataset::Permutation,
    /// Seed of the shuffled insertion order, `None` for the original order.
    pub seed: Option<Seed>,
    /// Resolved `--top-k`: the search width and the k every metric uses.
    pub top_k: usize,
    pub steps: usize,
    pub vectors_per_add: usize,
    pub queries_per_search: usize,
    pub self_search: bool,
    pub self_search_top_k: usize,
    pub self_search_sample: Option<SelfSearchSample>,
    pub output_dir: Option<PathBuf>,
    pub machine_info: MachineInfo,
    pub dataset_info: DatasetInfo,
    search_buffers: SearchBuffers,
    key_scratch: Vec<Key, System>,
    /// Shared scratch for `Dataset::gather` (during add) and `Dataset::slice`
    /// (during search). Add and search are sequential within a step, so a
    /// single buffer sized at the upper bound is enough. Sized to fit
    /// `max(vectors_per_add, queries_per_search) * native_vector_bytes` —
    /// covers any truncation since truncated bytes ≤ native.
    scratch_buf: Vec<u8, System>,
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

        // Create output directory if specified
        if let Some(dir) = &args.output {
            if dir.extension().is_some_and(|ext| ext == "json" || ext == "jsonl") {
                return Err(format!(
                    "--output should be a directory, not a file: {}. \
                     Each config produces its own JSON file inside this directory.",
                    dir.display()
                )
                .into());
            }
            std::fs::create_dir_all(dir)?;
        }

        let machine_info = collect_machine_info();

        eprintln!("Loading dataset: {}", args.base_vectors.display());
        let dataset = Dataset::load(&args.base_vectors)?;
        let file_vectors = dataset.rows();
        let total_vectors = match args.max_base_vectors {
            Some(cap) => cap.min(file_vectors),
            None => file_vectors,
        };
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
                let k = Keys::load(path)?;
                eprintln!("  {} keys", format_thousands(k.rows() as u64));
                // One key per base vector is a hard requirement: the add loop
                // indexes keys by base row, so a short file reads past the end
                // of the mapping (a bounds panic at best, a garbage key at
                // worst, since the check is only a `debug_assert`).
                if k.rows() < total_vectors {
                    return Err(format!(
                        "--base-keys has {} entries but {} base vectors are in play; \
                         one key per vector is required",
                        format_thousands(k.rows() as u64),
                        format_thousands(total_vectors as u64),
                    )
                    .into());
                }
                k
            }
            None => Keys::sequential(total_vectors),
        };

        eprintln!("Loading queries: {}", args.query_vectors.display());
        let query_dataset = Dataset::load(&args.query_vectors)?;
        let num_queries = query_dataset.rows();
        eprintln!(
            "  {} queries, {} dimensions",
            format_thousands(num_queries as u64),
            query_dataset.dimensions()
        );

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

        eprintln!("- Insertion order: {}", args.insertion_order);
        if let Some(seed) = seed {
            eprintln!("- Seed: {seed}");
        }
        eprintln!("- Top k: {top_k}");
        eprintln!("- Dims: {}", spell_list(&args.dimensions_sweep(dimensions)));
        eprintln!("- Steps: {}", args.steps);
        eprintln!("- Vectors per add: {}", args.vectors_per_add);
        eprintln!("- Queries per search: {}", args.queries_per_search);
        match args.max_base_vectors {
            Some(count) => eprintln!("- Max base vectors: {count}"),
            None => eprintln!("- Max base vectors: all"),
        }
        eprintln!("- Index: {}", args.index.as_deref().unwrap_or("none"));
        eprintln!("- Self-search: {}", args.self_search);
        if args.self_search {
            eprintln!("- Self-search top k: {}", args.self_search_top_k);
            match args.self_search_sample {
                Some(sample) => eprintln!("- Self-search sample: {sample}"),
                None => eprintln!("- Self-search sample: all"),
            }
        }
        match &args.output {
            Some(path) => eprintln!("- Output: {}", path.display()),
            None => eprintln!("- Output: none"),
        }

        let perm = match seed {
            None => dataset::Permutation::identity(total_vectors),
            Some(seed) => dataset::Permutation::shuffled(total_vectors, seed),
        };

        let dataset_info = DatasetInfo {
            base_vectors_path: args.base_vectors.display().to_string(),
            query_vectors_path: args.query_vectors.display().to_string(),
            query_neighbors_path: args.query_neighbors.display().to_string(),
            vectors_count: file_vectors,
            queries_count: num_queries,
            dimensions,
            neighbors_per_query: ground_truth_width,
        };

        Ok(Self {
            total_vectors,
            perm,
            seed,
            top_k,
            steps: args.steps,
            vectors_per_add: args.vectors_per_add,
            queries_per_search: args.queries_per_search,
            self_search: args.self_search,
            self_search_top_k: args.self_search_top_k,
            self_search_sample: args.self_search_sample,
            output_dir: args.output.clone(),
            machine_info,
            dataset_info,
            search_buffers: {
                let mut buffers = SearchBuffers::new_in(System);
                buffers.resize(num_queries, top_k);
                buffers
            },
            key_scratch: {
                let mut values = Vec::new_in(System);
                values.resize(args.vectors_per_add, 0 as Key);
                values
            },
            scratch_buf: {
                let max_batch = args.vectors_per_add.max(args.queries_per_search);
                let max_row_bytes = dataset.vector_bytes().max(query_dataset.vector_bytes());
                {
                    let mut values = Vec::new_in(System);
                    values.resize(max_batch * max_row_bytes, 0u8);
                    values
                }
            },
            dataset,
            keys,
            query_dataset,
            ground_truth,
        })
    }

    /// Native dimensions of the base dataset. Truncation is per-call: see
    /// `run` / `run_search_only`'s `dimensions` argument.
    pub fn dimensions(&self) -> usize {
        self.dataset.dimensions()
    }

    /// Validate that `dimensions` is a legal exposure for both base and query
    /// datasets. Used by binaries before kicking off a per-config run.
    pub fn check_dimensions(&self, dimensions: usize) -> Result<(), DatasetError> {
        self.dataset.check_dimensions(dimensions)?;
        self.query_dataset.check_dimensions(dimensions)?;
        Ok(())
    }
}

// #region Benchmark loop

type BenchResult<T> = Result<T, Box<dyn std::error::Error>>;

/// Outcome of one step's add phase.
struct AddPhaseOutcome {
    elapsed_secs: f64,
    throughput_per_sec: u64,
    counter_sample: Option<perf_counters::CounterSample>,
}

/// Arm perf counters before a phase; on failure disarm the whole capture so
/// later phases don't try again. Single place for the error-logging wording.
fn start_counter_capture(perf: &mut Option<perf_counters::PerfCounters>, phase: &str) {
    if let Some(pc) = perf.as_mut() {
        if let Err(err) = pc.reset_and_enable() {
            eprintln!("  perf counters: reset/enable before {phase} failed ({err})");
            *perf = None;
        }
    }
}

/// Read a sample at the end of a phase; same disarm-on-error policy.
fn stop_and_read_counters(
    perf: &mut Option<perf_counters::PerfCounters>,
    phase: &str,
) -> Option<perf_counters::CounterSample> {
    let pc = perf.as_mut()?;
    match pc.disable_and_read() {
        Ok(sample) => Some(sample),
        Err(err) => {
            eprintln!("  perf counters: read after {phase} failed ({err})");
            *perf = None;
            None
        }
    }
}

/// The slice of the base one step inserts, at the dimensionality this config
/// exposes to the backend.
struct StepSlice {
    start: usize,
    count: usize,
    dimensions: usize,
}

/// Insert one step's slice in `vectors_per_add` chunks, capturing perf counters
/// across the phase. Advances `vectors_indexed` to the new cumulative count.
fn run_add_phase(
    backend: &mut dyn Backend,
    state: &mut BenchState,
    perf: &mut Option<perf_counters::PerfCounters>,
    progress_style: &ProgressStyle,
    slice: StepSlice,
    vectors_indexed: &mut usize,
) -> BenchResult<AddPhaseOutcome> {
    let StepSlice {
        start: step_start,
        count: step_count,
        dimensions,
    } = slice;
    let total_vectors = state.total_vectors;
    let vectors_per_add = state.vectors_per_add;

    let progress = ProgressBar::new(total_vectors as u64);
    progress.set_style(progress_style.clone());
    progress.set_position(*vectors_indexed as u64);

    start_counter_capture(perf, "add");
    let add_start = Instant::now();
    let mut added = 0;
    while added < step_count {
        let batch = vectors_per_add.min(step_count - added);
        let logical_offset = step_start + added;
        let indices = state.perm.range(logical_offset, batch);

        for (slot, &row_index) in indices.iter().enumerate() {
            state.key_scratch[slot] = state.keys.get(row_index);
        }
        let batch_keys = &state.key_scratch[..batch];
        let vectors = state.dataset.gather(indices, dimensions, &mut state.scratch_buf);

        backend.add(batch_keys, vectors)?;
        added += batch;
        *vectors_indexed += batch;
        let elapsed = add_start.elapsed().as_secs_f64();
        let throughput = if elapsed > 0.0 {
            (added as f64 / elapsed) as u64
        } else {
            0
        };
        progress.set_position(*vectors_indexed as u64);
        progress.set_message(format!(
            "{}/{} ({} add/s)",
            format_thousands(*vectors_indexed as u64),
            format_thousands(total_vectors as u64),
            format_thousands(throughput),
        ));
    }
    let elapsed_secs = add_start.elapsed().as_secs_f64();
    let throughput_per_sec = if elapsed_secs > 0.0 {
        (step_count as f64 / elapsed_secs) as u64
    } else {
        0
    };
    let counter_sample = stop_and_read_counters(perf, "add");
    progress.set_message(format!(
        "{}/{} ({} add/s)",
        format_thousands(*vectors_indexed as u64),
        format_thousands(total_vectors as u64),
        format_thousands(throughput_per_sec),
    ));
    progress.finish();

    Ok(AddPhaseOutcome {
        elapsed_secs,
        throughput_per_sec,
        counter_sample,
    })
}

/// Run all queries against the current index and score them. `vectors_indexed`
/// only affects normalization; `is_final_step` drops the `~` from the progress
/// line once the index is complete.
fn run_search_phase(
    backend: &mut dyn Backend,
    state: &mut BenchState,
    perf: &mut Option<perf_counters::PerfCounters>,
    progress_style: &ProgressStyle,
    dimensions: usize,
    vectors_indexed: usize,
    is_final_step: bool,
) -> BenchResult<StepSearchEntry> {
    let num_queries = state.query_dataset.rows();
    let count = state.top_k;

    let progress = ProgressBar::new(num_queries as u64);
    progress.set_style(progress_style.clone());
    progress.set_position(0);

    start_counter_capture(perf, "search");
    let search_start = Instant::now();
    let mut searched = 0usize;
    while searched < num_queries {
        let batch_rows = state.queries_per_search.min(num_queries - searched);
        let batch_queries = state
            .query_dataset
            .slice(searched, batch_rows, dimensions, &mut state.scratch_buf);
        let key_offset = searched * count;
        let key_end = key_offset + batch_rows * count;

        backend.search(
            batch_queries,
            count,
            &mut state.search_buffers.keys[key_offset..key_end],
            &mut state.search_buffers.distances[key_offset..key_end],
            &mut state.search_buffers.counts[searched..searched + batch_rows],
        )?;

        searched += batch_rows;
        let elapsed = search_start.elapsed().as_secs_f64();
        let throughput = if elapsed > 0.0 {
            (searched as f64 / elapsed) as u64
        } else {
            0
        };
        progress.set_position(searched as u64);
        progress.set_message(format!(
            "{}/{} ({} search/s)",
            format_thousands(searched as u64),
            format_thousands(num_queries as u64),
            format_thousands(throughput),
        ));
    }
    let elapsed_secs = search_start.elapsed().as_secs_f64();
    let counter_sample = stop_and_read_counters(perf, "search");

    let throughput_per_sec = if elapsed_secs > 0.0 {
        (num_queries as f64 / elapsed_secs) as u64
    } else {
        0
    };
    let recall_at_1 = eval::recall_at_k(
        &state.search_buffers.keys,
        &state.search_buffers.counts,
        count,
        &state.ground_truth,
        1,
    );
    let recall_at_k = eval::recall_at_k(
        &state.search_buffers.keys,
        &state.search_buffers.counts,
        count,
        &state.ground_truth,
        count,
    );
    let intersection_at_k = eval::intersection_at_k(
        &state.search_buffers.keys,
        &state.search_buffers.counts,
        count,
        &state.ground_truth,
        count,
    );
    let ndcg_at_k = eval::ndcg_at_k(
        &state.search_buffers.keys,
        &state.search_buffers.counts,
        count,
        &state.ground_truth,
        count,
    );

    // Metrics are raw. A step that holds only part of the base cannot find the
    // ground truth it does not hold, so the coverage share is printed alongside
    // rather than divided out — that correction assumes the indexed slice is a
    // uniform sample, which a leading prefix is not.
    let coverage = vectors_indexed as f64 / state.dataset.rows().max(1) as f64;
    let approx = if is_final_step { "" } else { "~" };
    progress.finish_with_message(format!(
        "{} search/s, {approx}recall@1={recall_at_1:.4}, \
         {approx}recall@{count}={recall_at_k:.4}, \
         {approx}intersection@{count}={intersection_at_k:.4}, \
         {approx}NDCG@{count}={ndcg_at_k:.4} ({} vectors, {:.0}% coverage)",
        format_thousands(throughput_per_sec),
        format_thousands(vectors_indexed as u64),
        coverage * 100.0,
    ));

    Ok(StepSearchEntry {
        queries: num_queries,
        top_k: count,
        elapsed: elapsed_secs,
        throughput: throughput_per_sec,
        recall_at_1,
        recall_at_k,
        intersection_at_k: Some(intersection_at_k),
        ndcg_at_k: Some(ndcg_at_k),
        counters: PhaseCounters::from_sample(counter_sample.as_ref()),
    })
}

/// Replay every indexed vector as its own query: a vector that is in the index
/// is its own nearest neighbor, so the top-`count` result for row `i` must
/// contain `keys[i]`. Identity is the ground truth, which is what makes this
/// runnable on any dataset — including ones that ship without neighbor files.
///
/// Results are scored batch by batch and discarded rather than accumulated:
/// the whole base at 100M vectors and `count = 10` would otherwise need ~9 GB
/// of host buffers to hold an answer we reduce to a single counter. Requesting
/// `count` neighbors also yields self-recall@1 for free, so both are reported.
fn run_self_search_phase(
    backend: &mut dyn Backend,
    state: &mut BenchState,
    perf: &mut Option<perf_counters::PerfCounters>,
    progress_style: &ProgressStyle,
    count: usize,
    num_queries: usize,
    dimensions: usize,
) -> BenchResult<StepSearchEntry> {
    let batch_size = state.queries_per_search.min(num_queries).max(1);
    let mut batch_keys = {
        let mut values = Vec::new_in(System);
        values.resize(batch_size * count, 0 as Key);
        values
    };
    let mut batch_distances = {
        let mut values = Vec::new_in(System);
        values.resize(batch_size * count, 0.0 as Distance);
        values
    };
    let mut batch_counts = {
        let mut values = Vec::new_in(System);
        values.resize(batch_size, 0usize);
        values
    };
    let mut expected_scratch = {
        let mut values = Vec::new_in(System);
        values.resize(batch_size, 0 as Key);
        values
    };

    let progress = ProgressBar::new(num_queries as u64);
    progress.set_style(progress_style.clone());
    progress.set_position(0);

    start_counter_capture(perf, "self-recall");
    let self_search_start = Instant::now();
    let mut searched = 0usize;
    let mut hits_at_1 = 0usize;
    let mut hits_at_k = 0usize;
    while searched < num_queries {
        let batch_rows = batch_size.min(num_queries - searched);
        let expected = state.keys.slice(searched, batch_rows, &mut expected_scratch);
        let batch_queries = state
            .dataset
            .slice(searched, batch_rows, dimensions, &mut state.scratch_buf);
        let key_end = batch_rows * count;

        backend.search(
            batch_queries,
            count,
            &mut batch_keys[..key_end],
            &mut batch_distances[..key_end],
            &mut batch_counts[..batch_rows],
        )?;

        let counts = &batch_counts[..batch_rows];
        hits_at_1 += eval::self_recall_at_k(&batch_keys, counts, count, expected, 1);
        hits_at_k += eval::self_recall_at_k(&batch_keys, counts, count, expected, count);

        searched += batch_rows;
        let elapsed = self_search_start.elapsed().as_secs_f64();
        let throughput = if elapsed > 0.0 {
            (searched as f64 / elapsed) as u64
        } else {
            0
        };
        progress.set_position(searched as u64);
        progress.set_message(format!(
            "{}/{} ({} search/s)",
            format_thousands(searched as u64),
            format_thousands(num_queries as u64),
            format_thousands(throughput),
        ));
    }
    let elapsed = self_search_start.elapsed().as_secs_f64();
    let counter_sample = stop_and_read_counters(perf, "self-recall");

    let throughput = if elapsed > 0.0 {
        (num_queries as f64 / elapsed) as u64
    } else {
        0
    };
    let recall_at_1 = hits_at_1 as f64 / num_queries as f64;
    let recall_at_k = hits_at_k as f64 / num_queries as f64;

    progress.finish_with_message(format!(
        "{} search/s, recall@1={recall_at_1:.4}, recall@{count}={recall_at_k:.4} ({} queries)",
        format_thousands(throughput),
        format_thousands(num_queries as u64),
    ));

    Ok(StepSearchEntry {
        queries: num_queries,
        top_k: count,
        elapsed,
        throughput,
        recall_at_1,
        recall_at_k,
        // Identity truth is a single key per query, so neither a set overlap
        // nor a ranked gain says anything beyond `recall_at_1`.
        intersection_at_k: None,
        ndcg_at_k: None,
        counters: PhaseCounters::from_sample(counter_sample.as_ref()),
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

/// Resolve how many vectors the self-recall sweep should replay, given how many
/// are actually in the index. `None` when `--self-search` was not passed.
fn self_search_query_count(state: &BenchState, vectors_indexed: usize) -> Option<usize> {
    if !state.self_search {
        return None;
    }
    let resolved = resolve_self_search_sample(state.self_search_sample, vectors_indexed);
    (resolved > 0).then_some(resolved)
}

/// Progress-bar style for the self-recall sweep.
fn self_search_style() -> ProgressStyle {
    ProgressStyle::default_bar()
        .template("  self   [{elapsed_precise}] {bar:40.magenta/blue} {msg}")
        .unwrap()
        .progress_chars("##-")
}

/// Assemble a `ConfigReport` and write it to `<output_dir>/<backend>-<hash>.json`.
/// No-op when the state has no output directory configured.
fn save_report(
    state: &BenchState,
    mut metadata: HashMap<String, Value>,
    dimensions: usize,
    steps: Vec<StepEntry>,
) -> BenchResult<()> {
    let Some(dir) = &state.output_dir else {
        return Ok(());
    };

    // Every harness-level flag that changes what was measured belongs in the
    // hash, centrally: leaving it to each backend is how `--dims` came to
    // be missing on two of them, silently overwriting result files. Backends
    // contribute only their engine-specific knobs.
    metadata.insert("dimensions".into(), Value::from(dimensions));
    metadata.insert("vectors_count".into(), Value::from(state.total_vectors));
    metadata.insert("seed".into(), Value::from(state.seed.map(|seed| seed.0)));
    metadata.insert("steps".into(), Value::from(state.steps));
    metadata.insert("vectors_per_add".into(), Value::from(state.vectors_per_add));
    metadata.insert("queries_per_search".into(), Value::from(state.queries_per_search));
    metadata.insert("top_k".into(), Value::from(state.top_k));
    if let Some(entry) = steps.last().and_then(|step| step.self_search.as_ref()) {
        metadata.insert("self_search_top_k".into(), Value::from(entry.top_k));
        metadata.insert("self_search_sample".into(), Value::from(entry.queries));
    }
    let backend_name = metadata.get("backend").and_then(|v| v.as_str()).unwrap_or("unknown");
    let hash = config_hash(&metadata);
    let filename = format!("{backend_name}-{hash}.json");
    let path = dir.join(&filename);

    let report = ConfigReport {
        machine: collect_machine_info(),
        dataset: DatasetInfo {
            base_vectors_path: state.dataset_info.base_vectors_path.clone(),
            query_vectors_path: state.dataset_info.query_vectors_path.clone(),
            query_neighbors_path: state.dataset_info.query_neighbors_path.clone(),
            vectors_count: state.dataset_info.vectors_count,
            queries_count: state.dataset_info.queries_count,
            dimensions: state.dataset_info.dimensions,
            neighbors_per_query: state.dataset_info.neighbors_per_query,
        },
        config: metadata,
        steps,
    };

    write_report(&path, &report)?;
    eprintln!("  → {}", path.display());
    Ok(())
}

/// Run one benchmark configuration. Accumulates steps, writes JSON report at
/// the end. `dimensions` is the per-vector dimensionality this config exposes to
/// the backend (≤ native dimensions of the underlying datasets).
pub fn run(backend: &mut dyn Backend, state: &mut BenchState, dimensions: usize) -> BenchResult<()> {
    let total_vectors = state.total_vectors;

    let description = backend.description();
    let metadata = backend.metadata();
    eprintln!("\n── {description} ──");

    let num_steps = state.steps;
    let step_size = total_vectors.div_ceil(num_steps);
    let add_style = ProgressStyle::default_bar()
        .template("  add    [{elapsed_precise}] {bar:40.cyan/blue} {msg}")
        .unwrap()
        .progress_chars("##-");
    let search_style = ProgressStyle::default_bar()
        .template("  search [{elapsed_precise}] {bar:40.green/blue} {msg}")
        .unwrap()
        .progress_chars("##-");

    let mut vectors_indexed = 0usize;
    let mut steps: Vec<StepEntry> = Vec::with_capacity(num_steps);

    // Try to attach system-wide per-CPU hardware perf counters. If the caller
    // doesn't have CAP_PERFMON / the right paranoia setting, or we're on a
    // non-Linux target, this returns Err and we fall through to running without
    // counters — the StepEntry fields stay None and are serde-skipped.
    let mut perf = match perf_counters::PerfCounters::new() {
        Ok(pc) => {
            eprintln!("  perf counters: system-wide per-CPU capture enabled");
            Some(pc)
        }
        Err(err) => {
            eprintln!("  perf counters: unavailable ({err}); running without");
            None
        }
    };

    for step in 0..num_steps {
        let step_start = step * step_size;
        let step_count = step_size.min(total_vectors - step_start);
        let is_final_step = step == num_steps - 1;

        let add = run_add_phase(
            backend,
            state,
            &mut perf,
            &add_style,
            StepSlice {
                start: step_start,
                count: step_count,
                dimensions,
            },
            &mut vectors_indexed,
        )?;

        let search = run_search_phase(
            backend,
            state,
            &mut perf,
            &search_style,
            dimensions,
            vectors_indexed,
            is_final_step,
        )?;

        steps.push(StepEntry {
            vectors_indexed,
            memory_bytes: backend.memory_bytes() as u64,
            add: Some(StepAddEntry {
                elapsed: add.elapsed_secs,
                throughput: add.throughput_per_sec,
                counters: PhaseCounters::from_sample(add.counter_sample.as_ref()),
            }),
            ground_truth_search: Some(search),
            self_search: None,
        });
    }

    // Once, against the finished index — a per-step sweep would cost the sum of
    // every prefix, and on batch-built backends would force a rebuild each time.
    // It lands on the last step because that is the index state it ran against.
    if let Some(queries) = self_search_query_count(state, vectors_indexed) {
        let entry = run_self_search_phase(
            backend,
            state,
            &mut perf,
            &self_search_style(),
            state.self_search_top_k,
            queries,
            dimensions,
        )?;
        if let Some(last) = steps.last_mut() {
            last.self_search = Some(entry);
        }
    }

    let peak_memory = steps.iter().map(|s| s.memory_bytes).max().unwrap_or(0);
    save_report(state, metadata, dimensions, steps)?;

    eprintln!("  peak memory: {:.2} GB", peak_memory as f64 / (1u64 << 30) as f64);
    eprintln!();
    Ok(())
}

/// Run one benchmark configuration against a pre-built / loaded index — no
/// add phase. Emits a single `StepEntry` whose `add_*` fields are zero /
/// `None`. `dimensions` is the per-vector dimensionality the loaded index expects
/// (queries are sliced at this dimensions before being handed to `search`).
pub fn run_search_only(backend: &mut dyn Backend, state: &mut BenchState, dimensions: usize) -> BenchResult<()> {
    // A loaded index need not cover the same slice of the base as this run's
    // `--max-base-vectors` implies — it was built by an earlier invocation with
    // its own flags. Prefer the backend's own count; self-recall in particular
    // reports nonsense if it queries rows the index never saw.
    let total_vectors = match backend.indexed_count() {
        Some(indexed) if indexed != state.total_vectors => {
            eprintln!(
                "  loaded index holds {} vectors, dataset implies {} — using the index's count",
                format_thousands(indexed as u64),
                format_thousands(state.total_vectors as u64),
            );
            indexed
        }
        Some(indexed) => indexed,
        None => state.total_vectors,
    };

    let description = backend.description();
    let metadata = backend.metadata();
    eprintln!("\n── {description} (search-only) ──");

    let search_style = ProgressStyle::default_bar()
        .template("  search [{elapsed_precise}] {bar:40.green/blue} {msg}")
        .unwrap()
        .progress_chars("##-");

    let mut perf = match perf_counters::PerfCounters::new() {
        Ok(pc) => {
            eprintln!("  perf counters: system-wide per-CPU capture enabled");
            Some(pc)
        }
        Err(err) => {
            eprintln!("  perf counters: unavailable ({err}); running without");
            None
        }
    };

    let search = run_search_phase(
        backend,
        state,
        &mut perf,
        &search_style,
        dimensions,
        total_vectors,
        true,
    )?;

    let mut step = StepEntry {
        vectors_indexed: total_vectors,
        memory_bytes: backend.memory_bytes() as u64,
        add: None,
        ground_truth_search: Some(search),
        self_search: None,
    };

    if let Some(queries) = self_search_query_count(state, total_vectors) {
        step.self_search = Some(run_self_search_phase(
            backend,
            state,
            &mut perf,
            &self_search_style(),
            state.self_search_top_k,
            queries,
            dimensions,
        )?);
    }

    let peak_memory = step.memory_bytes;
    save_report(state, metadata, dimensions, vec![step])?;

    eprintln!("  peak memory: {:.2} GB", peak_memory as f64 / (1u64 << 30) as f64);
    eprintln!();
    Ok(())
}

// #region Sweep driver

/// How one config in a sweep ended up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigOutcome {
    /// Backend constructed and `run()` returned `Ok(())`.
    Ran,
    /// Backend construction failed — config is invalid for this engine (e.g. non-default ef on
    /// FAISS binary HNSW).
    Skipped,
    /// Backend ran but the benchmark loop errored mid-flight.
    Failed,
}

/// Run an already-constructed backend, routing a construction or run failure to
/// `Skipped` / `Failed` rather than aborting the sweep. For backends that build
/// their engine handle eagerly (the server-backed ones) and have no `--index`
/// branch to take.
pub fn try_run_config<Index, ConstructError>(
    description: &str,
    backend: Result<Index, ConstructError>,
    state: &mut BenchState,
    dimensions: usize,
) -> ConfigOutcome
where
    Index: Backend,
    ConstructError: std::fmt::Display,
{
    match backend {
        Ok(mut backend) => match run(&mut backend, state, dimensions) {
            Ok(()) => ConfigOutcome::Ran,
            Err(err) => {
                eprintln!("\n── {description} — failed ──\n  {err}");
                ConfigOutcome::Failed
            }
        },
        Err(err) => {
            eprintln!("\n── {description} — skipped ──\n  {err}");
            ConfigOutcome::Skipped
        }
    }
}

/// Run one config, routing failures to `Skipped` / `Failed` instead of
/// aborting the sweep, and handling the `--index` load-vs-build branch.
///
/// - `handle = None` → build and run.
/// - `handle = Some(h)` and the path **exists** → calls `load(h)` and runs
///   `run_search_only`. No add phase.
/// - `handle = Some(h)` and the path **does not exist** → calls `build()`,
///   runs the full `run`, then `idx.save(h)` once it returns.
///
/// Build / load / run / save errors are routed to `Skipped` / `Failed` and
/// do not abort the surrounding sweep — the caller records the outcome
/// against a `SweepSummary`.
pub fn run_config<Index, BuildFn, LoadFn>(
    description: &str,
    handle: Option<&str>,
    build: BuildFn,
    load: LoadFn,
    state: &mut BenchState,
    dimensions: usize,
) -> ConfigOutcome
where
    Index: Backend,
    BuildFn: FnOnce() -> Result<Index, String>,
    LoadFn: FnOnce(&str) -> Result<Index, String>,
{
    let load_existing = handle.is_some_and(|h| std::path::Path::new(h).exists());

    if load_existing {
        let h = handle.unwrap();
        match load(h) {
            Ok(mut idx) => match run_search_only(&mut idx, state, dimensions) {
                Ok(()) => ConfigOutcome::Ran,
                Err(err) => {
                    eprintln!("\n── {description} (loaded) — failed ──\n  {err}");
                    ConfigOutcome::Failed
                }
            },
            Err(err) => {
                eprintln!("\n── {description} (load[{h}]) — skipped ──\n  {err}");
                ConfigOutcome::Skipped
            }
        }
    } else {
        match build() {
            Ok(mut idx) => match run(&mut idx, state, dimensions) {
                Ok(()) => match handle {
                    Some(h) => match idx.save(h) {
                        Ok(()) => {
                            eprintln!("  saved index → {h}");
                            ConfigOutcome::Ran
                        }
                        Err(err) => {
                            eprintln!("\n── {description} (save[{h}]) — failed ──\n  {err}");
                            ConfigOutcome::Failed
                        }
                    },
                    None => ConfigOutcome::Ran,
                },
                Err(err) => {
                    eprintln!("\n── {description} — failed ──\n  {err}");
                    ConfigOutcome::Failed
                }
            },
            Err(err) => {
                eprintln!("\n── {description} — skipped ──\n  {err}");
                ConfigOutcome::Skipped
            }
        }
    }
}

/// Aggregate outcome counter for a sweep — record per config, print once at the end.
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

    pub fn print(&self) {
        eprintln!(
            "Benchmark complete: {} config(s) ran, {} skipped, {} failed.",
            self.ran, self.skipped, self.failed
        );
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
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(SPLITMIX64_GAMMA);
        mix(self.state)
    }

    /// A draw below `bound`: the high half of a draw times `bound`, with a bias of at most `bound` over 2^64.
    pub fn below(&mut self, bound: u64) -> u64 {
        ((u128::from(self.next()) * u128::from(bound)) >> 64) as u64
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
    let all_cores = std::thread::available_parallelism().unwrap_or(NonZeroUsize::MIN);
    parse_threads(text, all_cores).ok_or_else(|| "expected a count, 0 for all cores".into())
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
pub fn spell_list<T: fmt::Display>(values: &[T]) -> String {
    values.iter().map(T::to_string).collect::<Vec<_>>().join(",")
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
        parse_self_search_sample, resolve_self_search_sample, SearchBuffers, SelfSearchSample, SplitMix64, VectorSlice,
    };

    #[test]
    fn owned_search_and_conversion_buffers_reuse_allocations() {
        let mut buffers = SearchBuffers::new_in(System);
        buffers.resize(8, 4);
        let pointers = (
            buffers.keys.as_ptr(),
            buffers.distances.as_ptr(),
            buffers.counts.as_ptr(),
        );
        buffers.resize(3, 4);
        assert_eq!(
            pointers,
            (
                buffers.keys.as_ptr(),
                buffers.distances.as_ptr(),
                buffers.counts.as_ptr()
            )
        );
        let mut scratch = Vec::new_in(System);
        assert_eq!(VectorSlice::I8(&[-2, 3]).to_f32_in(&mut scratch).unwrap(), [-2.0, 3.0]);
        let pointer = scratch.as_ptr();
        assert_eq!(VectorSlice::U8(&[7]).to_f32_in(&mut scratch).unwrap(), [7.0]);
        assert_eq!(pointer, scratch.as_ptr());
        let source = [2.0, 4.0];
        assert_eq!(
            VectorSlice::F32(&source).to_f32_in(&mut scratch).unwrap().as_ptr(),
            source.as_ptr()
        );
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
        assert_eq!(SplitMix64 { state: 42 }.next(), 0xbdd7_3226_2feb_6e95);
    }
}
