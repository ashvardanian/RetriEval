//! Shared benchmark infrastructure for vector search engines.
//!
//! This is the library root. Backend binaries (`usearch.rs`, `faiss.rs`, etc.)
//! import from here and provide their own `main()`.

// `generate.rs` is compiled twice — once as the `retri-generate` binary (its
// own crate) and once as `pub mod generate` inside this library. The alias
// below lets items inside `generate.rs` write `retrieval::pod_slice_as_bytes`
// in both compilation contexts: the binary resolves via the extern crate
// dependency, the library resolves via this self-alias.
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

#[cfg(feature = "download")]
pub use error::DownloadError;
pub use error::{DatasetError, GroundTruthError, PerfCountersError};

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use indicatif::{ProgressBar, ProgressStyle};
use serde_json::Value;

pub use dataset::{Dataset, GroundTruth, Keys};
pub use output::{
    collect_machine_info, config_hash, write_report, ConfigReport, DatasetInfo, MachineInfo, PhaseCounters,
    StepAddEntry, StepEntry, StepSearchEntry,
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
    pub fn to_f32(&self) -> Cow<'_, [Distance]> {
        match self {
            VectorSlice::F32(d) => Cow::Borrowed(d),
            VectorSlice::I8(d) => Cow::Owned(d.iter().map(|&x| x as Distance).collect()),
            VectorSlice::U8(d) => Cow::Owned(d.iter().map(|&x| x as Distance).collect()),
            VectorSlice::B1x8(d) => Cow::Owned(d.iter().map(|&x| x as Distance).collect()),
        }
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
        &self,
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
pub struct IndexConfig<'a> {
    pub dimensions: usize,
    pub data_type: &'a str,
    pub metric: &'a str,
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
    #[arg(long)]
    pub search_count: Option<usize>,

    /// Disable shuffling of insertion order (shuffle is on by default)
    #[arg(long, default_value_t = false)]
    pub no_shuffle: bool,

    /// Number of measurement steps (dataset is split into this many equal parts)
    #[arg(long, default_value_t = 10)]
    pub steps: usize,

    /// Vectors per backend add() call
    #[arg(long, default_value_t = 10_000)]
    pub batch_size_add: usize,

    /// Queries per backend search() call
    #[arg(long, default_value_t = 10_000)]
    pub batch_size_search: usize,

    /// Output directory for JSON result files
    #[arg(long)]
    pub output: Option<PathBuf>,

    /// Cap the number of base vectors used (for calibration on a slice of a larger file).
    /// Queries and ground truth are unaffected; only the add/permutation range shrinks.
    #[arg(long)]
    pub max_base_vectors: Option<usize>,

    /// Persisted-index handle. For embedded backends (USearch, FAISS, cuVS): a
    /// filesystem path — if it exists the backend loads it and skips the add
    /// phase, otherwise the backend builds and saves to it. For server-style
    /// backends: a collection / table / index name (not yet implemented).
    /// Requires a single-config sweep — multi-valued sweep axes are rejected
    /// at startup when `--index` is set.
    #[arg(long)]
    pub index: Option<String>,

    /// Replay indexed vectors as their own queries and report the fraction that
    /// retrieve themselves within the top-`--self-search-count` ("self-recall").
    /// Runs once, after the last insertion, and needs no ground truth — a vector
    /// in the index is its own nearest neighbor. Because it can sweep the whole
    /// base rather than a short query file, it is also the only phase that
    /// measures throughput under sustained load.
    #[arg(long, default_value_t = false)]
    pub self_search: bool,

    /// Neighbors requested per query during the self-recall sweep (the `k` in
    /// self-recall@k). No effect without `--self-search`.
    #[arg(long, default_value_t = 10)]
    pub self_search_count: usize,

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
    #[arg(long, value_name = "N|FRACTION")]
    pub self_search_sample: Option<String>,

    /// Matryoshka-style embedding-dimension truncations to evaluate
    /// (comma-separated). Empty → use the file's native dimensions. Each value must
    /// be ≤ the native dimensions; for `.b1bin` files each must be a multiple of 8.
    #[arg(long, value_delimiter = ',')]
    pub dimensions: Vec<usize>,
}

impl CommonArgs {
    /// Resolve `--dimensions` into a sweep list. Empty CLI input expands to a
    /// single-element list at the file's native dimensions, so binaries can iterate
    /// uniformly without special-casing the no-truncation path.
    pub fn dimensions_sweep(&self, native: usize) -> Vec<usize> {
        if self.dimensions.is_empty() {
            vec![native]
        } else {
            self.dimensions.clone()
        }
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
    /// Resolved `--search-count`: the search width and the k every metric uses.
    pub count: usize,
    pub steps: usize,
    pub batch_size_add: usize,
    pub batch_size_search: usize,
    pub self_search: bool,
    pub self_search_count: usize,
    pub self_search_sample: Option<String>,
    pub output_dir: Option<PathBuf>,
    pub machine_info: MachineInfo,
    pub dataset_info: DatasetInfo,
    out_keys: Vec<Key>,
    out_distances: Vec<Distance>,
    out_counts: Vec<usize>,
    key_scratch: Vec<Key>,
    /// Shared scratch for `Dataset::gather` (during add) and `Dataset::slice`
    /// (during search). Add and search are sequential within a step, so a
    /// single buffer sized at the upper bound is enough. Sized to fit
    /// `max(batch_size_add, batch_size_search) * native_vector_bytes` —
    /// covers any truncation since truncated bytes ≤ native.
    scratch_buf: Vec<u8>,
}

impl BenchState {
    pub fn load(args: &CommonArgs) -> Result<Self, Box<dyn std::error::Error>> {
        if args.steps == 0 {
            return Err("--steps must be greater than 0".into());
        }
        if args.batch_size_add == 0 {
            return Err("--batch-size-add must be greater than 0".into());
        }
        if args.self_search {
            if args.self_search_count == 0 {
                return Err("--self-search-count must be greater than 0".into());
            }
            // Validate the sample spec's grammar now (before the base size is
            // known); the actual vector count is resolved per-run in
            // `resolve_self_search_sample`.
            if let Some(spec) = &args.self_search_sample {
                parse_self_search_sample(spec)?;
            }
        } else if args.self_search_sample.is_some() {
            return Err("--self-search-sample has no effect without --self-search".into());
        }

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

        let perm = if args.no_shuffle {
            dataset::Permutation::identity(total_vectors)
        } else {
            eprintln!("Shuffling insertion order...");
            dataset::Permutation::shuffled(total_vectors, 42)
        };

        let ground_truth_width = ground_truth.neighbors_per_query();
        let count = args.search_count.unwrap_or(ground_truth_width);
        if count == 0 {
            return Err("--search-count must be greater than 0".into());
        }
        if count > ground_truth_width {
            return Err(format!(
                "--search-count {count} exceeds the ground truth's {ground_truth_width} neighbors per query"
            )
            .into());
        }

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
            count,
            steps: args.steps,
            batch_size_add: args.batch_size_add,
            batch_size_search: args.batch_size_search,
            self_search: args.self_search,
            self_search_count: args.self_search_count,
            self_search_sample: args.self_search_sample.clone(),
            output_dir: args.output.clone(),
            machine_info,
            dataset_info,
            out_keys: vec![0 as Key; num_queries * count],
            out_distances: vec![0.0 as Distance; num_queries * count],
            out_counts: vec![0usize; num_queries],
            key_scratch: vec![0 as Key; args.batch_size_add],
            scratch_buf: {
                let max_batch = args.batch_size_add.max(args.batch_size_search);
                let max_row_bytes = dataset.vector_bytes().max(query_dataset.vector_bytes());
                vec![0u8; max_batch * max_row_bytes]
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

/// Insert one step's slice in `batch_size_add` chunks, capturing perf counters
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
    let batch_size_add = state.batch_size_add;

    let progress = ProgressBar::new(total_vectors as u64);
    progress.set_style(progress_style.clone());
    progress.set_position(*vectors_indexed as u64);

    start_counter_capture(perf, "add");
    let add_start = Instant::now();
    let mut added = 0;
    while added < step_count {
        let batch = batch_size_add.min(step_count - added);
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
    backend: &dyn Backend,
    state: &mut BenchState,
    perf: &mut Option<perf_counters::PerfCounters>,
    progress_style: &ProgressStyle,
    dimensions: usize,
    vectors_indexed: usize,
    is_final_step: bool,
) -> BenchResult<StepSearchEntry> {
    let num_queries = state.query_dataset.rows();
    let count = state.count;

    let progress = ProgressBar::new(num_queries as u64);
    progress.set_style(progress_style.clone());
    progress.set_position(0);

    start_counter_capture(perf, "search");
    let search_start = Instant::now();
    let mut searched = 0usize;
    while searched < num_queries {
        let batch_rows = state.batch_size_search.min(num_queries - searched);
        let batch_queries = state
            .query_dataset
            .slice(searched, batch_rows, dimensions, &mut state.scratch_buf);
        let key_offset = searched * count;
        let key_end = key_offset + batch_rows * count;

        backend.search(
            batch_queries,
            count,
            &mut state.out_keys[key_offset..key_end],
            &mut state.out_distances[key_offset..key_end],
            &mut state.out_counts[searched..searched + batch_rows],
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
    let recall_at_1 = eval::recall_at_k(&state.out_keys, &state.out_counts, count, &state.ground_truth, 1);
    let recall_at_k = eval::recall_at_k(&state.out_keys, &state.out_counts, count, &state.ground_truth, count);
    let intersection_at_k =
        eval::intersection_at_k(&state.out_keys, &state.out_counts, count, &state.ground_truth, count);
    let ndcg_at_k = eval::ndcg_at_k(&state.out_keys, &state.out_counts, count, &state.ground_truth, count);

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
        neighbor_count: count,
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
    backend: &dyn Backend,
    state: &mut BenchState,
    perf: &mut Option<perf_counters::PerfCounters>,
    progress_style: &ProgressStyle,
    count: usize,
    num_queries: usize,
    dimensions: usize,
) -> BenchResult<StepSearchEntry> {
    let batch_size = state.batch_size_search.min(num_queries).max(1);
    let mut batch_keys = vec![0 as Key; batch_size * count];
    let mut batch_distances = vec![0.0 as Distance; batch_size * count];
    let mut batch_counts = vec![0usize; batch_size];
    let mut expected_scratch = vec![0 as Key; batch_size];

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
        neighbor_count: count,
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
enum SelfSearchSample {
    Fraction(f64),
    Absolute(usize),
}

/// Validate the grammar of a `--self-search-sample` spec. The base size isn't
/// known at CLI-parse time, so resolution to a concrete count is deferred to
/// `resolve_self_search_sample`.
fn parse_self_search_sample(spec: &str) -> Result<SelfSearchSample, String> {
    let trimmed = spec.trim();
    if trimmed.contains('.') {
        let fraction: f64 = trimmed
            .parse()
            .map_err(|_| format!("--self-search-sample: invalid fraction `{spec}`"))?;
        if !(fraction > 0.0 && fraction <= 1.0) {
            return Err(format!(
                "--self-search-sample fraction must be in (0.0, 1.0], got {fraction}"
            ));
        }
        Ok(SelfSearchSample::Fraction(fraction))
    } else {
        let count: usize = trimmed
            .parse()
            .map_err(|_| format!("--self-search-sample: invalid count `{spec}`"))?;
        if count == 0 {
            return Err("--self-search-sample count must be greater than 0".into());
        }
        Ok(SelfSearchSample::Absolute(count))
    }
}

/// Resolve a sample spec against the number of indexed vectors. `None` → all.
/// A fraction rounds to at least one vector; an absolute count is clamped to
/// what is actually in the index.
fn resolve_self_search_sample(spec: Option<&str>, vectors_indexed: usize) -> Result<usize, String> {
    Ok(match spec {
        None => vectors_indexed,
        Some(spec) => match parse_self_search_sample(spec)? {
            SelfSearchSample::Fraction(fraction) => ((vectors_indexed as f64 * fraction).round() as usize).max(1),
            SelfSearchSample::Absolute(count) => count.min(vectors_indexed),
        },
    })
}

/// Resolve how many vectors the self-recall sweep should replay, given how many
/// are actually in the index. `None` when `--self-search` was not passed. The
/// sample grammar is validated in `BenchState::load`, so resolution here is
/// infallible in practice.
fn self_search_query_count(state: &BenchState, vectors_indexed: usize) -> Option<usize> {
    if !state.self_search {
        return None;
    }
    let resolved = resolve_self_search_sample(state.self_search_sample.as_deref(), vectors_indexed)
        .expect("self-recall sample spec validated at load");
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
    // hash, centrally: leaving it to each backend is how `--dimensions` came to
    // be missing on two of them, silently overwriting result files. Backends
    // contribute only their engine-specific knobs.
    metadata.insert("dimensions".into(), Value::from(dimensions));
    metadata.insert("vectors_count".into(), Value::from(state.total_vectors));
    metadata.insert("steps".into(), Value::from(state.steps));
    metadata.insert("batch_size_add".into(), Value::from(state.batch_size_add));
    metadata.insert("batch_size_search".into(), Value::from(state.batch_size_search));
    metadata.insert("search_count".into(), Value::from(state.count));
    if let Some(entry) = steps.last().and_then(|step| step.self_search.as_ref()) {
        metadata.insert("self_search_count".into(), Value::from(entry.neighbor_count));
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
            state.self_search_count,
            queries,
            dimensions,
        )?;
        if let Some(last) = steps.last_mut() {
            last.self_search = Some(entry);
        }
    }

    let peak_memory = steps.iter().map(|s| s.memory_bytes).max().unwrap_or(0);
    save_report(state, metadata, dimensions, steps)?;

    eprintln!("  peak memory: {:.2} GB", peak_memory as f64 / 1e9);
    eprintln!();
    Ok(())
}

/// Run one benchmark configuration against a pre-built / loaded index — no
/// add phase. Emits a single `StepEntry` whose `add_*` fields are zero /
/// `None`. `dimensions` is the per-vector dimensionality the loaded index expects
/// (queries are sliced at this dimensions before being handed to `search`).
pub fn run_search_only(backend: &dyn Backend, state: &mut BenchState, dimensions: usize) -> BenchResult<()> {
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
            state.self_search_count,
            queries,
            dimensions,
        )?);
    }

    let peak_memory = step.memory_bytes;
    save_report(state, metadata, dimensions, vec![step])?;

    eprintln!("  peak memory: {:.2} GB", peak_memory as f64 / 1e9);
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
            Ok(idx) => match run_search_only(&idx, state, dimensions) {
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
    use super::{resolve_self_search_sample, SelfSearchSample};

    #[test]
    fn self_recall_sample_none_is_all() {
        assert_eq!(resolve_self_search_sample(None, 1000).unwrap(), 1000);
    }

    #[test]
    fn self_recall_sample_absolute_is_clamped() {
        assert_eq!(resolve_self_search_sample(Some("250"), 1000).unwrap(), 250);
        assert_eq!(resolve_self_search_sample(Some("5000"), 1000).unwrap(), 1000);
    }

    #[test]
    fn self_recall_sample_fraction_rounds_and_floors_at_one() {
        assert_eq!(resolve_self_search_sample(Some("0.1"), 1000).unwrap(), 100);
        assert_eq!(resolve_self_search_sample(Some("1.0"), 1000).unwrap(), 1000);
        // A fraction so small it would round to zero still yields one vector.
        assert_eq!(resolve_self_search_sample(Some("0.0001"), 1000).unwrap(), 1);
    }

    #[test]
    fn self_recall_sample_one_vs_one_point_zero_disambiguate() {
        // `1` is an absolute count; `1.0` is the whole base.
        assert!(matches!(
            super::parse_self_search_sample("1").unwrap(),
            SelfSearchSample::Absolute(1)
        ));
        assert!(matches!(
            super::parse_self_search_sample("1.0").unwrap(),
            SelfSearchSample::Fraction(_)
        ));
    }

    #[test]
    fn self_recall_sample_rejects_zero_and_out_of_range() {
        assert!(resolve_self_search_sample(Some("0"), 1000).is_err());
        assert!(resolve_self_search_sample(Some("0.0"), 1000).is_err());
        assert!(resolve_self_search_sample(Some("1.5"), 1000).is_err());
        assert!(resolve_self_search_sample(Some("abc"), 1000).is_err());
    }
}
