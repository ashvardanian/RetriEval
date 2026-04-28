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
pub use error::{BackendError, DatasetError, GroundTruthError, PerfCountersError};

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use indicatif::{ProgressBar, ProgressStyle};
use serde_json::Value;

pub use dataset::{Dataset, GroundTruth, Keys};
pub use output::{collect_machine_info, config_hash, write_report, ConfigReport, DatasetInfo, MachineInfo, StepEntry};

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

    /// Persist the index under `handle`. For embedded backends `handle` is a
    /// filesystem path; for server-style backends it's a collection / table /
    /// index name. Default returns Err so backends opt in by overriding.
    fn save(&self, _handle: &str) -> Result<(), String> {
        Err("--index: save not yet implemented for this backend".into())
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
    /// Path to the base vectors file (.fbin, .u8bin, .i8bin)
    #[arg(long)]
    pub vectors: PathBuf,

    /// Path to the query vectors file
    #[arg(long)]
    pub queries: PathBuf,

    /// Path to the ground-truth neighbors file (.ibin)
    #[arg(long)]
    pub neighbors: PathBuf,

    /// Optional path to a keys file (.i32bin)
    #[arg(long)]
    pub keys: Option<PathBuf>,

    /// Disable shuffling of insertion order (shuffle is on by default)
    #[arg(long, default_value_t = false)]
    pub no_shuffle: bool,

    /// Number of measurement steps (dataset is split into this many equal parts)
    #[arg(long, default_value_t = 10)]
    pub epochs: usize,

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
    pub epochs: usize,
    pub batch_size_add: usize,
    pub batch_size_search: usize,
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
        if args.epochs == 0 {
            return Err("--epochs must be greater than 0".into());
        }
        if args.batch_size_add == 0 {
            return Err("--batch-size-add must be greater than 0".into());
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

        eprintln!("Loading dataset: {}", args.vectors.display());
        let dataset = Dataset::load(&args.vectors)?;
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
        } else {
            eprintln!(
                "  {} vectors, {} dimensions",
                format_thousands(total_vectors as u64),
                dimensions
            );
        }

        let keys = match &args.keys {
            Some(path) => {
                eprintln!("Loading keys: {}", path.display());
                let k = Keys::load(path)?;
                eprintln!("  {} keys", format_thousands(k.count() as u64));
                k
            }
            None => Keys::sequential(total_vectors),
        };

        eprintln!("Loading queries: {}", args.queries.display());
        let query_dataset = Dataset::load(&args.queries)?;
        let num_queries = query_dataset.rows();
        eprintln!(
            "  {} queries, {} dimensions",
            format_thousands(num_queries as u64),
            query_dataset.dimensions()
        );

        eprintln!("Loading ground truth: {}", args.neighbors.display());
        let ground_truth = GroundTruth::load(&args.neighbors)?;
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

        let search_count = ground_truth.neighbors_per_query();
        let add_chunk_size = args.batch_size_add;

        let dataset_info = DatasetInfo {
            vectors_path: args.vectors.display().to_string(),
            queries_path: args.queries.display().to_string(),
            neighbors_path: args.neighbors.display().to_string(),
            vectors_count: total_vectors,
            queries_count: num_queries,
            dimensions,
            neighbors_per_query: search_count,
        };

        Ok(Self {
            total_vectors,
            perm,
            epochs: args.epochs,
            batch_size_add: add_chunk_size,
            batch_size_search: args.batch_size_search,
            output_dir: args.output.clone(),
            machine_info,
            dataset_info,
            out_keys: vec![0 as Key; num_queries * search_count],
            out_distances: vec![0.0 as Distance; num_queries * search_count],
            out_counts: vec![0usize; num_queries],
            key_scratch: vec![0 as Key; add_chunk_size],
            scratch_buf: {
                let max_batch = add_chunk_size.max(args.batch_size_search);
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

/// Outcome of one step's search phase.
struct SearchPhaseOutcome {
    elapsed_secs: f64,
    throughput_per_sec: u64,
    counter_sample: Option<perf_counters::CounterSample>,
    recall_at_1: f64,
    recall_at_10: f64,
    ndcg_at_10: f64,
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

/// Run the add-phase of one step: insert `step_count` vectors in batches of
/// `add_chunk_size`, capturing perf counters across the whole phase.
/// Mutates `vectors_indexed` to the new cumulative count. `dimensions` is the
/// per-vector dimensionality exposed to the backend (≤ native).
#[allow(clippy::too_many_arguments)]
fn run_add_phase(
    index: &mut dyn Backend,
    state: &mut BenchState,
    perf: &mut Option<perf_counters::PerfCounters>,
    progress_style: &ProgressStyle,
    step_start: usize,
    step_count: usize,
    total_vectors: usize,
    add_chunk_size: usize,
    dimensions: usize,
    vectors_indexed: &mut usize,
) -> BenchResult<AddPhaseOutcome> {
    let progress = ProgressBar::new(total_vectors as u64);
    progress.set_style(progress_style.clone());
    progress.set_position(*vectors_indexed as u64);

    start_counter_capture(perf, "add");
    let add_start = Instant::now();
    let mut added = 0;
    while added < step_count {
        let batch = add_chunk_size.min(step_count - added);
        let logical_offset = step_start + added;
        let indices = state.perm.range(logical_offset, batch);

        for (slot, &row_index) in indices.iter().enumerate() {
            state.key_scratch[slot] = state.keys.get(row_index);
        }
        let batch_keys = &state.key_scratch[..batch];
        let vectors = state.dataset.gather(indices, dimensions, &mut state.scratch_buf);

        index.add(batch_keys, vectors)?;
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

/// Run the search-phase of one step: execute all queries against the current
/// index, compute recall/NDCG against the ground truth, emit a progress bar.
/// `dimensions` is the per-vector dimensionality exposed to the backend.
#[allow(clippy::too_many_arguments)]
fn run_search_phase(
    index: &dyn Backend,
    state: &mut BenchState,
    perf: &mut Option<perf_counters::PerfCounters>,
    progress_style: &ProgressStyle,
    num_queries: usize,
    search_count: usize,
    vectors_indexed: usize,
    total_vectors: usize,
    dimensions: usize,
    is_final_step: bool,
) -> BenchResult<SearchPhaseOutcome> {
    let progress = ProgressBar::new(num_queries as u64);
    progress.set_style(progress_style.clone());
    progress.set_position(0);

    start_counter_capture(perf, "search");
    let search_start = Instant::now();
    let mut searched = 0usize;
    while searched < num_queries {
        let batch_count = state.batch_size_search.min(num_queries - searched);
        let batch_queries = state
            .query_dataset
            .slice(searched, batch_count, dimensions, &mut state.scratch_buf);
        let key_offset = searched * search_count;
        let key_end = key_offset + batch_count * search_count;

        index.search(
            batch_queries,
            search_count,
            &mut state.out_keys[key_offset..key_end],
            &mut state.out_distances[key_offset..key_end],
            &mut state.out_counts[searched..searched + batch_count],
        )?;

        searched += batch_count;
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
    let recall_at_1 = eval::recall_at_k(&state.out_keys, &state.out_counts, search_count, &state.ground_truth, 1);
    let recall_at_10 = eval::recall_at_k(
        &state.out_keys,
        &state.out_counts,
        search_count,
        &state.ground_truth,
        10,
    );
    let ndcg_at_10 = eval::ndcg_at_k(
        &state.out_keys,
        &state.out_counts,
        search_count,
        &state.ground_truth,
        10,
    );

    let recall_at_1_norm = eval::normalize_metric(recall_at_1, vectors_indexed, total_vectors);
    let recall_at_10_norm = eval::normalize_metric(recall_at_10, vectors_indexed, total_vectors);
    let ndcg_at_10_norm = eval::normalize_metric(ndcg_at_10, vectors_indexed, total_vectors);

    let approx = if is_final_step { "" } else { "~" };
    progress.finish_with_message(format!(
        "{} search/s, {approx}recall@1={recall_at_1_norm:.4}, \
         {approx}recall@10={recall_at_10_norm:.4}, \
         {approx}NDCG@10={ndcg_at_10_norm:.4} ({} vectors)",
        format_thousands(throughput_per_sec),
        format_thousands(vectors_indexed as u64),
    ));

    Ok(SearchPhaseOutcome {
        elapsed_secs,
        throughput_per_sec,
        counter_sample,
        recall_at_1,
        recall_at_10,
        ndcg_at_10,
    })
}

/// Assemble a `ConfigReport` and write it to `<output_dir>/<backend>-<hash>.json`.
/// No-op when the state has no output directory configured.
fn save_report(state: &BenchState, metadata: HashMap<String, Value>, steps: Vec<StepEntry>) -> BenchResult<()> {
    let Some(dir) = &state.output_dir else {
        return Ok(());
    };
    let backend_name = metadata.get("backend").and_then(|v| v.as_str()).unwrap_or("unknown");
    let hash = config_hash(&metadata);
    let filename = format!("{backend_name}-{hash}.json");
    let path = dir.join(&filename);

    let report = ConfigReport {
        machine: collect_machine_info(),
        dataset: DatasetInfo {
            vectors_path: state.dataset_info.vectors_path.clone(),
            queries_path: state.dataset_info.queries_path.clone(),
            neighbors_path: state.dataset_info.neighbors_path.clone(),
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
pub fn run(index: &mut dyn Backend, state: &mut BenchState, dimensions: usize) -> BenchResult<()> {
    let total_vectors = state.total_vectors;
    let num_queries = state.query_dataset.rows();
    let search_count = state.ground_truth.neighbors_per_query();
    let add_chunk_size = state.key_scratch.len();

    let description = index.description();
    let metadata = index.metadata();
    eprintln!("\n── {description} ──");

    let num_steps = state.epochs;
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
            index,
            state,
            &mut perf,
            &add_style,
            step_start,
            step_count,
            total_vectors,
            add_chunk_size,
            dimensions,
            &mut vectors_indexed,
        )?;

        let search = run_search_phase(
            index,
            state,
            &mut perf,
            &search_style,
            num_queries,
            search_count,
            vectors_indexed,
            total_vectors,
            dimensions,
            is_final_step,
        )?;

        let recall_at_1_norm = eval::normalize_metric(search.recall_at_1, vectors_indexed, total_vectors);
        let recall_at_10_norm = eval::normalize_metric(search.recall_at_10, vectors_indexed, total_vectors);
        let ndcg_at_10_norm = eval::normalize_metric(search.ndcg_at_10, vectors_indexed, total_vectors);

        steps.push(StepEntry {
            vectors_indexed,
            add_elapsed: add.elapsed_secs,
            add_throughput: add.throughput_per_sec,
            memory_bytes: index.memory_bytes() as u64,
            search_elapsed: search.elapsed_secs,
            search_throughput: search.throughput_per_sec,
            recall_at_1: search.recall_at_1,
            recall_at_10: search.recall_at_10,
            ndcg_at_10: search.ndcg_at_10,
            recall_at_1_normalized: recall_at_1_norm,
            recall_at_10_normalized: recall_at_10_norm,
            ndcg_at_10_normalized: ndcg_at_10_norm,
            cycles_add: add.counter_sample.map(|s| s.cycles),
            instructions_add: add.counter_sample.map(|s| s.instructions),
            cache_misses_add: add.counter_sample.map(|s| s.cache_misses),
            branch_misses_add: add.counter_sample.map(|s| s.branch_misses),
            cycles_search: search.counter_sample.map(|s| s.cycles),
            instructions_search: search.counter_sample.map(|s| s.instructions),
            cache_misses_search: search.counter_sample.map(|s| s.cache_misses),
            branch_misses_search: search.counter_sample.map(|s| s.branch_misses),
        });
    }

    let peak_memory = steps.iter().map(|s| s.memory_bytes).max().unwrap_or(0);
    save_report(state, metadata, steps)?;

    eprintln!("  peak memory: {:.2} GB", peak_memory as f64 / 1e9);
    eprintln!();
    Ok(())
}

/// Run one benchmark configuration against a pre-built / loaded index — no
/// add phase. Emits a single `StepEntry` whose `add_*` fields are zero /
/// `None`. `dimensions` is the per-vector dimensionality the loaded index expects
/// (queries are sliced at this dimensions before being handed to `search`).
pub fn run_search_only(index: &dyn Backend, state: &mut BenchState, dimensions: usize) -> BenchResult<()> {
    let total_vectors = state.total_vectors;
    let num_queries = state.query_dataset.rows();
    let search_count = state.ground_truth.neighbors_per_query();

    let description = index.description();
    let metadata = index.metadata();
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
        index,
        state,
        &mut perf,
        &search_style,
        num_queries,
        search_count,
        total_vectors,
        total_vectors,
        dimensions,
        true,
    )?;

    // Recall normalization assumes the index covers the whole base; for
    // search-only we always pass `vectors_indexed == total_vectors`, so the
    // raw and normalized values coincide.
    let recall_at_1_norm = eval::normalize_metric(search.recall_at_1, total_vectors, total_vectors);
    let recall_at_10_norm = eval::normalize_metric(search.recall_at_10, total_vectors, total_vectors);
    let ndcg_at_10_norm = eval::normalize_metric(search.ndcg_at_10, total_vectors, total_vectors);

    let step = StepEntry {
        vectors_indexed: total_vectors,
        add_elapsed: 0.0,
        add_throughput: 0,
        memory_bytes: index.memory_bytes() as u64,
        search_elapsed: search.elapsed_secs,
        search_throughput: search.throughput_per_sec,
        recall_at_1: search.recall_at_1,
        recall_at_10: search.recall_at_10,
        ndcg_at_10: search.ndcg_at_10,
        recall_at_1_normalized: recall_at_1_norm,
        recall_at_10_normalized: recall_at_10_norm,
        ndcg_at_10_normalized: ndcg_at_10_norm,
        cycles_add: None,
        instructions_add: None,
        cache_misses_add: None,
        branch_misses_add: None,
        cycles_search: search.counter_sample.map(|s| s.cycles),
        instructions_search: search.counter_sample.map(|s| s.instructions),
        cache_misses_search: search.counter_sample.map(|s| s.cache_misses),
        branch_misses_search: search.counter_sample.map(|s| s.branch_misses),
    };

    let peak_memory = step.memory_bytes;
    save_report(state, metadata, vec![step])?;

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

/// Run one config of a sweep without aborting the whole process: any construction error is logged as
/// "skipped", any runtime error as "failed", and the remaining configs still execute.
///
/// Binaries build the per-config pre-construction `description` themselves (we don't have a
/// `Backend::description()` yet when construction fails), then pass in the backend `Result` and let this
/// helper route the outcome.
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

/// `try_run_config`-style graceful-skip wrapper that also handles the
/// `--index` load-vs-build branch.
///
/// - `handle = None` → behaves like `try_run_config` against a freshly-built
///   backend.
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
