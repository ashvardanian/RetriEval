//! Shared benchmark infrastructure for vector search engines.
//!
//! This is the library root. Backend binaries (`usearch.rs`, `faiss.rs`, etc.)
//! import from here and provide their own `main()`.

pub mod dataset;
#[cfg(feature = "tier2")]
pub mod docker;
pub mod eval;
pub mod output;

use std::borrow::Cow;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use indicatif::{ProgressBar, ProgressStyle};

pub use dataset::{Dataset, GroundTruth, Keys};
pub use output::{collect_machine_info, emit, StepRecord};

// #region Core types

/// Integer division rounding up: `div_ceil(7, 3) == 3`.
pub const fn div_ceil(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

/// Vector key type used throughout the benchmark.
/// Matches the 32-bit indices in BigANN `.ibin` ground-truth files.
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
    /// Convert to f32, borrowing when already f32.
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
    /// Number of vectors in the batch.
    pub fn len(&self) -> usize {
        let d = self.dimensions;
        match self.data {
            VectorSlice::F32(data) => data.len() / d,
            VectorSlice::I8(data) => data.len() / d,
            VectorSlice::U8(data) => data.len() / d,
            VectorSlice::B1x8(data) => data.len() / div_ceil(d, 8),
        }
    }
}

// #region Backend trait

/// Common trait for all vector search backends.
pub trait Backend: Send {
    /// Human-readable description for progress output.
    fn description(&self) -> String;

    /// Add vectors with the given keys to the index.
    fn add(&mut self, keys: &[Key], vectors: Vectors) -> Result<(), String>;

    /// Search for the `count` nearest neighbors of each query vector.
    /// Writes into pre-allocated output slices (length `num_queries * count`).
    /// Unfilled slots are set to `Key::MAX` / `Distance::INFINITY`.
    fn search(
        &self,
        queries: Vectors,
        count: usize,
        out_keys: &mut [Key],
        out_distances: &mut [Distance],
        out_counts: &mut [usize],
    ) -> Result<(), String>;

    /// Resident memory used by the index.
    fn memory_bytes(&self) -> usize;
}

// #region Formatting

/// Format a number with thousand separators: 1234567 → "1,234,567"
pub fn fmt_thousands(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            result.push(',');
        }
        result.push(c);
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

    /// Optional path to a keys file (.i32bin). If omitted, sequential keys 0..N are used.
    #[arg(long)]
    pub keys: Option<PathBuf>,

    /// Number of vectors to index per measurement step
    #[arg(long, default_value_t = 1_000_000)]
    pub step_size: usize,

    /// Disable shuffling of insertion order (shuffle is on by default)
    #[arg(long, default_value_t = false)]
    pub no_shuffle: bool,

    /// Output file path (defaults to stdout)
    #[arg(long)]
    pub output: Option<PathBuf>,
}

// #region Benchmark loop

/// Pre-loaded benchmark state. Call `BenchState::load()` once, then `bench()` per configuration.
pub struct BenchState {
    pub dataset: Dataset,
    pub keys: Keys,
    pub query_dataset: Dataset,
    pub ground_truth: GroundTruth,
    pub perm: dataset::Permutation,
    pub step_size: usize,
    writer: Box<dyn Write>,
    out_keys: Vec<Key>,
    out_distances: Vec<Distance>,
    out_counts: Vec<usize>,
    key_scratch: Vec<Key>,
    gather_buf: Vec<u8>,
}

impl BenchState {
    /// Load all datasets and allocate buffers. Call once before benchmarking.
    pub fn load(args: &CommonArgs) -> Result<Self, Box<dyn std::error::Error>> {
        let mut writer: Box<dyn Write> = match &args.output {
            Some(path) => Box::new(std::fs::File::create(path)?),
            None => Box::new(std::io::stdout().lock()),
        };

        emit(&mut writer, &collect_machine_info())?;

        eprintln!("Loading dataset: {}", args.vectors.display());
        let dataset = Dataset::load(&args.vectors)?;
        let total_vectors = dataset.rows();
        eprintln!(
            "  {} vectors, {} dimensions",
            fmt_thousands(total_vectors as u64),
            dataset.dimensions()
        );

        let keys = match &args.keys {
            Some(path) => {
                eprintln!("Loading keys: {}", path.display());
                let k = Keys::load(path)?;
                eprintln!("  {} keys", fmt_thousands(k.count() as u64));
                k
            }
            None => Keys::sequential(total_vectors),
        };

        eprintln!("Loading queries: {}", args.queries.display());
        let query_dataset = Dataset::load(&args.queries)?;
        let num_queries = query_dataset.rows();
        eprintln!(
            "  {} queries, {} dimensions",
            fmt_thousands(num_queries as u64),
            query_dataset.dimensions()
        );

        eprintln!("Loading ground truth: {}", args.neighbors.display());
        let ground_truth = GroundTruth::load(&args.neighbors)?;
        eprintln!(
            "  {} queries, {} neighbors each",
            fmt_thousands(ground_truth.queries() as u64),
            ground_truth.neighbors_per_query(),
        );

        let perm = if args.no_shuffle {
            dataset::Permutation::identity(total_vectors)
        } else {
            eprintln!("Shuffling insertion order...");
            dataset::Permutation::shuffled(total_vectors, 42)
        };

        let search_count = ground_truth.neighbors_per_query();
        let batch_size = 10_000.min(args.step_size);

        Ok(Self {
            perm,
            step_size: args.step_size,
            writer,
            out_keys: vec![0 as Key; num_queries * search_count],
            out_distances: vec![0.0 as Distance; num_queries * search_count],
            out_counts: vec![0usize; num_queries],
            key_scratch: vec![0 as Key; batch_size],
            gather_buf: vec![0u8; batch_size * dataset.vector_bytes()],
            dataset,
            keys,
            query_dataset,
            ground_truth,
        })
    }

    pub fn dimensions(&self) -> usize {
        self.dataset.dimensions()
    }
}

/// Run one benchmark: incremental add with sub-batched progress, search + eval after each step.
pub fn run(
    index: &mut dyn Backend,
    state: &mut BenchState,
) -> Result<(), Box<dyn std::error::Error>> {
    let total_vectors = state.dataset.rows();
    let num_queries = state.query_dataset.rows();
    let search_count = state.ground_truth.neighbors_per_query();
    let batch_size = state.key_scratch.len();

    let description = index.description();
    eprintln!("\n── {description} ──");

    let num_steps = div_ceil(total_vectors, state.step_size);
    let add_style = ProgressStyle::default_bar()
        .template("  add    [{elapsed_precise}] {bar:40.cyan/blue} {msg}")
        .unwrap()
        .progress_chars("##-");
    let search_style = ProgressStyle::default_spinner()
        .template("  search [{elapsed_precise}] {msg}")
        .unwrap();

    let mut vectors_indexed = 0usize;

    for step in 0..num_steps {
        let step_start = step * state.step_size;
        let step_count = state.step_size.min(total_vectors - step_start);

        // --- Add (sub-batched for progress updates) ---
        let pb_add = ProgressBar::new(total_vectors as u64);
        pb_add.set_style(add_style.clone());
        pb_add.set_position(vectors_indexed as u64);

        let add_start = Instant::now();
        let mut added = 0;
        while added < step_count {
            let batch = batch_size.min(step_count - added);
            let logical_offset = step_start + added;
            let indices = state.perm.range(logical_offset, batch);

            for (j, &idx) in indices.iter().enumerate() {
                state.key_scratch[j] = state.keys.get(idx);
            }
            let batch_keys = &state.key_scratch[..batch];
            let vectors = state.dataset.gather(indices, &mut state.gather_buf);

            index.add(batch_keys, vectors)?;
            added += batch;
            vectors_indexed += batch;
            let elapsed = add_start.elapsed().as_secs_f64();
            let throughput = if elapsed > 0.0 {
                (added as f64 / elapsed) as u64
            } else {
                0
            };
            pb_add.set_position(vectors_indexed as u64);
            pb_add.set_message(format!(
                "{}/{} ({} IPS)",
                fmt_thousands(vectors_indexed as u64),
                fmt_thousands(total_vectors as u64),
                fmt_thousands(throughput),
            ));
        }
        let add_elapsed = add_start.elapsed().as_secs_f64();
        let add_throughput = if add_elapsed > 0.0 {
            (step_count as f64 / add_elapsed) as u64
        } else {
            0
        };
        pb_add.set_message(format!(
            "{}/{} ({} IPS)",
            fmt_thousands(vectors_indexed as u64),
            fmt_thousands(total_vectors as u64),
            fmt_thousands(add_throughput),
        ));
        pb_add.finish();

        emit(
            &mut state.writer,
            &StepRecord {
                description: description.clone(),
                phase: "add".to_string(),
                vectors_indexed,
                vectors_total: total_vectors,
                elapsed_seconds: Some(add_elapsed),
                vectors_per_second: Some(add_throughput),
                queries_per_second: None,
                memory_bytes: Some(index.memory_bytes() as u64),
                recall_at_1: None,
                recall_at_10: None,
                ndcg_at_10: None,
            },
        )?;

        // --- Search ---
        let pb_search = ProgressBar::new_spinner();
        pb_search.set_style(search_style.clone());
        pb_search.set_message(format!(
            "{} queries against {} vectors...",
            fmt_thousands(num_queries as u64),
            fmt_thousands(vectors_indexed as u64),
        ));
        pb_search.enable_steady_tick(std::time::Duration::from_millis(100));

        let search_start = Instant::now();
        index.search(
            state.query_dataset.all(),
            search_count,
            &mut state.out_keys,
            &mut state.out_distances,
            &mut state.out_counts,
        )?;
        let search_elapsed = search_start.elapsed().as_secs_f64();

        let search_throughput = if search_elapsed > 0.0 {
            (num_queries as f64 / search_elapsed) as u64
        } else {
            0
        };
        let recall1 = eval::recall_at_k(
            &state.out_keys,
            &state.out_counts,
            search_count,
            &state.ground_truth,
            1,
        );
        let recall10 = eval::recall_at_k(
            &state.out_keys,
            &state.out_counts,
            search_count,
            &state.ground_truth,
            10,
        );
        let ndcg10 = eval::ndcg_at_k(
            &state.out_keys,
            &state.out_counts,
            search_count,
            &state.ground_truth,
            10,
        );

        pb_search.finish_with_message(format!(
            "{} QPS, recall@1={recall1:.4}, recall@10={recall10:.4}, NDCG@10={ndcg10:.4} ({} vectors)",
            fmt_thousands(search_throughput),
            fmt_thousands(vectors_indexed as u64),
        ));

        emit(
            &mut state.writer,
            &StepRecord {
                description: description.clone(),
                phase: "search".to_string(),
                vectors_indexed,
                vectors_total: total_vectors,
                elapsed_seconds: Some(search_elapsed),
                vectors_per_second: None,
                queries_per_second: Some(search_throughput),
                memory_bytes: None,
                recall_at_1: Some(recall1),
                recall_at_10: Some(recall10),
                ndcg_at_10: Some(ndcg10),
            },
        )?;
    }

    eprintln!();
    Ok(())
}
