//! Synthetic dataset generator with SIMD-accelerated ground truth via NumKong.
//!
//! This file doubles as:
//! - the `retri-generate` binary (has a `main()`), and
//! - the `retrieval::generate` library module, exposing
//!   [`compute_hamming_top_k`], [`DEFAULT_QUERY_BATCH`], [`binary_view`], and
//!   [`matrix_span`] for the dataset-download binaries under `scripts/`.
//!
//! ## Build & Install
//!
//! ```sh
//! cargo install --path . --features generate
//! ```
//!
//! ## Examples
//!
//! Clustered binary dataset (1M base, 10K queries, 1024-bit vectors, hamming):
//! ```sh
//! retri-generate \
//!     --format b1bin \
//!     --base-count 1000000 \
//!     --query-count 10000 \
//!     --dims 1024 \
//!     --clusters 256 \
//!     --top-k 10 \
//!     --output datasets/binary_1M/
//! ```

use std::{collections::BinaryHeap, fmt, io::Write, path::PathBuf};

use clap::Parser;
use fork_union::{IndexedSplit, SyncMutPtr, ThreadPool};
use numkong::{MatrixSpan, MatrixView};
use rand::{rngs::StdRng, RngExt, SeedableRng};

use retrieval::{error::GroundTruthError, packed_distance::PackedDistance, Seed, Threads};

// #region Ground truth helpers — consumed by download scripts

/// Queries per distance-kernel call when `--ground-truth-batch` is not given.
pub const DEFAULT_QUERY_BATCH: usize = 4096;

/// Bytes of the per-call distance tile, enough cells for one kernel call to keep every core busy.
/// The base is walked in tiles of this size, so memory stays fixed however large the base grows.
const DISTANCE_TILE_BYTES: usize = 1 << 30;

/// Exact top-K via NumKong's packed distance kernels plus a ForkUnion-parallel
/// top-K pass (NumKong provides distances but no top-K).
///
/// `base` lists the parts of one base set, numbered on from each other.
/// All views must have contiguous rows; the base parts and `queries` must agree
/// on width, and `ground_truth` is `[query_count, top_k]`. `batch_size = None`
/// uses [`DEFAULT_QUERY_BATCH`]. Ties break toward the lower base index.
pub fn compute_top_k<Metric: PackedDistance>(
    base: &[MatrixView<'_, Metric>],
    queries: MatrixView<'_, Metric>,
    ground_truth: MatrixSpan<'_, u32>,
    batch_size: Option<usize>,
    threads: Threads,
) -> Result<(), GroundTruthError> {
    let batch = batch_size.unwrap_or(DEFAULT_QUERY_BATCH).min(queries.shape()[0].max(1));
    let tile_rows = DISTANCE_TILE_BYTES / (batch * std::mem::size_of::<Metric::Distance>());
    compute_top_k_tiled(base, queries, ground_truth, batch, tile_rows, threads)
}

/// [`compute_top_k`] with an explicit base tile, which tests shrink to cross tile boundaries.
fn compute_top_k_tiled<Metric: PackedDistance>(
    base: &[MatrixView<'_, Metric>],
    queries: MatrixView<'_, Metric>,
    mut ground_truth: MatrixSpan<'_, u32>,
    batch: usize,
    tile_rows: usize,
    threads: Threads,
) -> Result<(), GroundTruthError> {
    if !queries.has_contiguous_rows() {
        return Err(GroundTruthError::NonContiguousView { which: "queries" });
    }
    if !ground_truth.has_contiguous_rows() {
        return Err(GroundTruthError::NonContiguousView { which: "ground_truth" });
    }

    let dimensions = queries.shape()[1];
    for part in base {
        if !part.has_contiguous_rows() {
            return Err(GroundTruthError::NonContiguousView { which: "base" });
        }
        if part.shape()[1] != dimensions {
            return Err(GroundTruthError::DimensionMismatch {
                queries: dimensions,
                base: part.shape()[1],
            });
        }
    }
    let base_count: usize = base.iter().map(|part| part.shape()[0]).sum();
    if base_count > 1 << 32 {
        return Err(GroundTruthError::IndexOverflow { base_count });
    }
    let query_count = queries.shape()[0];
    if ground_truth.shape()[0] != query_count {
        return Err(GroundTruthError::RowCountMismatch {
            ground_truth_rows: ground_truth.shape()[0],
            query_rows: query_count,
        });
    }
    let top_k = ground_truth.shape()[1];
    if top_k == 0 || top_k > base_count {
        return Err(GroundTruthError::TopKTooLarge { top_k, base_count });
    }
    let threads = threads.0.get();
    let tile_rows = tile_rows.clamp(top_k, base_count);

    eprintln!(
        "  ground truth ({metric}): base_count={base_count}, query_count={query_count}, \
         dimensions={dimensions}, top_k={top_k}, batch_size={batch}, tile_rows={tile_rows}, threads={threads}",
        metric = Metric::metric_name()
    );

    let mut pool = ThreadPool::try_spawn(threads).map_err(|e| GroundTruthError::ThreadPool(format!("{e}")))?;

    // One reusable `[batch, tile_rows]` distance buffer; each call views a contiguous prefix of it.
    let mut distance_tensor =
        numkong::Matrix::<Metric::Distance>::try_full(&[batch, tile_rows], Metric::Distance::default())?;

    // Per-query size-K max-heaps keyed on `(OrderingKey, base_idx)`, carried across tiles.
    // `BinaryHeap` is a max-heap, so the largest kept distance is evicted first, and
    // the tuple order breaks ties toward the lower global index.
    let mut heaps: Vec<BinaryHeap<(Metric::OrderingKey, u32)>> =
        (0..query_count).map(|_| BinaryHeap::with_capacity(top_k + 1)).collect();
    let heaps_ptr = SyncMutPtr::new(heaps.as_mut_ptr());

    let progress = indicatif::ProgressBar::new(base_count as u64);
    progress.set_style(
        indicatif::ProgressStyle::with_template("  {msg} [{bar:40.cyan/blue}] {pos}/{len} ({per_sec}, ETA {eta})")
            .unwrap()
            .progress_chars("##-"),
    );
    progress.set_message(format!("ground truth ({})", Metric::metric_name()));

    let mut part_start = 0;
    for part in base {
        let part_count = part.shape()[0];
        for part_tile_start in (0..part_count).step_by(tile_rows) {
            let part_tile_end = (part_tile_start + tile_rows).min(part_count);
            let (tile_start, tile_end) = (part_start + part_tile_start, part_start + part_tile_end);
            let tile_count = tile_end - tile_start;
            let tile_tensor = padded_matrix(part.slice((part_tile_start..part_tile_end, ..))?)?;
            let packed_tile = numkong::PackedMatrix::pack(&tile_tensor);
            drop(tile_tensor);

            for batch_start in (0..query_count).step_by(batch) {
                let batch_end = (batch_start + batch).min(query_count);
                let batch_count = batch_end - batch_start;
                let query_tensor = padded_matrix(queries.slice((batch_start..batch_end, ..))?)?;

                let distances = &mut distance_tensor.as_mut_slice()[..batch_count * tile_count];
                let mut distance_span = matrix_span(distances, batch_count, tile_count);
                Metric::distances_parallel_into(&query_tensor, &packed_tile, &mut distance_span, &mut pool)?;

                let distances = &distance_tensor.as_slice()[..batch_count * tile_count];
                let split = IndexedSplit::new(batch_count, threads);
                pool.for_threads(|thread_index, _| {
                    for local_idx in split.get(thread_index) {
                        // SAFETY: `IndexedSplit` hands each query of the batch to exactly one thread.
                        let heap = unsafe { &mut *heaps_ptr.as_ptr().add(batch_start + local_idx) };
                        let row = &distances[local_idx * tile_count..][..tile_count];
                        for (tile_idx, &distance) in row.iter().enumerate() {
                            let key = Metric::ordering_key(distance);
                            let base_idx = (tile_start + tile_idx) as u32;
                            if heap.len() < top_k {
                                heap.push((key, base_idx));
                            } else if key < heap.peek().unwrap().0 {
                                heap.pop();
                                heap.push((key, base_idx));
                            }
                        }
                    }
                });
            }
            progress.set_position(tile_end as u64);
        }
        part_start += part_count;
    }
    progress.finish_and_clear();

    // Drain each heap into its output row in ascending-distance order.
    let gt_row_stride = (ground_truth.stride_bytes(0) as usize) / std::mem::size_of::<u32>();
    for (query_idx, heap) in heaps.iter_mut().enumerate() {
        // SAFETY: rows are contiguous and `query_idx < query_count` rows of `gt_row_stride` were checked above.
        let row: &mut [u32] =
            unsafe { std::slice::from_raw_parts_mut(ground_truth.as_mut_ptr().add(query_idx * gt_row_stride), top_k) };
        for rank in (0..top_k).rev() {
            // `top_k <= base_count` guarantees every heap holds exactly `top_k` entries.
            row[rank] = heap.pop().unwrap().1;
        }
    }
    Ok(())
}

/// Copies `rows` into an owned matrix whose width is zero-padded to a multiple of 64 dimensions.
/// NumKong 7.8.3's SME `f32` norm drops odd lanes of a partial tail vector, so every row ends on a full one.
fn padded_matrix<Metric: PackedDistance>(
    rows: MatrixView<'_, Metric>,
) -> Result<numkong::Matrix<Metric>, GroundTruthError> {
    let (count, dimensions) = (rows.shape()[0], rows.shape()[1]);
    let row_values = dimensions / Metric::dimensions_per_value();
    let padded_dimensions = dimensions.next_multiple_of(64);
    let row_stride = rows.stride_bytes(0) as usize / std::mem::size_of::<Metric>();
    // SAFETY: rows are contiguous, so `count` rows of `row_stride` values start at `as_ptr`.
    let source = unsafe { std::slice::from_raw_parts(rows.as_ptr(), count * row_stride) };
    let mut padded = numkong::Matrix::<Metric>::try_full(&[count, padded_dimensions], Metric::zero())?;
    let padded_rows = padded
        .as_mut_slice()
        .chunks_exact_mut(padded_dimensions / Metric::dimensions_per_value());
    for (target, row) in padded_rows.zip(source.chunks_exact(row_stride)) {
        target[..row_values].copy_from_slice(&row[..row_values]);
    }
    Ok(padded)
}

/// Backward-compat alias for the Hamming path. Preferred for new callers:
/// use [`compute_top_k::<numkong::u1x8>`] directly.
pub fn compute_hamming_top_k(
    base: &[MatrixView<'_, numkong::u1x8>],
    queries: MatrixView<'_, numkong::u1x8>,
    ground_truth: MatrixSpan<'_, u32>,
    batch_size: Option<usize>,
    threads: Threads,
) -> Result<(), GroundTruthError> {
    compute_top_k::<numkong::u1x8>(base, queries, ground_truth, batch_size, threads)
}

/// Build a zero-copy `MatrixView<u1x8>` of shape `[rows, bits_per_vector]`
/// over a flat contiguous byte buffer. Convenience helper for callers that
/// already hold bit-packed data as `&[u8]`.
///
/// # Panics
///
/// - if `bits_per_vector % 8 != 0` (the `u1x8` storage requires byte-aligned
///   logical dimensions);
/// - if `bytes.len() != rows * (bits_per_vector / 8)` (buffer shape must
///   match the declared matrix shape exactly — no trailing padding, no
///   underfill).
pub fn binary_view<'a>(bytes: &'a [u8], rows: usize, bits_per_vector: usize) -> MatrixView<'a, numkong::u1x8> {
    assert_eq!(
        bits_per_vector % 8,
        0,
        "bits_per_vector ({bits_per_vector}) must be a multiple of 8 for u1x8 storage"
    );
    let bytes_per_vector = bits_per_vector / 8;
    assert_eq!(
        bytes.len(),
        rows * bytes_per_vector,
        "buffer length {} doesn't match rows*bytes_per_vector = {}",
        bytes.len(),
        rows * bytes_per_vector
    );
    let data = bytes.as_ptr() as *const numkong::u1x8;
    let row_stride_bytes = bytes_per_vector as isize;
    unsafe { MatrixView::from_raw_parts(data, &[rows, bits_per_vector], &[row_stride_bytes, 1]) }
}

/// Build a read-only `MatrixView<T>` of shape `[rows, cols]` over a flat
/// slice. Mirror of [`matrix_span`] for the immutable case.
///
/// # Panics
///
/// Panics if `data.len() != rows * cols` — buffer shape must match the
/// declared matrix shape exactly.
pub fn matrix_view<'a, T>(data: &'a [T], rows: usize, cols: usize) -> MatrixView<'a, T> {
    assert_eq!(
        data.len(),
        rows * cols,
        "buffer length {} doesn't match rows*cols = {}",
        data.len(),
        rows * cols
    );
    let ptr = data.as_ptr();
    let inner = std::mem::size_of::<T>() as isize;
    let row = (cols as isize) * inner;
    unsafe { MatrixView::from_raw_parts(ptr, &[rows, cols], &[row, inner]) }
}

/// Build a `MatrixSpan<T>` of shape `[rows, cols]` over a flat mutable slice.
/// Convenience helper mirroring [`binary_view`] for non-bit-packed types.
///
/// # Panics
///
/// Panics if `data.len() != rows * cols` — buffer shape must match the
/// declared matrix shape exactly.
pub fn matrix_span<'a, T>(data: &'a mut [T], rows: usize, cols: usize) -> MatrixSpan<'a, T> {
    assert_eq!(
        data.len(),
        rows * cols,
        "buffer length {} doesn't match rows*cols = {}",
        data.len(),
        rows * cols
    );
    let ptr = data.as_mut_ptr();
    let inner = std::mem::size_of::<T>() as isize;
    let row = (cols as isize) * inner;
    unsafe { MatrixSpan::from_raw_parts(ptr, &[rows, cols], &[row, inner]) }
}

// #endregion

// #region Binary — synthetic clustered binary dataset
//
// Everything below is used only when this file is compiled as the
// `retri-generate` binary. The library compilation (`pub mod generate` in
// `bench.rs`) sees these as dead code, which is expected.

#[allow(dead_code)]
#[derive(Parser, Debug)]
#[command(name = "retri-generate", about = "Generate synthetic vector datasets")]
struct Cli {
    /// Output format: `b1bin` (clustered binary, Hamming GT) or
    /// `fbin` (Gaussian f32, L2 GT)
    #[arg(long, value_enum)]
    format: Format,

    /// Number of base vectors
    #[arg(long, value_parser = retrieval::parse_count_flag)]
    base_count: usize,

    /// Number of query vectors
    #[arg(long, value_parser = retrieval::parse_count_flag)]
    query_count: usize,

    /// Vector dimensions. For `b1bin` this is the bit count (must be a
    /// multiple of 8); for `fbin` it's the f32 scalar count.
    #[arg(long, value_parser = retrieval::parse_count_flag)]
    dims: usize,

    /// Number of cluster centers for non-uniform data
    #[arg(long, default_value_t = 256, value_parser = retrieval::parse_count_flag)]
    clusters: usize,

    /// Bit-flip probability for cluster noise (0.0 = identical to center, 0.5 = random)
    #[arg(long, default_value_t = 0.1, value_parser = parse_probability)]
    noise: f64,

    /// Neighbors to compute per query (the k in the ground-truth file)
    #[arg(long, default_value_t = 10, value_parser = retrieval::parse_count_flag)]
    top_k: usize,

    /// Queries per ground-truth kernel call; memory stays bounded by the distance tile either way.
    #[arg(long, value_parser = retrieval::parse_count_flag)]
    ground_truth_batch: Option<usize>,

    /// Threads for ground-truth top-K extraction (defaults to all logical cores).
    #[arg(long, default_value = "0", value_parser = retrieval::parse_threads_flag)]
    threads: Threads,

    /// Random seed, or `random` to draw one
    #[arg(long, default_value = "42", value_parser = retrieval::parse_seed_flag)]
    seed: Seed,

    /// Output directory
    #[arg(long)]
    output: PathBuf,
}

/// Parses a probability like `0.1`, from 0 to 1 in plain digits.
#[allow(dead_code)]
fn parse_probability(text: &str) -> Result<f64, String> {
    let digits = text.bytes().all(|byte| byte.is_ascii_digit() || byte == b'.');
    digits
        .then(|| text.parse().ok())
        .flatten()
        .filter(|probability| (0.0..=1.0).contains(probability))
        .ok_or_else(|| "expected a probability like 0.1".into())
}

/// Generated dataset layouts, as `--format` spells them.
#[allow(dead_code)]
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum Format {
    B1bin,
    Fbin,
}

impl fmt::Display for Format {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        retrieval::spell_value(self, formatter)
    }
}

/// Write the 8-byte BigANN-style header: two little-endian `u32` values
/// encoding row count and dimension count, in that order. Matches the
/// format produced by Meta / Microsoft / Yandex for `.fbin`, `.u8bin`,
/// `.i8bin`, `.b1bin`, and `.ibin` files.
#[allow(dead_code)]
fn write_bin_header(file: &mut std::fs::File, rows: u32, dimensions: u32) -> std::io::Result<()> {
    file.write_all(&rows.to_le_bytes())?;
    file.write_all(&dimensions.to_le_bytes())
}

/// Generate `count` clustered binary vectors row-major in `[count, bytes_per_vector]` layout.
///
/// For each vector we pick a random center from `centers` (which stores
/// `num_clusters` centers packed row-major as `[num_clusters, bytes_per_vector]`),
/// then XOR it with a per-bit noise mask. Each bit in the mask is set with
/// probability `noise` — implemented by drawing a `u8` uniform and comparing
/// against the threshold `(noise * 256.0) as u8`. This keeps the hot loop
/// branch-free beyond the one compare and gives a resolution of `1/256`
/// around the target probability, which is plenty for benchmark datasets.
///
/// The returned `Vec<u8>` is flat but logically `[count, bytes_per_vector]`:
/// vector `i` lives at `data[i * bytes_per_vector .. (i + 1) * bytes_per_vector]`.
#[allow(dead_code)]
fn generate_clustered_binary(
    rng: &mut StdRng,
    count: usize,
    bytes_per_vector: usize,
    centers: &[u8],
    num_clusters: usize,
    noise: f64,
) -> Vec<u8, std::alloc::System> {
    let noise_threshold = (noise * 256.0) as u8;
    let mut base_bytes = {
        let mut values = Vec::new_in(std::alloc::System);
        values.resize(count * bytes_per_vector, 0u8);
        values
    };
    for vector_index in 0..count {
        let center_index = rng.random_range(0..num_clusters);
        let center = &centers[center_index * bytes_per_vector..(center_index + 1) * bytes_per_vector];
        let vector = &mut base_bytes[vector_index * bytes_per_vector..(vector_index + 1) * bytes_per_vector];
        for (byte_index, &center_byte) in center.iter().enumerate() {
            let mut noise_mask = 0u8;
            for bit_index in 0..8u8 {
                if rng.random::<u8>() < noise_threshold {
                    noise_mask |= 1 << bit_index;
                }
            }
            vector[byte_index] = center_byte ^ noise_mask;
        }
    }
    base_bytes
}

/// Generate `count` row-major Gaussian f32 vectors of dimension `dimensions`,
/// drawn from the standard normal distribution (mean 0, variance 1).
///
/// # Algorithm
///
/// Uses the Box–Muller transform: each iteration draws two uniform(0,1]
/// samples and emits two independent N(0, 1) values. We hand-roll it rather
/// than pull `rand_distr` just for `StandardNormal`.
///
/// The first uniform sample (`uniform_radius`) is floored at `f32::EPSILON`
/// before the `ln()` — `ln(0)` is `-∞`, which propagates to `NaN` through
/// the `sqrt()`. The bias introduced by this clamp is negligible
/// (`EPSILON ≈ 1.19e-7`, so roughly one sample in every 8.4M would have
/// been clamped; the output density inside the clamp region is essentially
/// zero already).
///
/// # Output shape
///
/// `Vec<f32>` of length `count * dimensions`, row-major: row `i` occupies
/// `data[i * dimensions .. (i + 1) * dimensions]`. Matches the `.fbin` file layout that
/// follows this function's output.
///
/// # Odd-length tail
///
/// If `count * dimensions` is odd, one extra Box–Muller pair is generated and
/// the second sample is discarded. Costs one surplus `ln/sqrt/cos`.
#[allow(dead_code)]
fn generate_gaussian_f32(rng: &mut StdRng, count: usize, dimensions: usize) -> Vec<f32, std::alloc::System> {
    let scalar_count = count * dimensions;
    let mut base_bytes = {
        let mut values = Vec::new_in(std::alloc::System);
        values.resize(scalar_count, 0f32);
        values
    };
    let mut cursor = 0;
    while cursor + 1 < scalar_count {
        let (z_cos, z_sin) = sample_standard_normal_pair(rng);
        base_bytes[cursor] = z_cos;
        base_bytes[cursor + 1] = z_sin;
        cursor += 2;
    }
    if cursor < scalar_count {
        // Odd tail: waste one paired sample.
        let (z_cos, _z_sin) = sample_standard_normal_pair(rng);
        base_bytes[cursor] = z_cos;
    }
    base_bytes
}

/// One iteration of Box–Muller — two independent N(0, 1) samples from two
/// uniform(0, 1] samples. See [`generate_gaussian_f32`] for the EPSILON
/// clamp rationale.
#[allow(dead_code)]
#[inline]
fn sample_standard_normal_pair(rng: &mut StdRng) -> (f32, f32) {
    let uniform_radius = rng.random::<f32>().max(f32::EPSILON);
    let uniform_angle = rng.random::<f32>();
    let radius = (-2.0 * uniform_radius.ln()).sqrt();
    let angle = 2.0 * std::f32::consts::PI * uniform_angle;
    (radius * angle.cos(), radius * angle.sin())
}

/// Parse CLI args and dispatch to the format-specific generator. Output
/// directory is created up front; format-specific validation (dimensions
/// alignment, etc.) happens inside the per-format runner.
#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli: Cli = retrieval::parse_cli();
    eprintln!("- Format: {}", cli.format);
    eprintln!("- Base count: {}", cli.base_count);
    eprintln!("- Query count: {}", cli.query_count);
    eprintln!("- Dims: {}", cli.dims);
    eprintln!("- Clusters: {}", cli.clusters);
    eprintln!("- Noise: {}", cli.noise);
    eprintln!("- Top k: {}", cli.top_k);
    eprintln!(
        "- Ground-truth batch: {}",
        cli.ground_truth_batch.unwrap_or(DEFAULT_QUERY_BATCH)
    );
    eprintln!("- Threads: {}", cli.threads);
    eprintln!("- Seed: {}", cli.seed);
    eprintln!("- Output: {}", cli.output.display());
    std::fs::create_dir_all(&cli.output)?;

    match cli.format {
        Format::B1bin => run_b1bin(&cli),
        Format::Fbin => run_fbin(&cli),
    }
}

/// Run the clustered-binary (`b1bin`) pipeline end-to-end: generate cluster
/// centers → generate noisy base + query vectors → compute Hamming top-K
/// ground truth → write `base.N.b1bin`, `query.M.b1bin`, `groundtruth.M.ibin`.
#[allow(dead_code)]
fn run_b1bin(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    if !cli.dims.is_multiple_of(8) {
        return Err("--dims must be a multiple of 8 for b1bin format".into());
    }
    let bytes_per_vector = cli.dims / 8;
    let mut rng = StdRng::seed_from_u64(cli.seed.into());

    eprintln!(
        "Generating {} cluster centers ({} bits = {} bytes each)...",
        cli.clusters, cli.dims, bytes_per_vector
    );
    let mut centers = {
        let mut values = Vec::new_in(std::alloc::System);
        values.resize(cli.clusters * bytes_per_vector, 0u8);
        values
    };
    rng.fill(&mut centers[..]);

    eprintln!("Generating {} base vectors (noise={})...", cli.base_count, cli.noise);
    let base = generate_clustered_binary(
        &mut rng,
        cli.base_count,
        bytes_per_vector,
        &centers,
        cli.clusters,
        cli.noise,
    );

    eprintln!("Generating {} query vectors...", cli.query_count);
    let queries = generate_clustered_binary(
        &mut rng,
        cli.query_count,
        bytes_per_vector,
        &centers,
        cli.clusters,
        cli.noise,
    );

    eprintln!(
        "Computing brute-force hamming top-{} ground truth (NumKong + ForkUnion)...",
        cli.top_k
    );
    let base_view = binary_view(&base, cli.base_count, cli.dims);
    let query_view = binary_view(&queries, cli.query_count, cli.dims);
    let mut ground_truth_storage = {
        let mut values = Vec::new_in(std::alloc::System);
        values.resize(cli.query_count * cli.top_k, 0u32);
        values
    };
    let ground_truth_span = matrix_span(&mut ground_truth_storage, cli.query_count, cli.top_k);
    compute_hamming_top_k(
        &[base_view],
        query_view,
        ground_truth_span,
        cli.ground_truth_batch,
        cli.threads,
    )?;

    write_dataset::<u8>(cli, "b1bin", &base, &queries, &ground_truth_storage)
}

/// Run the Gaussian-f32 (`fbin`) pipeline end-to-end: generate standard-normal
/// base + query vectors → compute L2 top-K ground truth → write
/// `base.N.fbin`, `query.M.fbin`, `groundtruth.M.ibin`.
#[allow(dead_code)]
fn run_fbin(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = StdRng::seed_from_u64(cli.seed.into());

    eprintln!(
        "Generating {} base vectors (f32, {} dimensions)...",
        cli.base_count, cli.dims
    );
    let base = generate_gaussian_f32(&mut rng, cli.base_count, cli.dims);

    eprintln!("Generating {} query vectors...", cli.query_count);
    let queries = generate_gaussian_f32(&mut rng, cli.query_count, cli.dims);

    eprintln!(
        "Computing brute-force L2 top-{} ground truth (NumKong + ForkUnion)...",
        cli.top_k
    );
    let base_view = matrix_view(&base, cli.base_count, cli.dims);
    let query_view = matrix_view(&queries, cli.query_count, cli.dims);
    let mut ground_truth = {
        let mut values = Vec::new_in(std::alloc::System);
        values.resize(cli.query_count * cli.top_k, 0u32);
        values
    };
    let ground_truth_span = matrix_span(&mut ground_truth, cli.query_count, cli.top_k);
    compute_top_k::<f32>(
        &[base_view],
        query_view,
        ground_truth_span,
        cli.ground_truth_batch,
        cli.threads,
    )?;

    write_dataset::<f32>(cli, "fbin", &base, &queries, &ground_truth)
}

/// Write the three-file contract for a generated dataset:
/// `base.N.<extension>`  (row-major vectors of element type `T`),
/// `query.M.<extension>` (ditto),
/// `groundtruth.M.ibin`  (`u32` neighbor IDs, top-K per query).
///
/// Each file starts with the 8-byte header from [`write_bin_header`].
/// The `T: Copy` bound plus the `pod_slice_as_bytes` contract means T must
/// be a POD type — `u8`, `f32`, `i8`, etc.
#[allow(dead_code)]
fn write_dataset<T: Copy>(
    cli: &Cli,
    extension: &str,
    base: &[T],
    queries: &[T],
    ground_truth: &[u32],
) -> Result<(), Box<dyn std::error::Error>> {
    let base_path = cli.output.join(format!("base.{}.{extension}", cli.base_count));
    let query_path = cli.output.join(format!("query.{}.{extension}", cli.query_count));
    let gt_path = cli.output.join(format!("groundtruth.{}.ibin", cli.query_count));

    eprintln!("Writing {}", base_path.display());
    let mut file = std::fs::File::create(&base_path)?;
    write_bin_header(&mut file, cli.base_count as u32, cli.dims as u32)?;
    // SAFETY: `T: Copy` is our stand-in for POD; the caller picks `T` from
    // {u8, f32, i8, u32} all of which have no padding and no invalid bit
    // patterns. `pod_slice_as_bytes` just reinterprets the slice's bytes.
    file.write_all(unsafe { retrieval::pod_slice_as_bytes(base) })?;

    eprintln!("Writing {}", query_path.display());
    let mut file = std::fs::File::create(&query_path)?;
    write_bin_header(&mut file, cli.query_count as u32, cli.dims as u32)?;
    file.write_all(unsafe { retrieval::pod_slice_as_bytes(queries) })?;

    eprintln!("Writing {}", gt_path.display());
    let mut file = std::fs::File::create(&gt_path)?;
    write_bin_header(&mut file, cli.query_count as u32, cli.top_k as u32)?;
    file.write_all(unsafe { retrieval::pod_slice_as_bytes(ground_truth) })?;

    let base_mb = (8 + std::mem::size_of_val(base)) as f64 / (1u64 << 20) as f64;
    eprintln!("Done! base: {base_mb:.1} MB, {} output files", cli.output.display());
    Ok(())
}

// #endregion

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use rand::{rngs::StdRng, RngExt, SeedableRng};

    use super::{binary_view, compute_top_k_tiled, generate_gaussian_f32, matrix_span, matrix_view};
    use retrieval::Threads;

    const BASE_COUNT: usize = 300;
    const QUERY_COUNT: usize = 20;
    const TOP_K: usize = 10;
    const PART_ROWS: usize = 137;

    /// Runs the tiled top-K across several query batches and base tiles, the last tile short,
    /// and returns each query's neighbor ids.
    fn top_k<Metric: retrieval::packed_distance::PackedDistance>(
        base: &[numkong::MatrixView<'_, Metric>],
        queries: numkong::MatrixView<'_, Metric>,
    ) -> Vec<u32> {
        let mut ground_truth = vec![0u32; QUERY_COUNT * TOP_K];
        let span = matrix_span(&mut ground_truth, QUERY_COUNT, TOP_K);
        compute_top_k_tiled(base, queries, span, 7, 64, Threads(NonZeroUsize::new(2).unwrap())).unwrap();
        ground_truth
    }

    /// Ascending `(distance, index)` order of `distances`, cut to `TOP_K`.
    fn naive_top_k<Distance: PartialOrd + Copy>(distances: &[Distance]) -> Vec<usize> {
        let mut order: Vec<usize> = (0..distances.len()).collect();
        order.sort_by(|&a, &b| distances[a].partial_cmp(&distances[b]).unwrap().then(a.cmp(&b)));
        order.truncate(TOP_K);
        order
    }

    #[test]
    fn f32_top_k_matches_scalar_l2_at_tail_widths() {
        for dimensions in [2, 3, 7, 20, 64, 100] {
            let mut rng = StdRng::seed_from_u64(dimensions as u64);
            let base = generate_gaussian_f32(&mut rng, BASE_COUNT, dimensions);
            let queries = generate_gaussian_f32(&mut rng, QUERY_COUNT, dimensions);
            let found = top_k(
                &[matrix_view(&base, BASE_COUNT, dimensions)],
                matrix_view(&queries, QUERY_COUNT, dimensions),
            );
            for (query, found) in queries.chunks_exact(dimensions).zip(found.as_chunks::<TOP_K>().0) {
                let distances: Vec<f64> = base
                    .chunks_exact(dimensions)
                    .map(|row| {
                        row.iter()
                            .zip(query)
                            .map(|(&b, &q)| (b as f64 - q as f64).powi(2))
                            .sum()
                    })
                    .collect();
                let expected = naive_top_k(&distances);
                for (&id, &want) in found.iter().zip(&expected) {
                    let (got, want) = (distances[id as usize], distances[want]);
                    assert!(
                        (got - want).abs() <= 1e-9 * want.max(1.0),
                        "dims {dimensions}: {got} != {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn b1bin_top_k_matches_scalar_hamming() {
        for bits in [8, 24, 136] {
            let bytes = bits / 8;
            let mut rng = StdRng::seed_from_u64(bits as u64);
            let base: Vec<u8> = (0..BASE_COUNT * bytes).map(|_| rng.random()).collect();
            let queries: Vec<u8> = (0..QUERY_COUNT * bytes).map(|_| rng.random()).collect();
            // Two base parts, split off a tile boundary, so ids carry on across the parts.
            let (head, tail) = base.split_at(PART_ROWS * bytes);
            let found = top_k(
                &[
                    binary_view(head, PART_ROWS, bits),
                    binary_view(tail, BASE_COUNT - PART_ROWS, bits),
                ],
                binary_view(&queries, QUERY_COUNT, bits),
            );
            for (query, found) in queries.chunks_exact(bytes).zip(found.as_chunks::<TOP_K>().0) {
                let distances: Vec<u32> = base
                    .chunks_exact(bytes)
                    .map(|row| row.iter().zip(query).map(|(&b, &q)| (b ^ q).count_ones()).sum())
                    .collect();
                let found: Vec<usize> = found.iter().map(|&id| id as usize).collect();
                assert_eq!(found, naive_top_k(&distances), "bits {bits}");
            }
        }
    }
}
