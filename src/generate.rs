//! Synthetic dataset generator, with exact ground truth from `retrieval::ground_truth`.
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

use std::{fmt, path::PathBuf};

use clap::Parser;
use numkong as nk;
use rand::{rngs::StdRng, RngExt, SeedableRng};

use retrieval::{
    ground_truth::{compute_top_k, write_bin_file},
    pod_slice_as_bytes, Seed, Threads,
};

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
fn parse_probability(text: &str) -> Result<f64, String> {
    let digits = text.bytes().all(|byte| byte.is_ascii_digit() || byte == b'.');
    digits
        .then(|| text.parse().ok())
        .flatten()
        .filter(|probability| (0.0..=1.0).contains(probability))
        .ok_or_else(|| "expected a probability like 0.1".into())
}

/// Generated dataset layouts, as `--format` spells them.
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

/// `count` clustered binary vectors, `bits` wide: each copies a random row of `centers` and flips
/// every bit with probability `noise`, at a resolution of 1/256 from one `u8` draw per bit.
fn generate_clustered_binary(
    random: &mut StdRng,
    count: usize,
    bits: usize,
    centers: &[u8],
    noise: f64,
) -> Result<nk::Matrix<nk::u1x8>, nk::Error> {
    let noise_threshold = (noise * 256.0) as u8;
    let bytes_per_vector = bits / 8;
    let clusters = centers.len() / bytes_per_vector;
    let mut vectors = nk::Matrix::full(&[count, bits], nk::u1x8(0))?;
    for vector_index in 0..count {
        let center_index = random.random_range(0..clusters);
        let center = centers
            .chunks_exact(bytes_per_vector)
            .nth(center_index)
            .expect("a center row");
        let vector = vectors.row_mut(vector_index).expect("a vector row");
        for (byte, &center_byte) in vector.iter_mut().zip(center) {
            let noise_mask = (0..8)
                .filter(|_| random.random::<u8>() < noise_threshold)
                .fold(0u8, |mask, bit| mask | 1 << bit);
            *byte = nk::u1x8(center_byte ^ noise_mask);
        }
    }
    Ok(vectors)
}

/// `count` vectors of `dimensions` standard-normal `f32` values, drawn in Box–Muller pairs rather
/// than through `rand_distr`; an odd total discards the last pair's second sample.
fn generate_gaussian_f32(random: &mut StdRng, count: usize, dimensions: usize) -> Result<nk::Matrix<f32>, nk::Error> {
    let mut vectors = nk::Matrix::full(&[count, dimensions], 0.0f32)?;
    for pair in vectors.as_mut_slice().chunks_mut(2) {
        let (cosine, sine) = sample_standard_normal_pair(random);
        pair[0] = cosine;
        if let Some(second) = pair.get_mut(1) {
            *second = sine;
        }
    }
    Ok(vectors)
}

/// One Box–Muller step: two independent N(0, 1) samples from two uniform(0, 1] samples. The radius
/// draw is floored at `f32::EPSILON`, since `ln(0)` is −∞ and would turn into `NaN`.
#[inline]
fn sample_standard_normal_pair(random: &mut StdRng) -> (f32, f32) {
    let uniform_radius = random.random::<f32>().max(f32::EPSILON);
    let uniform_angle = random.random::<f32>();
    let radius = (-2.0 * uniform_radius.ln()).sqrt();
    let angle = 2.0 * std::f32::consts::PI * uniform_angle;
    (radius * angle.cos(), radius * angle.sin())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli: Cli = retrieval::parse_cli();
    eprintln!("- Format: {}", cli.format);
    eprintln!("- Base count: {}", cli.base_count);
    eprintln!("- Query count: {}", cli.query_count);
    eprintln!("- Dims: {}", cli.dims);
    eprintln!("- Clusters: {}", cli.clusters);
    eprintln!("- Noise: {}", cli.noise);
    eprintln!("- Top k: {}", cli.top_k);
    eprintln!("- Threads: {}", cli.threads);
    eprintln!("- Seed: {}", cli.seed);
    eprintln!("- Output: {}", cli.output.display());
    std::fs::create_dir_all(&cli.output)?;

    match cli.format {
        Format::B1bin => run_b1bin(&cli),
        Format::Fbin => run_fbin(&cli),
    }
}

/// Clustered binary base and queries with exact Hamming ground truth.
fn run_b1bin(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    if !cli.dims.is_multiple_of(8) {
        return Err("--dims must be a multiple of 8 for b1bin format".into());
    }
    let bytes_per_vector = cli.dims / 8;
    let mut random = StdRng::seed_from_u64(cli.seed.into());

    eprintln!(
        "Generating {} cluster centers ({} bits = {} bytes each)...",
        cli.clusters, cli.dims, bytes_per_vector
    );
    let mut centers = Vec::with_capacity_in(cli.clusters * bytes_per_vector, std::alloc::System);
    centers.resize(cli.clusters * bytes_per_vector, 0u8);
    random.fill(&mut centers[..]);

    eprintln!("Generating {} base vectors (noise={})...", cli.base_count, cli.noise);
    let base = generate_clustered_binary(&mut random, cli.base_count, cli.dims, &centers, cli.noise)?;
    eprintln!("Generating {} query vectors...", cli.query_count);
    let queries = generate_clustered_binary(&mut random, cli.query_count, cli.dims, &centers, cli.noise)?;

    eprintln!(
        "Computing brute-force hamming top-{} ground truth (NumKong + ForkUnion)...",
        cli.top_k
    );
    let mut ground_truth = nk::Matrix::full(&[cli.query_count, cli.top_k], 0u32)?;
    compute_top_k(&[base.view()], queries.view(), ground_truth.span(), cli.threads)?;
    write_dataset(cli, "b1bin", &base, &queries, &ground_truth)
}

/// Gaussian `f32` base and queries with exact L2 ground truth.
fn run_fbin(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    let mut random = StdRng::seed_from_u64(cli.seed.into());

    eprintln!(
        "Generating {} base vectors (f32, {} dimensions)...",
        cli.base_count, cli.dims
    );
    let base = generate_gaussian_f32(&mut random, cli.base_count, cli.dims)?;
    eprintln!("Generating {} query vectors...", cli.query_count);
    let queries = generate_gaussian_f32(&mut random, cli.query_count, cli.dims)?;

    eprintln!(
        "Computing brute-force L2 top-{} ground truth (NumKong + ForkUnion)...",
        cli.top_k
    );
    let mut ground_truth = nk::Matrix::full(&[cli.query_count, cli.top_k], 0u32)?;
    compute_top_k(&[base.view()], queries.view(), ground_truth.span(), cli.threads)?;
    write_dataset(cli, "fbin", &base, &queries, &ground_truth)
}

/// Writes `base.N.<extension>`, `query.M.<extension>` and `groundtruth.M.ibin`.
fn write_dataset<T: nk::StorageElement + Copy>(
    cli: &Cli,
    extension: &str,
    base: &nk::Matrix<T>,
    queries: &nk::Matrix<T>,
    ground_truth: &nk::Matrix<u32>,
) -> Result<(), Box<dyn std::error::Error>> {
    let output = |name: String| cli.output.join(name);
    // SAFETY: the generated scalars and the `u32` neighbor ids are POD.
    let (base_bytes, query_bytes, ground_truth_bytes) = unsafe {
        (
            pod_slice_as_bytes(base.as_slice()),
            pod_slice_as_bytes(queries.as_slice()),
            pod_slice_as_bytes(ground_truth.as_slice()),
        )
    };
    let base_path = output(format!("base.{}.{extension}", cli.base_count));
    write_bin_file(&base_path, cli.base_count, cli.dims, base_bytes)?;
    let query_path = output(format!("query.{}.{extension}", cli.query_count));
    write_bin_file(&query_path, cli.query_count, cli.dims, query_bytes)?;
    let ground_truth_path = output(format!("groundtruth.{}.ibin", cli.query_count));
    write_bin_file(&ground_truth_path, cli.query_count, cli.top_k, ground_truth_bytes)?;
    eprintln!(
        "Done! base: {:.1} MB",
        (8 + base_bytes.len()) as f64 / (1u64 << 20) as f64
    );
    Ok(())
}
