//! Synthetic dataset generator with SIMD-accelerated ground truth via NumKong.
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
//!     --dimensions 1024 \
//!     --clusters 256 \
//!     --neighbors 10 \
//!     --output datasets/binary_1M/
//! ```

use std::io::Write;
use std::path::PathBuf;

use clap::Parser;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Parser, Debug)]
#[command(name = "retri-generate", about = "Generate synthetic vector datasets")]
struct Cli {
    /// Output format: b1bin
    #[arg(long)]
    format: String,

    /// Number of base vectors
    #[arg(long)]
    base_count: usize,

    /// Number of query vectors
    #[arg(long)]
    query_count: usize,

    /// Vector dimensions (bits for b1bin)
    #[arg(long)]
    dimensions: usize,

    /// Number of cluster centers for non-uniform data
    #[arg(long, default_value_t = 256)]
    clusters: usize,

    /// Bit-flip probability for cluster noise (0.0 = identical to center, 0.5 = random)
    #[arg(long, default_value_t = 0.1)]
    noise: f64,

    /// Number of ground-truth neighbors per query
    #[arg(long, default_value_t = 10)]
    neighbors: usize,

    /// Ground-truth query batch size (memory vs speed tradeoff)
    #[arg(long, default_value_t = 100)]
    ground_truth_batch: usize,

    /// Random seed
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Output directory
    #[arg(long)]
    output: PathBuf,
}

fn write_bin_header(file: &mut std::fs::File, rows: u32, dims: u32) -> std::io::Result<()> {
    file.write_all(&rows.to_le_bytes())?;
    file.write_all(&dims.to_le_bytes())
}

/// Generate clustered binary vectors: pick a random center and flip bits with `noise` probability.
fn generate_clustered_binary(
    rng: &mut StdRng,
    count: usize,
    bytes_per_vector: usize,
    centers: &[u8],
    num_clusters: usize,
    noise: f64,
) -> Vec<u8> {
    let noise_threshold = (noise * 256.0) as u8;
    let mut data = vec![0u8; count * bytes_per_vector];
    for i in 0..count {
        let center_idx = rng.random_range(0..num_clusters);
        let center = &centers[center_idx * bytes_per_vector..(center_idx + 1) * bytes_per_vector];
        let vector = &mut data[i * bytes_per_vector..(i + 1) * bytes_per_vector];
        for (byte_idx, &center_byte) in center.iter().enumerate() {
            let mut noise_mask = 0u8;
            for bit in 0..8u8 {
                if (rng.random::<u8>()) < noise_threshold {
                    noise_mask |= 1 << bit;
                }
            }
            vector[byte_idx] = center_byte ^ noise_mask;
        }
    }
    data
}

/// Compute brute-force hamming top-k ground truth using NumKong's SIMD kernels.
fn compute_hamming_ground_truth(
    base: &[u8],
    queries: &[u8],
    base_count: usize,
    query_count: usize,
    bytes_per_vector: usize,
    top_k: usize,
    batch_size: usize,
) -> Vec<u32> {
    // Cast &[u8] → &[numkong::u1x8] (repr(transparent), zero-cost).
    // Tensor shape uses bit dimensions — NumKong divides by dimensions_per_value()==8 internally.
    let dimensions_bits = bytes_per_vector * 8;
    let base_b1: &[numkong::u1x8] =
        unsafe { std::slice::from_raw_parts(base.as_ptr() as *const numkong::u1x8, base.len()) };

    let base_tensor =
        numkong::Tensor::<numkong::u1x8>::from_slice(base_b1, &[base_count, dimensions_bits]);
    let packed_base = numkong::PackedMatrix::pack(&base_tensor);

    let mut ground_truth = vec![0u32; query_count * top_k];
    let start = std::time::Instant::now();

    for batch_start in (0..query_count).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(query_count);
        let batch_count = batch_end - batch_start;

        let query_slice = &queries[batch_start * bytes_per_vector..batch_end * bytes_per_vector];
        let query_b1: &[numkong::u1x8] = unsafe {
            std::slice::from_raw_parts(
                query_slice.as_ptr() as *const numkong::u1x8,
                query_slice.len(),
            )
        };

        let query_tensor =
            numkong::Tensor::<numkong::u1x8>::from_slice(query_b1, &[batch_count, dimensions_bits]);
        let distances = query_tensor.hammings_packed(&packed_base);
        let distances_flat = distances.as_slice();

        // Extract top-k for each query in this batch
        for local_idx in 0..batch_count {
            let query_idx = batch_start + local_idx;
            let row_offset = local_idx * base_count;

            // Collect (distance, index) pairs and partial-sort for top-k
            let mut candidates: Vec<(u32, u32)> = (0..base_count)
                .map(|j| (distances_flat[row_offset + j], j as u32))
                .collect();

            candidates.select_nth_unstable_by_key(top_k - 1, |&(dist, _)| dist);
            candidates.truncate(top_k);
            candidates.sort_by_key(|&(dist, _)| dist);

            let gt_offset = query_idx * top_k;
            for (rank, &(_, idx)) in candidates.iter().enumerate() {
                ground_truth[gt_offset + rank] = idx;
            }
        }

        let elapsed = start.elapsed().as_secs_f64();
        let done = batch_end;
        let rate = done as f64 / elapsed;
        let eta = (query_count - done) as f64 / rate;
        eprint!("\r  ground truth: {done}/{query_count} ({rate:.0} queries/s, ETA {eta:.0}s)   ");
    }
    eprintln!();

    ground_truth
}

fn main() {
    let cli = Cli::parse();

    match cli.format.as_str() {
        "b1bin" => {}
        other => {
            eprintln!("unsupported format: {other} (supported: b1bin)");
            std::process::exit(1);
        }
    }

    if cli.dimensions % 8 != 0 {
        eprintln!("--dimensions must be a multiple of 8 for b1bin format");
        std::process::exit(1);
    }

    let bytes_per_vector = cli.dimensions / 8;
    let mut rng = StdRng::seed_from_u64(cli.seed);

    std::fs::create_dir_all(&cli.output).unwrap_or_else(|e| {
        eprintln!("failed to create output directory: {e}");
        std::process::exit(1);
    });

    // Generate cluster centers
    eprintln!(
        "Generating {} cluster centers ({} bits = {} bytes each)...",
        cli.clusters, cli.dimensions, bytes_per_vector
    );
    let mut centers = vec![0u8; cli.clusters * bytes_per_vector];
    rng.fill(&mut centers[..]);

    // Generate base vectors
    eprintln!(
        "Generating {} base vectors (noise={})...",
        cli.base_count, cli.noise
    );
    let base = generate_clustered_binary(
        &mut rng,
        cli.base_count,
        bytes_per_vector,
        &centers,
        cli.clusters,
        cli.noise,
    );

    // Generate query vectors
    eprintln!("Generating {} query vectors...", cli.query_count);
    let queries = generate_clustered_binary(
        &mut rng,
        cli.query_count,
        bytes_per_vector,
        &centers,
        cli.clusters,
        cli.noise,
    );

    // Compute ground truth
    eprintln!(
        "Computing brute-force hamming top-{} ground truth (NumKong)...",
        cli.neighbors
    );
    let ground_truth = compute_hamming_ground_truth(
        &base,
        &queries,
        cli.base_count,
        cli.query_count,
        bytes_per_vector,
        cli.neighbors,
        cli.ground_truth_batch,
    );

    // Write files
    let base_path = cli.output.join(format!("base.{}.b1bin", cli.base_count));
    let query_path = cli.output.join(format!("query.{}.b1bin", cli.query_count));
    let gt_path = cli
        .output
        .join(format!("groundtruth.{}.ibin", cli.query_count));

    eprintln!("Writing {}", base_path.display());
    let mut file = std::fs::File::create(&base_path).expect("create base file");
    write_bin_header(&mut file, cli.base_count as u32, cli.dimensions as u32).unwrap();
    file.write_all(&base).unwrap();

    eprintln!("Writing {}", query_path.display());
    let mut file = std::fs::File::create(&query_path).expect("create query file");
    write_bin_header(&mut file, cli.query_count as u32, cli.dimensions as u32).unwrap();
    file.write_all(&queries).unwrap();

    eprintln!("Writing {}", gt_path.display());
    let mut file = std::fs::File::create(&gt_path).expect("create groundtruth file");
    write_bin_header(&mut file, cli.query_count as u32, cli.neighbors as u32).unwrap();
    let gt_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(ground_truth.as_ptr() as *const u8, ground_truth.len() * 4)
    };
    file.write_all(gt_bytes).unwrap();

    let base_mb = (8 + cli.base_count * bytes_per_vector) as f64 / 1e6;
    eprintln!(
        "Done! base: {base_mb:.1} MB, {} output files",
        cli.output.display()
    );
}
