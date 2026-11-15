//! Exact top-K ground truth for an existing base and query set, with `retrieval::ground_truth`.
//!
//! ## Examples
//!
//! SpaceV's queries over its first 100M base rows, under squared Euclidean distance:
//! ```sh
//! retri-ground-truth \
//!     --base-vectors spacev-100m/base.100M.i8bin \
//!     --query-vectors spacev-100m/query.30K.i8bin \
//!     --metric euclidean \
//!     --top-k 100 \
//!     --output spacev-100m/groundtruth.30K.100M.ibin
//! ```

use std::path::PathBuf;

use clap::Parser;

use retrieval::{eval::Metric, ground_truth::write_ground_truth, Dataset, Threads, UnwrapOrBail};

#[derive(Parser, Debug)]
#[command(
    name = "retri-ground-truth",
    about = "Compute exact ground truth for existing vectors"
)]
struct Cli {
    /// Base vectors, a file or a glob of shards: .fbin, .u8bin, .i8bin or .b1bin
    #[arg(long)]
    base_vectors: PathBuf,

    /// Query vectors of the base's scalar type and width
    #[arg(long)]
    query_vectors: PathBuf,

    /// euclidean over .fbin, .i8bin and .u8bin, or hamming over .b1bin
    #[arg(long, value_parser = parse_metric)]
    metric: Metric,

    /// Neighbors per query in the ground-truth file
    #[arg(long, default_value_t = 100, value_parser = retrieval::parse_count_flag)]
    top_k: usize,

    /// Ground-truth threads, 0 for all cores
    #[arg(long, default_value = "0", value_parser = retrieval::parse_threads_flag)]
    threads: Threads,

    /// Ground-truth file to write: `[queries, top_k]` neighbor rows, then their distances
    #[arg(long)]
    output: PathBuf,
}

fn parse_metric(text: &str) -> Result<Metric, String> {
    Metric::from_label(text).ok_or_else(|| "expected euclidean or hamming".into())
}

fn main() {
    let cli: Cli = retrieval::parse_cli();
    let base = Dataset::load(&cli.base_vectors).unwrap_or_bail("base vectors");
    let queries = Dataset::load(&cli.query_vectors).unwrap_or_bail("query vectors");
    write_ground_truth(&base, &queries, cli.metric, cli.top_k, cli.threads, &cli.output).unwrap_or_bail("ground truth");
}
