//! Download a subset of the USearchMolecules dataset, extract one binary
//! fingerprint column into the standard `.b1bin` format, and precompute
//! Hamming top-K ground truth with NumKong + ForkUnion.
//!
//! ## Build & Install
//!
//! ```sh
//! cargo install --path . --features download
//! ```
//!
//! ## Examples
//!
//! 10M PubChem molecules with MACCS (166-bit) fingerprints:
//! ```sh
//! retri-download-molecules \
//!     --source pubchem \
//!     --fingerprint maccs \
//!     --limit 10000000 \
//!     --query-count 10000 \
//!     --top-k 100 \
//!     --output datasets/pubchem_10M_maccs/
//! ```
//!
//! 100M GDB-13 molecules with ECFP4 (2048-bit) fingerprints, auto-tuned batch:
//! ```sh
//! retri-download-molecules \
//!     --source gdb13 \
//!     --fingerprint ecfp4 \
//!     --limit 100000000 \
//!     --query-count 10000 \
//!     --top-k 100 \
//!     --output datasets/gdb13_100M_ecfp4/
//! ```

use std::{
    fmt,
    fs::File,
    io::{BufWriter, Seek, SeekFrom, Write},
    path::PathBuf,
    time::Duration,
};

use arrow_array::{Array, FixedSizeBinaryArray};
use bytes::Bytes;
use clap::Parser;
use futures::stream::{self, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use parquet::arrow::{arrow_reader::ParquetRecordBatchReaderBuilder, ProjectionMask};
use rand::{rngs::StdRng, seq::index::sample as sample_without_replacement, SeedableRng};

use retrieval::{
    generate::{binary_view, compute_hamming_top_k, matrix_span},
    Seed, Threads,
};

const SHARD_ROWS: usize = 1_000_000;
const BUCKET_URL_PREFIX: &str = "https://s3.us-west-2.amazonaws.com/usearch-molecules/data";

/// Output-file BufWriter capacity. 1 MB amortizes the per-row
/// `FixedSizeBinaryArray::value(i)` → `write_all` calls over syscall boundaries.
const OUTPUT_BUFFER_BYTES: usize = 1 << 20;

/// Time limit for each HTTP request, shard downloads included.
const HTTP_REQUEST_TIME_LIMIT: Duration = Duration::from_secs(600);

#[derive(Parser, Debug)]
#[command(
    name = "retri-download-molecules",
    about = "Download USearchMolecules fingerprints and prepare a benchmark dataset"
)]
struct Cli {
    /// Dataset source: `pubchem` (115M), `gdb13` (977M), or `enamine` (6.04B).
    #[arg(long, value_enum)]
    source: Source,

    /// Fingerprint column: `maccs` (166 bits), `pubchem` (881 bits),
    /// `ecfp4` (2048 bits), or `fcfp4` (2048 bits).
    #[arg(long, value_enum)]
    fingerprint: Fingerprint,

    /// Maximum molecules to extract (default: all available shards).
    #[arg(long, value_parser = retrieval::parse_count_flag)]
    limit: Option<usize>,

    /// Number of query vectors to randomly sample from the base set.
    #[arg(long, default_value_t = 10_000, value_parser = retrieval::parse_count_flag)]
    query_count: usize,

    /// Top-K neighbors to record per query in the ground truth file.
    #[arg(long, default_value_t = 10, value_parser = retrieval::parse_count_flag)]
    top_k: usize,

    /// Ground-truth query batch size. Auto-tuned from free RAM when omitted.
    #[arg(long, value_parser = retrieval::parse_count_flag)]
    ground_truth_batch: Option<usize>,

    /// Threads for the ground-truth pass, 0 for all cores.
    #[arg(long, default_value = "0", value_parser = retrieval::parse_threads_flag)]
    threads: Threads,

    /// Concurrent shard downloads.
    #[arg(long, default_value_t = 4, value_parser = retrieval::parse_count_flag)]
    download_concurrency: usize,

    /// Random seed for query sampling, or `random` to draw one.
    #[arg(long, default_value = "42", value_parser = retrieval::parse_seed_flag)]
    seed: Seed,

    /// Keep downloaded Parquet shards in `<output>/parquet/` after conversion.
    #[arg(long)]
    keep_parquet: bool,

    /// Override the S3/HTTPS URL prefix (defaults to the public bucket).
    #[arg(long, default_value_t = String::from(BUCKET_URL_PREFIX), value_parser = clap::builder::NonEmptyStringValueParser::new())]
    url_prefix: String,

    /// Output directory (created if missing).
    #[arg(long)]
    output: PathBuf,
}

/// Static description of one fingerprint flavour.
struct FingerprintInfo {
    column_name: &'static str,
    bytes_per_vector: usize,
    dimensions_bits: usize,
}

/// Fingerprint columns, as `--fingerprint` spells them.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum Fingerprint {
    Maccs,
    Pubchem,
    Ecfp4,
    Fcfp4,
}

impl Fingerprint {
    fn info(self) -> FingerprintInfo {
        // `dimensions_bits` is the STORAGE bit count (bytes_per_vector * 8).
        // The logical fingerprint width is smaller for MACCS (166) and PubChem (881),
        // but the trailing padding bits are always zero and contribute nothing to
        // Hamming / Jaccard distance. Storing the padded count keeps downstream
        // tooling (FAISS IndexBinary, NumKong u1x8 tensors) happy — both require
        // a multiple of 8.
        match self {
            Self::Maccs => FingerprintInfo {
                column_name: "maccs",
                bytes_per_vector: 21,
                dimensions_bits: 168, // 166 logical + 2 pad
            },
            Self::Pubchem => FingerprintInfo {
                column_name: "pubchem",
                bytes_per_vector: 111,
                dimensions_bits: 888, // 881 logical + 7 pad
            },
            Self::Ecfp4 => FingerprintInfo {
                column_name: "ecfp4",
                bytes_per_vector: 256,
                dimensions_bits: 2048,
            },
            Self::Fcfp4 => FingerprintInfo {
                column_name: "fcfp4",
                bytes_per_vector: 256,
                dimensions_bits: 2048,
            },
        }
    }
}

impl fmt::Display for Fingerprint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        retrieval::spell_value(self, formatter)
    }
}

/// USearchMolecules sources, as `--source` spells them.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum Source {
    Pubchem,
    Gdb13,
    Enamine,
}

impl Source {
    /// The source's directory in the bucket.
    fn directory(self) -> &'static str {
        match self {
            Self::Pubchem => "pubchem",
            Self::Gdb13 => "gdb13",
            Self::Enamine => "real",
        }
    }
}

impl fmt::Display for Source {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        retrieval::spell_value(self, formatter)
    }
}

fn shard_url(url_prefix: &str, source: &str, shard_index: usize) -> String {
    // Shard filenames encode the [start, end) row range with 10-digit padding
    // under a `parquet/` subdirectory, e.g. `parquet/0000000000-0001000000.parquet`
    // for molecules [0, 1_000_000).
    let start = shard_index * SHARD_ROWS;
    let end = start + SHARD_ROWS;
    format!("{url_prefix}/{source}/parquet/{start:010}-{end:010}.parquet")
}

fn write_bin_header(file: &mut File, rows: u32, dimensions_bits: u32) -> std::io::Result<()> {
    file.write_all(&rows.to_le_bytes())?;
    file.write_all(&dimensions_bits.to_le_bytes())
}

fn write_bin_header_buf<W: Write>(writer: &mut W, rows: u32, dimensions_bits: u32) -> std::io::Result<()> {
    writer.write_all(&rows.to_le_bytes())?;
    writer.write_all(&dimensions_bits.to_le_bytes())
}

async fn download_shard(client: &reqwest::Client, url: &str) -> Result<Option<Bytes>, Box<dyn std::error::Error>> {
    let response = client.get(url).send().await?;
    let status = response.status();
    if status == reqwest::StatusCode::NOT_FOUND || status == reqwest::StatusCode::FORBIDDEN {
        // S3 returns 403 (not 404) for missing keys with this bucket's ACL.
        return Ok(None);
    }
    if !status.is_success() {
        return Err(format!("GET {url} -> {status}").into());
    }
    Ok(Some(response.bytes().await?))
}

/// Read one shard's Parquet bytes, extract the requested fingerprint column,
/// and append at most `take` rows to `out`. Returns the number of rows appended.
fn extract_and_append<W: Write>(
    shard_bytes: Bytes,
    info: &FingerprintInfo,
    take: usize,
    out: &mut W,
) -> Result<usize, Box<dyn std::error::Error>> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(shard_bytes)?;
    let column_index = builder
        .parquet_schema()
        .root_schema()
        .get_fields()
        .iter()
        .position(|f| f.name() == info.column_name)
        .ok_or_else(|| {
            format!(
                "Parquet schema has no column `{}` — is this the right source?",
                info.column_name
            )
        })?;
    let mask = ProjectionMask::roots(builder.parquet_schema(), [column_index]);
    let reader = builder.with_projection(mask).build()?;

    let mut appended = 0usize;
    for batch in reader {
        let batch = batch?;
        let column = batch.column(0);
        let fsb = column
            .as_any()
            .downcast_ref::<FixedSizeBinaryArray>()
            .ok_or("expected FixedSizeBinaryArray for fingerprint column")?;
        let width = fsb.value_length() as usize;
        if width != info.bytes_per_vector {
            return Err(format!(
                "fingerprint `{}` has width {width} in Parquet but expected {}",
                info.column_name, info.bytes_per_vector
            )
            .into());
        }
        let rows = fsb.len();
        let remaining = take.saturating_sub(appended);
        if remaining == 0 {
            break;
        }
        let rows_to_take = rows.min(remaining);
        // `.value(i)` correctly applies the array's slice offset; FixedSizeBinaryArray
        // rows are contiguous so BufWriter makes the per-row calls cheap.
        for i in 0..rows_to_take {
            out.write_all(fsb.value(i))?;
        }
        appended += rows_to_take;
        if appended >= take {
            break;
        }
    }
    Ok(appended)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli: Cli = retrieval::parse_cli();
    eprintln!("- Source: {}", cli.source);
    eprintln!("- Fingerprint: {}", cli.fingerprint);
    match cli.limit {
        Some(limit) => eprintln!("- Limit: {limit}"),
        None => eprintln!("- Limit: all"),
    }
    eprintln!("- Query count: {}", cli.query_count);
    eprintln!("- Top k: {}", cli.top_k);
    match cli.ground_truth_batch {
        Some(batch) => eprintln!("- Ground-truth batch: {batch}"),
        None => eprintln!("- Ground-truth batch: auto"),
    }
    eprintln!("- Threads: {}", cli.threads);
    eprintln!("- Download concurrency: {}", cli.download_concurrency);
    eprintln!("- Seed: {}", cli.seed);
    eprintln!("- Keep Parquet: {}", cli.keep_parquet);
    eprintln!("- URL prefix: {}", cli.url_prefix);
    eprintln!("- Output: {}", cli.output.display());
    let source = cli.source.directory();
    let info = cli.fingerprint.info();

    std::fs::create_dir_all(&cli.output)?;
    let parquet_dir = cli.output.join("parquet");
    if cli.keep_parquet {
        std::fs::create_dir_all(&parquet_dir)?;
    }

    let client = reqwest::Client::builder().timeout(HTTP_REQUEST_TIME_LIMIT).build()?;

    // Shard planning: request enough shards to cover `--limit`; if omitted,
    // walk shards until we hit a 403/404. We still need an upper bound to
    // build the shard-index stream — 16K > any USearchMolecules source (enamine
    // has 6040 shards), so missing-shard detection always terminates first.
    const SHARD_CEILING: usize = 16_384;
    let limit = cli.limit.unwrap_or(usize::MAX);
    let max_shards = if cli.limit.is_some() {
        limit.div_ceil(SHARD_ROWS).min(SHARD_CEILING)
    } else {
        SHARD_CEILING
    };

    // Stream shards to a temporary base file first without the header; we'll
    // seek back and patch the row count in once the total is known. BufWriter
    // keeps per-row writes cheap when copying from FixedSizeBinaryArray.
    let base_path_tmp = cli.output.join("base.tmp.b1bin");
    let base_file_handle = File::create(&base_path_tmp)?;
    let mut base_file = BufWriter::with_capacity(OUTPUT_BUFFER_BYTES, base_file_handle);
    // Placeholder header (8 bytes of zero). We'll rewrite it at the end.
    write_bin_header_buf(&mut base_file, 0, info.dimensions_bits as u32)?;

    let progress = ProgressBar::new(max_shards as u64);
    progress.set_style(
        ProgressStyle::with_template("  {spinner} downloading shards: {pos}/{len} [{elapsed_precise}] {msg}").unwrap(),
    );

    // Concurrent download with ordered processing: we need rows in shard order
    // for deterministic output. `buffered(n)` preserves request order.
    let urls = (0..max_shards).map(|shard_index| (shard_index, shard_url(&cli.url_prefix, source, shard_index)));

    let mut total_rows: usize = 0;
    let mut stream = stream::iter(urls)
        .map(|(shard_index, url)| {
            let client = client.clone();
            async move {
                let bytes = download_shard(&client, &url).await;
                (shard_index, url, bytes)
            }
        })
        .buffered(cli.download_concurrency);

    while let Some((shard_index, url, result)) = stream.next().await {
        let bytes = match result {
            Ok(Some(bytes)) => bytes,
            Ok(None) => {
                eprintln!("\n  shard {shard_index} missing (stopping): {url}");
                break;
            }
            Err(error) => {
                return Err(format!("download failed for {url}: {error}").into());
            }
        };

        if cli.keep_parquet {
            let shard_path = parquet_dir.join(format!("shard_{shard_index:03}.parquet"));
            std::fs::write(&shard_path, &bytes)?;
        }

        let remaining = limit.saturating_sub(total_rows);
        if remaining == 0 {
            break;
        }
        let appended = extract_and_append(bytes, &info, remaining, &mut base_file)?;
        total_rows += appended;
        progress.set_position((shard_index + 1) as u64);
        progress.set_message(format!(
            "{total_rows} rows ({:.1} GB)",
            (total_rows * info.bytes_per_vector) as f64 / (1u64 << 30) as f64
        ));
        if total_rows >= limit {
            break;
        }
    }
    progress.finish_and_clear();

    if total_rows == 0 {
        return Err("no rows downloaded — check --source / --url-prefix".into());
    }
    eprintln!(
        "Downloaded {total_rows} molecules ({:.2} GB base data)",
        (total_rows * info.bytes_per_vector) as f64 / (1u64 << 30) as f64
    );

    // Flush BufWriter, patch the header in place.
    base_file.flush()?;
    let mut base_file_handle = base_file.into_inner().map_err(|e| format!("flush base writer: {e}"))?;
    base_file_handle.seek(SeekFrom::Start(0))?;
    write_bin_header(&mut base_file_handle, total_rows as u32, info.dimensions_bits as u32)?;
    base_file_handle.sync_all()?;
    drop(base_file_handle);

    let base_path = cli.output.join(format!("base.{total_rows}.b1bin"));
    std::fs::rename(&base_path_tmp, &base_path)?;

    // Memory-map the base for query sampling + ground truth.
    let base_dataset = retrieval::Dataset::load(&base_path)?;
    assert_eq!(base_dataset.rows(), total_rows);
    let base_bytes = base_dataset.slice(0, total_rows, base_dataset.dimensions(), &mut []);
    let base_slice: &[u8] = match base_bytes.data {
        retrieval::VectorSlice::B1x8(data) => data,
        _ => return Err("unexpected non-binary base after load".into()),
    };

    // Sample query indices without replacement.
    let query_count = cli.query_count.min(total_rows);
    let mut rng = StdRng::seed_from_u64(cli.seed.into());
    let query_indices = sample_without_replacement(&mut rng, total_rows, query_count).into_vec();
    let mut sorted_indices = query_indices.clone();
    sorted_indices.sort_unstable();

    // Copy queries into a contiguous buffer that doubles as the ground-truth input.
    let mut query_buffer = {
        let mut values = Vec::new_in(std::alloc::System);
        values.resize(query_count * info.bytes_per_vector, 0u8);
        values
    };
    for (output_index, &base_index) in query_indices.iter().enumerate() {
        let source_offset = base_index * info.bytes_per_vector;
        let dest_offset = output_index * info.bytes_per_vector;
        query_buffer[dest_offset..dest_offset + info.bytes_per_vector]
            .copy_from_slice(&base_slice[source_offset..source_offset + info.bytes_per_vector]);
    }

    let query_path = cli.output.join(format!("query.{query_count}.b1bin"));
    let mut query_file = File::create(&query_path)?;
    write_bin_header(&mut query_file, query_count as u32, info.dimensions_bits as u32)?;
    query_file.write_all(&query_buffer)?;
    eprintln!("Wrote {}", query_path.display());

    // Ground truth.
    eprintln!(
        "Computing brute-force hamming top-{} ground truth (NumKong + ForkUnion)...",
        cli.top_k
    );
    // NumKong's u1x8 storage requires bit dimensions to be multiples of 8.
    // MACCS (166) and PubChem (881) fingerprints round up to full-byte counts;
    // trailing unused bits contribute 0 to Hamming (identical across vectors).
    let storage_bits = info.bytes_per_vector * 8;
    let base_view = binary_view(base_slice, total_rows, storage_bits);
    let query_view = binary_view(&query_buffer, query_count, storage_bits);
    let mut ground_truth_indices = {
        let mut values = Vec::new_in(std::alloc::System);
        values.resize(query_count * cli.top_k, 0u32);
        values
    };
    let ground_truth_span = matrix_span(&mut ground_truth_indices, query_count, cli.top_k);
    compute_hamming_top_k(
        base_view,
        query_view,
        ground_truth_span,
        cli.ground_truth_batch,
        cli.threads,
    )?;

    let gt_path = cli.output.join(format!("groundtruth.{query_count}.ibin"));
    let mut gt_file = File::create(&gt_path)?;
    write_bin_header(&mut gt_file, query_count as u32, cli.top_k as u32)?;
    // SAFETY: `u32` is POD.
    gt_file.write_all(unsafe { retrieval::pod_slice_as_bytes(&ground_truth_indices) })?;
    eprintln!("Wrote {}", gt_path.display());

    if !cli.keep_parquet && parquet_dir.exists() {
        let _ = std::fs::remove_dir_all(&parquet_dir);
    }

    eprintln!("Done: {} base x {} queries, top-{}", total_rows, query_count, cli.top_k);
    Ok(())
}
