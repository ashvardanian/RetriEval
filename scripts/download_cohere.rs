//! Download a subset of the Cohere Wikipedia multilingual embeddings dataset,
//! extract the 1024-bit binary embeddings into `.b1bin`, optionally keep the
//! aligned text metadata (title / paragraph / url), and precompute Hamming
//! top-K ground truth with NumKong + ForkUnion.
//!
//! Source: `Cohere/wikipedia-2023-11-embed-multilingual-v3-int8-binary` on
//! Hugging Face. The `emb_ubinary` column is a `list<uint8>` of length 128
//! (= 1024 bits packed per row).
//!
//! ## Build & Install
//!
//! ```sh
//! cargo install --path . --features download
//! ```
//!
//! ## Example
//!
//! ```sh
//! retri-download-cohere \
//!     --language en \
//!     --limit 10000000 \
//!     --query-count 10000 \
//!     --neighbors 100 \
//!     --output datasets/cohere_en_10M/
//! ```

use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::time::Duration;

use arrow_array::{Array, FixedSizeListArray, LargeStringArray, ListArray, RecordBatchReader, StringArray, UInt8Array};
use bytes::Bytes;
use clap::Parser;
use futures::stream::{self, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ProjectionMask;
use rand::rngs::StdRng;
use rand::seq::index::sample as sample_without_replacement;
use rand::SeedableRng;
use retrieval::generate as ground_truth;
use serde_json::Value;

/// 1024-bit (128-byte) packed binary embedding column.
const BYTES_PER_VECTOR: usize = 128;
const DIMENSIONS_BITS: usize = 1024;

/// Output-file BufWriter capacity, 1 MiB. Matches the molecules binary.
const OUTPUT_BUFFER_BYTES: usize = 1 << 20;

const HF_API_TREE: &str =
    "https://huggingface.co/api/datasets/CohereLabs/wikipedia-2023-11-embed-multilingual-v3-int8-binary/tree/main";
const HF_RESOLVE: &str =
    "https://huggingface.co/datasets/CohereLabs/wikipedia-2023-11-embed-multilingual-v3-int8-binary/resolve/main";

#[derive(Parser, Debug)]
#[command(
    name = "retri-download-cohere",
    about = "Download Cohere Wikipedia binary embeddings and prepare a benchmark dataset"
)]
struct Cli {
    /// Language config under the HF dataset (e.g. `en`, `de`, `fr`, ...).
    #[arg(long, default_value = "en")]
    language: String,

    /// Maximum rows to extract (default: all rows in that language).
    #[arg(long)]
    limit: Option<usize>,

    /// Number of query vectors to randomly sample from the base set.
    #[arg(long, default_value_t = 10_000)]
    query_count: usize,

    /// Top-K neighbors to record per query in the ground truth file.
    #[arg(long, default_value_t = 10)]
    neighbors: usize,

    /// Ground-truth query batch size. Auto-tuned from free RAM when omitted.
    #[arg(long)]
    batch_size: Option<usize>,

    /// Threads for the ground-truth pass (default: all logical cores).
    #[arg(long, default_value_t = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1))]
    threads: usize,

    /// Concurrent shard downloads.
    #[arg(long, default_value_t = 4)]
    download_concurrency: usize,

    /// Also extract `title` / `text` / `url` columns into aligned newline-
    /// delimited files. Newlines inside the text column are escaped as `\\n`
    /// to preserve 1:1 row ↔ line correspondence with `base.b1bin`.
    #[arg(long, default_value_t = true)]
    with_text: bool,

    /// Random seed for query sampling.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Keep downloaded Parquet shards under `<output>/parquet/`.
    #[arg(long)]
    keep_parquet: bool,

    /// Output directory (created if missing).
    #[arg(long)]
    output: PathBuf,
}

fn write_bin_header<W: Write>(writer: &mut W, rows: u32, dimensions_bits: u32) -> std::io::Result<()> {
    writer.write_all(&rows.to_le_bytes())?;
    writer.write_all(&dimensions_bits.to_le_bytes())
}

/// Escape newlines + carriage returns so one row ↔ one output line.
fn escape_line(source: &str, out: &mut String) {
    out.clear();
    for character in source.chars() {
        match character {
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\\' => out.push_str("\\\\"),
            other => out.push(other),
        }
    }
}

/// List the shard URLs for `<language>/` via the HF tree API (sorted by filename).
async fn list_shard_urls(client: &reqwest::Client, language: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let url = format!("{HF_API_TREE}/{language}");
    let response = client.get(&url).send().await?;
    if !response.status().is_success() {
        return Err(format!("GET {url} -> {}", response.status()).into());
    }
    let entries: Vec<Value> = response.json().await?;
    let mut shard_paths: Vec<String> = entries
        .into_iter()
        .filter_map(|entry| {
            let path = entry.get("path")?.as_str()?.to_string();
            let entry_type = entry.get("type")?.as_str()?;
            if entry_type == "file" && path.ends_with(".parquet") {
                Some(path)
            } else {
                None
            }
        })
        .collect();
    shard_paths.sort();
    Ok(shard_paths
        .into_iter()
        .map(|path| format!("{HF_RESOLVE}/{path}"))
        .collect())
}

async fn download_shard(client: &reqwest::Client, url: &str) -> Result<Bytes, Box<dyn std::error::Error>> {
    let response = client.get(url).send().await?;
    if !response.status().is_success() {
        return Err(format!("GET {url} -> {}", response.status()).into());
    }
    Ok(response.bytes().await?)
}

/// Extract `emb_ubinary` bytes (+ optional text columns) from one Parquet shard
/// and append at most `take` rows.
#[allow(clippy::too_many_arguments)]
fn extract_and_append<B: Write>(
    shard_bytes: Bytes,
    take: usize,
    base_out: &mut B,
    text_out: Option<&mut TextWriters>,
) -> Result<usize, Box<dyn std::error::Error>> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(shard_bytes)?;
    let parquet_schema = builder.parquet_schema();

    let mut columns_wanted: Vec<&str> = vec!["emb_ubinary"];
    if text_out.is_some() {
        columns_wanted.extend_from_slice(&["title", "text", "url"]);
    }
    let mask = ProjectionMask::columns(parquet_schema, columns_wanted.iter().copied());
    let reader = builder.with_projection(mask).build()?;

    // Column indices within the projected RecordBatch depend on their order in
    // the Parquet file; re-resolve by name from the reader's schema.
    let arrow_schema = reader.schema();
    let emb_idx = arrow_schema
        .index_of("emb_ubinary")
        .map_err(|_| "Parquet shard missing emb_ubinary column")?;
    let (title_idx, text_idx, url_idx): (Option<usize>, Option<usize>, Option<usize>) = if text_out.is_some() {
        (
            Some(arrow_schema.index_of("title").map_err(|_| "missing title")?),
            Some(arrow_schema.index_of("text").map_err(|_| "missing text")?),
            Some(arrow_schema.index_of("url").map_err(|_| "missing url")?),
        )
    } else {
        (None, None, None)
    };

    let mut text_out = text_out;
    let mut appended = 0usize;
    let mut escape_scratch = String::new();
    for batch in reader {
        let batch = batch?;
        let remaining = take.saturating_sub(appended);
        if remaining == 0 {
            break;
        }
        let rows = batch.num_rows();
        let rows_to_take = rows.min(remaining);

        let emb_column = batch.column(emb_idx);
        // `emb_ubinary` is declared as list<uint8>[128] in the HF schema, but
        // Arrow decoders may represent it as either `FixedSizeList` (fast path)
        // or `List` (variable-length) depending on the Parquet encoding.
        if let Some(fsl) = emb_column.as_any().downcast_ref::<FixedSizeListArray>() {
            let width = fsl.value_length() as usize;
            if width != BYTES_PER_VECTOR {
                return Err(format!("emb_ubinary has width {width}, expected {BYTES_PER_VECTOR}").into());
            }
            let values = fsl
                .values()
                .as_any()
                .downcast_ref::<UInt8Array>()
                .ok_or("emb_ubinary inner values must be UInt8Array")?;
            for i in 0..rows_to_take {
                let offset = (fsl.offset() + i) * width;
                let row_bytes = &values.values()[offset..offset + width];
                base_out.write_all(row_bytes)?;
            }
        } else if let Some(list) = emb_column.as_any().downcast_ref::<ListArray>() {
            let values = list
                .values()
                .as_any()
                .downcast_ref::<UInt8Array>()
                .ok_or("emb_ubinary inner values must be UInt8Array")?;
            let raw_values = values.values();
            for i in 0..rows_to_take {
                let start = list.value_offsets()[i] as usize;
                let end = list.value_offsets()[i + 1] as usize;
                let width = end - start;
                if width != BYTES_PER_VECTOR {
                    return Err(format!("emb_ubinary row {i} has width {width}, expected {BYTES_PER_VECTOR}").into());
                }
                base_out.write_all(&raw_values[start..end])?;
            }
        } else {
            return Err(format!("unexpected emb_ubinary type: {:?}", emb_column.data_type()).into());
        }

        if let Some(writers) = text_out.as_deref_mut() {
            let title_column = batch.column(title_idx.unwrap());
            let text_column = batch.column(text_idx.unwrap());
            let url_column = batch.column(url_idx.unwrap());
            write_text_column(title_column, rows_to_take, &mut writers.titles, &mut escape_scratch)?;
            write_text_column(text_column, rows_to_take, &mut writers.texts, &mut escape_scratch)?;
            write_text_column(url_column, rows_to_take, &mut writers.urls, &mut escape_scratch)?;
        }

        appended += rows_to_take;
        if appended >= take {
            break;
        }
    }
    Ok(appended)
}

fn write_text_column<W: Write>(
    column: &dyn Array,
    rows_to_take: usize,
    out: &mut W,
    scratch: &mut String,
) -> std::io::Result<()> {
    if let Some(strings) = column.as_any().downcast_ref::<StringArray>() {
        for i in 0..rows_to_take {
            escape_line(strings.value(i), scratch);
            out.write_all(scratch.as_bytes())?;
            out.write_all(b"\n")?;
        }
    } else if let Some(strings) = column.as_any().downcast_ref::<LargeStringArray>() {
        for i in 0..rows_to_take {
            escape_line(strings.value(i), scratch);
            out.write_all(scratch.as_bytes())?;
            out.write_all(b"\n")?;
        }
    } else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("text column has non-string type: {:?}", column.data_type()),
        ));
    }
    Ok(())
}

struct TextWriters {
    titles: BufWriter<File>,
    texts: BufWriter<File>,
    urls: BufWriter<File>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    std::fs::create_dir_all(&cli.output)?;
    let parquet_dir = cli.output.join("parquet");
    if cli.keep_parquet {
        std::fs::create_dir_all(&parquet_dir)?;
    }

    let client = reqwest::Client::builder().timeout(Duration::from_secs(600)).build()?;

    // Resolve shard URLs for the requested language via HF's tree API.
    eprintln!("Listing shards for language `{}`...", cli.language);
    let shard_urls = list_shard_urls(&client, &cli.language).await?;
    if shard_urls.is_empty() {
        return Err(format!("no shards found for language `{}`", cli.language).into());
    }
    eprintln!("Found {} Parquet shards.", shard_urls.len());

    let limit = cli.limit.unwrap_or(usize::MAX);

    // Open output writers.
    let base_path_tmp = cli.output.join("base.tmp.b1bin");
    let base_file_handle = File::create(&base_path_tmp)?;
    let mut base_file = BufWriter::with_capacity(OUTPUT_BUFFER_BYTES, base_file_handle);
    write_bin_header(&mut base_file, 0, DIMENSIONS_BITS as u32)?;

    let mut text_writers = if cli.with_text {
        Some(TextWriters {
            titles: BufWriter::new(File::create(cli.output.join("titles.txt"))?),
            texts: BufWriter::new(File::create(cli.output.join("texts.txt"))?),
            urls: BufWriter::new(File::create(cli.output.join("urls.txt"))?),
        })
    } else {
        None
    };

    let progress = ProgressBar::new(shard_urls.len() as u64);
    progress.set_style(
        ProgressStyle::with_template("  {spinner} downloading shards: {pos}/{len} [{elapsed_precise}] {msg}").unwrap(),
    );

    let shard_indices_and_urls: Vec<(usize, String)> = shard_urls.into_iter().enumerate().collect();

    let mut stream = stream::iter(shard_indices_and_urls)
        .map(|(shard_index, url)| {
            let client = client.clone();
            async move {
                let bytes = download_shard(&client, &url).await;
                (shard_index, url, bytes)
            }
        })
        .buffered(cli.download_concurrency);

    let mut total_rows: usize = 0;
    while let Some((shard_index, url, result)) = stream.next().await {
        let bytes = match result {
            Ok(bytes) => bytes,
            Err(error) => {
                return Err(format!("download failed for {url}: {error}").into());
            }
        };

        if cli.keep_parquet {
            let shard_path = parquet_dir.join(format!("shard_{shard_index:05}.parquet"));
            std::fs::write(&shard_path, &bytes)?;
        }

        let remaining = limit.saturating_sub(total_rows);
        if remaining == 0 {
            break;
        }
        let appended = extract_and_append(bytes, remaining, &mut base_file, text_writers.as_mut())?;
        total_rows += appended;
        progress.set_position((shard_index + 1) as u64);
        progress.set_message(format!(
            "{total_rows} rows ({:.1} GB)",
            (total_rows * BYTES_PER_VECTOR) as f64 / 1e9
        ));
        if total_rows >= limit {
            break;
        }
    }
    progress.finish_and_clear();

    if total_rows == 0 {
        return Err("no rows downloaded".into());
    }
    eprintln!(
        "Downloaded {total_rows} rows ({:.2} GB base data)",
        (total_rows * BYTES_PER_VECTOR) as f64 / 1e9
    );

    // Flush + patch base header.
    base_file.flush()?;
    let mut base_file_handle = base_file.into_inner().map_err(|e| format!("flush base writer: {e}"))?;
    base_file_handle.seek(SeekFrom::Start(0))?;
    {
        let rows = total_rows as u32;
        let dims = DIMENSIONS_BITS as u32;
        base_file_handle.write_all(&rows.to_le_bytes())?;
        base_file_handle.write_all(&dims.to_le_bytes())?;
    }
    base_file_handle.sync_all()?;
    drop(base_file_handle);

    let base_path = cli.output.join(format!("base.{total_rows}.b1bin"));
    std::fs::rename(&base_path_tmp, &base_path)?;

    if let Some(writers) = text_writers {
        let TextWriters {
            mut titles,
            mut texts,
            mut urls,
        } = writers;
        titles.flush()?;
        texts.flush()?;
        urls.flush()?;
    }

    // Sample queries and compute ground truth via mmapped base.
    let base_dataset = retrieval::Dataset::load(&base_path)?;
    assert_eq!(base_dataset.rows(), total_rows);
    let base_bytes = base_dataset.all();
    let base_slice: &[u8] = match base_bytes.data {
        retrieval::VectorSlice::B1x8(data) => data,
        _ => return Err("unexpected non-binary base after load".into()),
    };

    let query_count = cli.query_count.min(total_rows);
    let mut rng = StdRng::seed_from_u64(cli.seed);
    let query_indices = sample_without_replacement(&mut rng, total_rows, query_count).into_vec();

    let mut query_buffer = vec![0u8; query_count * BYTES_PER_VECTOR];
    for (output_index, &base_index) in query_indices.iter().enumerate() {
        let source_offset = base_index * BYTES_PER_VECTOR;
        let dest_offset = output_index * BYTES_PER_VECTOR;
        query_buffer[dest_offset..dest_offset + BYTES_PER_VECTOR]
            .copy_from_slice(&base_slice[source_offset..source_offset + BYTES_PER_VECTOR]);
    }

    let query_path = cli.output.join(format!("query.{query_count}.b1bin"));
    let mut query_file = File::create(&query_path)?;
    write_bin_header(&mut query_file, query_count as u32, DIMENSIONS_BITS as u32)?;
    query_file.write_all(&query_buffer)?;
    eprintln!("Wrote {}", query_path.display());

    eprintln!(
        "Computing brute-force hamming top-{} ground truth (NumKong + ForkUnion)...",
        cli.neighbors
    );
    let base_view = ground_truth::binary_view(base_slice, total_rows, DIMENSIONS_BITS);
    let query_view = ground_truth::binary_view(&query_buffer, query_count, DIMENSIONS_BITS);
    let mut ground_truth_indices = vec![0u32; query_count * cli.neighbors];
    let ground_truth_span = ground_truth::matrix_span(&mut ground_truth_indices, query_count, cli.neighbors);
    ground_truth::compute_hamming_top_k(base_view, query_view, ground_truth_span, cli.batch_size, cli.threads)?;

    let gt_path = cli.output.join(format!("groundtruth.{query_count}.ibin"));
    let mut gt_file = File::create(&gt_path)?;
    write_bin_header(&mut gt_file, query_count as u32, cli.neighbors as u32)?;
    // SAFETY: `u32` is POD.
    gt_file.write_all(unsafe { retrieval::pod_slice_as_bytes(&ground_truth_indices) })?;
    eprintln!("Wrote {}", gt_path.display());

    if !cli.keep_parquet && parquet_dir.exists() {
        let _ = std::fs::remove_dir_all(&parquet_dir);
    }

    eprintln!("Done: {total_rows} base × {query_count} queries, top-{}", cli.neighbors);
    Ok(())
}
