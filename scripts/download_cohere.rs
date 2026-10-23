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
//!     --top-k 100 \
//!     --output datasets/cohere_en_10M/
//! ```

use std::{
    fs::File,
    io::{BufWriter, Seek, SeekFrom, Write},
    path::PathBuf,
};

use arrow_array::{
    cast::AsArray, Array, FixedSizeListArray, GenericStringArray, ListArray, OffsetSizeTrait, RecordBatchReader,
    UInt8Array,
};
use bytes::Bytes;
use clap::Parser;
use futures::stream::{self, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use parquet::arrow::{arrow_reader::ParquetRecordBatchReaderBuilder, ProjectionMask};
use serde_json::Value;

use retrieval::{
    download::{download_shard, http_client, OUTPUT_BUFFER_BYTES},
    ground_truth::{write_bin_header, write_sampled_ground_truth},
    Dataset, Seed, Threads,
};

/// 1024-bit (128-byte) packed binary embedding column.
const BYTES_PER_VECTOR: usize = 128;
const DIMENSIONS_BITS: usize = 1024;

const HF_API_TREE: &str =
    "https://huggingface.co/api/datasets/CohereLabs/wikipedia-2023-11-embed-multilingual-v3-int8-binary/tree/main";
const HF_RESOLVE: &str =
    "https://huggingface.co/datasets/CohereLabs/wikipedia-2023-11-embed-multilingual-v3-int8-binary/resolve/main";

/// The aligned text columns, each written one escaped line per row.
const TEXT_COLUMNS: [&str; 3] = ["title", "text", "url"];

#[derive(Parser, Debug)]
#[command(
    name = "retri-download-cohere",
    about = "Download Cohere Wikipedia binary embeddings and prepare a benchmark dataset"
)]
struct Cli {
    /// Language config under the HF dataset (e.g. `en`, `de`, `fr`, ...).
    #[arg(long, default_value = "en", value_parser = clap::builder::NonEmptyStringValueParser::new())]
    language: String,

    /// Maximum rows to extract (default: all rows in that language).
    #[arg(long, value_parser = retrieval::parse_count_flag)]
    limit: Option<usize>,

    /// Number of query vectors to randomly sample from the base set.
    #[arg(long, default_value_t = 10_000, value_parser = retrieval::parse_count_flag)]
    query_count: usize,

    /// Top-K neighbors to record per query in the ground truth file.
    #[arg(long, default_value_t = 10, value_parser = retrieval::parse_count_flag)]
    top_k: usize,

    /// Threads for the ground-truth pass, 0 for all cores.
    #[arg(long, default_value = "0", value_parser = retrieval::parse_threads_flag)]
    threads: Threads,

    /// Concurrent shard downloads.
    #[arg(long, default_value_t = 4, value_parser = retrieval::parse_count_flag)]
    download_concurrency: usize,

    /// Skip extracting the `title` / `text` / `url` columns into aligned newline-
    /// delimited files. Newlines inside the text column are escaped as `\\n`
    /// to preserve 1:1 row ↔ line correspondence with `base.b1bin`.
    #[arg(long)]
    no_text: bool,

    /// Random seed for query sampling, or `random` to draw one.
    #[arg(long, default_value = "42", value_parser = retrieval::parse_seed_flag)]
    seed: Seed,

    /// Keep downloaded Parquet shards under `<output>/parquet/`.
    #[arg(long)]
    keep_parquet: bool,

    /// Output directory (created if missing).
    #[arg(long)]
    output: PathBuf,
}

/// Escape newlines + carriage returns so one row ↔ one output line.
fn escape_line(source: &str, output: &mut String) {
    output.clear();
    for character in source.chars() {
        match character {
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\\' => output.push_str("\\\\"),
            other => output.push(other),
        }
    }
}

/// List the shard URLs for `<language>/` via the HF tree API (sorted by filename).
async fn list_shard_urls(
    client: &reqwest::Client,
    language: &str,
) -> Result<Vec<String, std::alloc::System>, Box<dyn std::error::Error>> {
    let url = format!("{HF_API_TREE}/{language}");
    let response = client.get(&url).send().await?;
    if !response.status().is_success() {
        return Err(format!("GET {url} -> {}", response.status()).into());
    }
    let entries: Vec<Value> = response.json().await?;
    let mut shard_paths = Vec::new_in(std::alloc::System);
    shard_paths.extend(entries.into_iter().filter_map(|entry| {
        let path = entry.get("path")?.as_str()?.to_string();
        let entry_type = entry.get("type")?.as_str()?;
        if entry_type == "file" && path.ends_with(".parquet") {
            Some(path)
        } else {
            None
        }
    }));
    shard_paths.sort();
    for path in &mut shard_paths {
        *path = format!("{HF_RESOLVE}/{path}");
    }
    Ok(shard_paths)
}

/// Extract `emb_ubinary` bytes, and the text columns when `text_outputs` asks, from one Parquet
/// shard, appending at most `take` rows.
fn extract_and_append<B: Write>(
    shard_bytes: Bytes,
    take: usize,
    base_output: &mut B,
    mut text_outputs: Option<&mut [BufWriter<File>; 3]>,
) -> Result<usize, Box<dyn std::error::Error>> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(shard_bytes)?;
    let columns_wanted: &[&str] = match text_outputs {
        Some(_) => &["emb_ubinary", "title", "text", "url"],
        None => &["emb_ubinary"],
    };
    let mask = ProjectionMask::columns(builder.parquet_schema(), columns_wanted.iter().copied());
    let reader = builder.with_projection(mask).build()?;

    // Projected column indices follow the file's order, so they are looked up by name.
    let schema = reader.schema();
    let embedding_column = schema
        .index_of("emb_ubinary")
        .map_err(|_| "Parquet shard missing emb_ubinary column")?;
    let text_column_indices = match text_outputs {
        Some(_) => Some(
            TEXT_COLUMNS
                .map(|name| schema.index_of(name))
                .into_iter()
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| "Parquet shard missing a text column")?,
        ),
        None => None,
    };

    let mut appended = 0;
    let mut escape_scratch = String::new();
    for batch in reader {
        let batch = batch?;
        let rows = batch.num_rows().min(take - appended);
        let embeddings = batch.column(embedding_column);
        // `emb_ubinary` is list<uint8>[128] in the schema, which Arrow decodes as either a
        // `FixedSizeList` or a variable-length `List` depending on the Parquet encoding.
        if let Some(fixed_list) = embeddings.as_any().downcast_ref::<FixedSizeListArray>() {
            let width = fixed_list.value_length() as usize;
            if width != BYTES_PER_VECTOR {
                return Err(format!("emb_ubinary has width {width}, expected {BYTES_PER_VECTOR}").into());
            }
            let values = fixed_list
                .values()
                .as_primitive_opt::<arrow_array::types::UInt8Type>()
                .ok_or("emb_ubinary inner values must be UInt8Array")?;
            let start = fixed_list.offset() * width;
            base_output.write_all(&values.values()[start..start + rows * width])?;
        } else if let Some(list) = embeddings.as_any().downcast_ref::<ListArray>() {
            let values = list
                .values()
                .as_any()
                .downcast_ref::<UInt8Array>()
                .ok_or("emb_ubinary inner values must be UInt8Array")?
                .values();
            for (row, bounds) in list.value_offsets().windows(2).take(rows).enumerate() {
                let (start, end) = (bounds[0] as usize, bounds[1] as usize);
                if end - start != BYTES_PER_VECTOR {
                    return Err(format!(
                        "emb_ubinary row {row} has width {}, expected {BYTES_PER_VECTOR}",
                        end - start
                    )
                    .into());
                }
                base_output.write_all(&values[start..end])?;
            }
        } else {
            return Err(format!("unexpected emb_ubinary type: {:?}", embeddings.data_type()).into());
        }

        if let (Some(outputs), Some(indices)) = (text_outputs.as_deref_mut(), &text_column_indices) {
            for (output, &column) in outputs.iter_mut().zip(indices) {
                write_text_column(batch.column(column).as_ref(), rows, output, &mut escape_scratch)?;
            }
        }

        appended += rows;
        if appended == take {
            break;
        }
    }
    Ok(appended)
}

fn write_text_column<W: Write>(
    column: &dyn Array,
    rows: usize,
    output: &mut W,
    scratch: &mut String,
) -> std::io::Result<()> {
    match (column.as_string_opt::<i32>(), column.as_string_opt::<i64>()) {
        (Some(strings), _) => write_strings(strings, rows, output, scratch),
        (None, Some(strings)) => write_strings(strings, rows, output, scratch),
        (None, None) => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("text column has non-string type: {:?}", column.data_type()),
        )),
    }
}

fn write_strings<Offset: OffsetSizeTrait, W: Write>(
    strings: &GenericStringArray<Offset>,
    rows: usize,
    output: &mut W,
    scratch: &mut String,
) -> std::io::Result<()> {
    for row in 0..rows {
        escape_line(strings.value(row), scratch);
        output.write_all(scratch.as_bytes())?;
        output.write_all(b"\n")?;
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli: Cli = retrieval::parse_cli();
    eprintln!("- Language: {}", cli.language);
    match cli.limit {
        Some(limit) => eprintln!("- Limit: {limit}"),
        None => eprintln!("- Limit: all"),
    }
    eprintln!("- Query count: {}", cli.query_count);
    eprintln!("- Top k: {}", cli.top_k);
    eprintln!("- Threads: {}", cli.threads);
    eprintln!("- Download concurrency: {}", cli.download_concurrency);
    eprintln!("- No text: {}", cli.no_text);
    eprintln!("- Seed: {}", cli.seed);
    eprintln!("- Keep Parquet: {}", cli.keep_parquet);
    eprintln!("- Output: {}", cli.output.display());
    std::fs::create_dir_all(&cli.output)?;
    let parquet_directory = cli.output.join("parquet");
    if cli.keep_parquet {
        std::fs::create_dir_all(&parquet_directory)?;
    }
    let client = http_client()?;

    eprintln!("Listing shards for language `{}`...", cli.language);
    let shard_urls = list_shard_urls(&client, &cli.language).await?;
    if shard_urls.is_empty() {
        return Err(format!("no shards found for language `{}`", cli.language).into());
    }
    eprintln!("Found {} Parquet shards.", shard_urls.len());
    let limit = cli.limit.unwrap_or(usize::MAX);

    let base_path_scratch = cli.output.join("base.tmp.b1bin");
    let mut base_writer = BufWriter::with_capacity(OUTPUT_BUFFER_BYTES, File::create(&base_path_scratch)?);
    write_bin_header(&mut base_writer, 0, DIMENSIONS_BITS)?;
    let mut text_writers = match cli.no_text {
        true => None,
        false => Some(
            TEXT_COLUMNS
                .map(|name| File::create(cli.output.join(format!("{name}s.txt"))).map(BufWriter::new))
                .into_iter()
                .collect::<std::io::Result<Vec<_>>>()?
                .try_into()
                .map_err(|_| "three text writers")?,
        ),
    };

    let progress = ProgressBar::new(shard_urls.len() as u64);
    progress.set_style(
        ProgressStyle::with_template("  {spinner} downloading shards: {pos}/{len} [{elapsed_precise}] {msg}").unwrap(),
    );
    let mut downloads = stream::iter(shard_urls.into_iter().enumerate())
        .map(|(shard_index, url)| {
            let client = client.clone();
            async move {
                let bytes = download_shard(&client, &url).await;
                (shard_index, url, bytes)
            }
        })
        .buffered(cli.download_concurrency);

    let mut total_rows = 0;
    while let Some((shard_index, url, result)) = downloads.next().await {
        let bytes = result?.ok_or_else(|| format!("{url} answered 403 or 404 on every attempt"))?;
        if cli.keep_parquet {
            std::fs::write(
                parquet_directory.join(format!("shard_{shard_index:05}.parquet")),
                &bytes,
            )?;
        }
        total_rows += extract_and_append(bytes, limit - total_rows, &mut base_writer, text_writers.as_mut())?;
        progress.set_position((shard_index + 1) as u64);
        progress.set_message(format!(
            "{total_rows} rows ({:.1} GB)",
            (total_rows * BYTES_PER_VECTOR) as f64 / (1u64 << 30) as f64
        ));
        if total_rows == limit {
            break;
        }
    }
    progress.finish_and_clear();
    if total_rows == 0 {
        return Err("no rows downloaded".into());
    }
    eprintln!(
        "Downloaded {total_rows} rows ({:.2} GB base data)",
        (total_rows * BYTES_PER_VECTOR) as f64 / (1u64 << 30) as f64
    );

    // The base opened with a zero row count, patched in now that every row landed.
    let mut base_file = base_writer
        .into_inner()
        .map_err(|error| format!("flush base writer: {error}"))?;
    base_file.seek(SeekFrom::Start(0))?;
    write_bin_header(&mut base_file, total_rows, DIMENSIONS_BITS)?;
    base_file.sync_all()?;
    drop(base_file);
    let base_path = cli.output.join(format!("base.{total_rows}.b1bin"));
    std::fs::rename(&base_path_scratch, &base_path)?;
    for writer in text_writers.iter_mut().flatten() {
        writer.flush()?;
    }

    write_sampled_ground_truth(
        &Dataset::load(&base_path)?,
        cli.query_count,
        cli.top_k,
        cli.seed,
        cli.threads,
        &cli.output,
    )?;
    eprintln!(
        "Done: {total_rows} base × {} queries, top-{}",
        cli.query_count.min(total_rows),
        cli.top_k
    );
    Ok(())
}
