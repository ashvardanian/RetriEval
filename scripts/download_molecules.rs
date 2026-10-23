//! Download a subset of the USearchMolecules dataset, extract one binary
//! fingerprint column into `.b1bin` base parts, and precompute
//! Hamming top-K ground truth with NumKong + ForkUnion; or extract the
//! first conformer of every embedded molecule into point-cloud parts.
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
//! 100M GDB-13 molecules with ECFP4 (2048-bit) fingerprints:
//! ```sh
//! retri-download-molecules \
//!     --source gdb13 \
//!     --fingerprint ecfp4 \
//!     --limit 100000000 \
//!     --query-count 10000 \
//!     --top-k 100 \
//!     --output datasets/gdb13_100M_ecfp4/
//! ```
//!
//! Every embedded PubChem molecule's first conformer, hydrogens included, in 8 parts
//! of `f16` xyz points beside their per-cloud counts:
//! ```sh
//! retri-download-molecules \
//!     --source pubchem \
//!     --conformers \
//!     --part-rows 15000000 \
//!     --output datasets/pubchem-conformers-113m/
//! ```

use std::{
    fmt,
    fs::File,
    io::{BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    time::Duration,
};

use arrow_array::{
    cast::AsArray,
    types::{Float16Type, UInt8Type},
    Array, FixedSizeBinaryArray,
};
use bytes::Bytes;
use clap::Parser;
use futures::stream::{self, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use parquet::arrow::{arrow_reader::ParquetRecordBatchReaderBuilder, ProjectionMask};
use rand::{rngs::StdRng, seq::index::sample as sample_without_replacement, SeedableRng};

use retrieval::{
    generate::{binary_view, compute_top_k, matrix_span},
    Seed, Threads,
};

const SHARD_ROWS: usize = 1_000_000;
const BUCKET_URL_PREFIX: &str = "https://s3.us-west-2.amazonaws.com/usearch-molecules/data";

/// Output-file BufWriter capacity. 1 MB amortizes the per-row
/// `FixedSizeBinaryArray::value(i)` → `write_all` calls over syscall boundaries.
const OUTPUT_BUFFER_BYTES: usize = 1 << 20;

/// Time limit for each HTTP request, shard downloads included.
const HTTP_REQUEST_TIME_LIMIT: Duration = Duration::from_secs(600);

/// Attempts per shard before a failure ends the run.
const SHARD_ATTEMPTS: u32 = 5;

/// Delay before the first retry, doubled for each one after it.
const SHARD_RETRY_DELAY: Duration = Duration::from_secs(2);

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
    #[arg(long, value_enum, required_unless_present = "conformers")]
    fingerprint: Option<Fingerprint>,

    /// Extract each embedded molecule's first conformer from the `.3D.parquet` shards instead of a
    /// fingerprint: `base.part_N.f16bin` xyz points and `counts.part_N.ibin` points per cloud.
    #[arg(long, conflicts_with_all = ["fingerprint", "top_k"])]
    conformers: bool,

    /// Maximum molecules to extract (default: all available shards).
    #[arg(long, value_parser = retrieval::parse_count_flag)]
    limit: Option<usize>,

    /// Rows per base part, `base.part_N.b1bin`, a whole number of the bucket's 1M-row shards.
    /// Keeps each part's `u32` row count in range; Enamine's 6.04B rows overflow a single file.
    #[arg(long, default_value_t = 512 * SHARD_ROWS, value_parser = retrieval::parse_count_flag)]
    part_rows: usize,

    /// Number of query vectors to randomly sample from the base set; needs `--top-k`.
    #[arg(long, default_value_t = 10_000, value_parser = retrieval::parse_count_flag)]
    query_count: usize,

    /// Top-K neighbors to record per query in the ground truth file.
    /// Without it, only the base parts are written, for self-recall benchmarks.
    #[arg(long, value_parser = retrieval::parse_count_flag)]
    top_k: Option<usize>,

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

fn shard_url(url_prefix: &str, source: &str, shard_index: usize, suffix: &str) -> String {
    // Shard filenames encode the [start, end) row range with 10-digit padding
    // under a `parquet/` subdirectory, e.g. `parquet/0000000000-0001000000.parquet`
    // for molecules [0, 1_000_000), and `.3D.parquet` for their conformers.
    let start = shard_index * SHARD_ROWS;
    let end = start + SHARD_ROWS;
    format!("{url_prefix}/{source}/parquet/{start:010}-{end:010}{suffix}")
}

fn write_bin_header(file: &mut File, rows: u32, dimensions_bits: u32) -> std::io::Result<()> {
    file.write_all(&rows.to_le_bytes())?;
    file.write_all(&dimensions_bits.to_le_bytes())
}

fn write_bin_header_buf<W: Write>(writer: &mut W, rows: u32, dimensions_bits: u32) -> std::io::Result<()> {
    writer.write_all(&rows.to_le_bytes())?;
    writer.write_all(&dimensions_bits.to_le_bytes())
}

fn part_path(output: &Path, part_index: usize) -> PathBuf {
    output.join(format!("base.part_{part_index:03}.b1bin"))
}

fn points_part_path(output: &Path, part_index: usize) -> PathBuf {
    output.join(format!("base.part_{part_index:03}.f16bin"))
}

/// A conformer part's point counts, written last, so a nonzero header marks the pair finished.
fn counts_part_path(output: &Path, part_index: usize) -> PathBuf {
    output.join(format!("counts.part_{part_index:03}.ibin"))
}

/// Rows a part's header counts: zero while the part is being written, `None` when it is missing.
fn part_rows_of(path: &Path) -> std::io::Result<Option<usize>> {
    let mut header = [0u8; 4];
    match File::open(path) {
        Ok(mut file) => file
            .read_exact(&mut header)
            .map(|_| Some(u32::from_le_bytes(header) as usize)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

/// Patches a written part's row count into its header and syncs it to disk.
fn finish_part(writer: BufWriter<File>, rows: usize, dimensions_bits: usize) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = writer
        .into_inner()
        .map_err(|error| format!("flush part writer: {error}"))?;
    file.seek(SeekFrom::Start(0))?;
    write_bin_header(&mut file, rows as u32, dimensions_bits as u32)?;
    file.sync_all()?;
    Ok(())
}

/// Shard download progress, with a length only when `--limit` bounds the shards.
fn shard_progress(shards: Option<usize>) -> ProgressBar {
    let progress = shards.map_or_else(ProgressBar::no_length, |shards| ProgressBar::new(shards as u64));
    progress.set_style(
        ProgressStyle::with_template("  {spinner} downloading shards: {pos}/{len} [{elapsed_precise}] {msg}").unwrap(),
    );
    progress
}

/// Fetches one shard, retrying failures with doubling delays.
/// Returns `None` only when every attempt answered 403 or 404, which this bucket's ACL uses past the last shard.
async fn download_shard(client: &reqwest::Client, url: &str) -> Result<Option<Bytes>, Box<dyn std::error::Error>> {
    let mut missing_answers = 0;
    let mut last_failure = String::new();
    for attempt in 0..SHARD_ATTEMPTS {
        if attempt > 0 {
            tokio::time::sleep(SHARD_RETRY_DELAY * (1 << (attempt - 1))).await;
        }
        match client.get(url).send().await {
            Ok(response) if response.status().is_success() => match response.bytes().await {
                Ok(bytes) => return Ok(Some(bytes)),
                Err(error) => last_failure = format!("GET {url}: {error}"),
            },
            Ok(response) => {
                let status = response.status();
                if status == reqwest::StatusCode::NOT_FOUND || status == reqwest::StatusCode::FORBIDDEN {
                    missing_answers += 1;
                }
                last_failure = format!("GET {url} -> {status}");
            }
            Err(error) => last_failure = format!("GET {url}: {error}"),
        }
    }
    if missing_answers == SHARD_ATTEMPTS {
        return Ok(None);
    }
    Err(last_failure.into())
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

/// Root column positions of `names` in a shard's schema, for a projection.
fn root_columns<const COUNT: usize>(
    builder: &ParquetRecordBatchReaderBuilder<Bytes>,
    names: [&str; COUNT],
) -> Result<[usize; COUNT], Box<dyn std::error::Error>> {
    let fields = builder.parquet_schema().root_schema().get_fields();
    let mut positions = [0; COUNT];
    for (position, name) in positions.iter_mut().zip(names) {
        *position = fields
            .iter()
            .position(|field| field.name() == name)
            .ok_or_else(|| format!("Parquet schema has no column `{name}` — is this the right source?"))?;
    }
    Ok(positions)
}

/// Appends the first conformer of every embedded molecule in one `.3D.parquet` shard: its xyz
/// points to `points` and its point count to `counts`. Returns the clouds and points appended.
fn extract_conformers<W: Write>(
    shard_bytes: Bytes,
    points: &mut W,
    counts: &mut W,
) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(shard_bytes)?;
    let columns = root_columns(&builder, ["conformer_index", "conformer_coords"])?;
    let mask = ProjectionMask::roots(builder.parquet_schema(), columns);
    let (mut clouds, mut points_appended) = (0, 0);
    for batch in builder.with_projection(mask).build()? {
        let batch = batch?;
        let indices = batch.column(0).as_primitive::<UInt8Type>();
        let coordinates = batch.column(1).as_list::<i32>();
        let scalars = coordinates.values().as_primitive::<Float16Type>().values();
        let offsets = coordinates.value_offsets();
        for row in 0..batch.num_rows() {
            // Molecules that failed to embed keep their rows with null coordinates.
            if indices.value(row) != 0 || coordinates.is_null(row) {
                continue;
            }
            let cloud = &scalars[offsets[row] as usize..offsets[row + 1] as usize];
            if !cloud.len().is_multiple_of(3) {
                return Err(format!("a conformer holds {} scalars, not xyz triples", cloud.len()).into());
            }
            // SAFETY: `f16` is POD.
            points.write_all(unsafe { retrieval::pod_slice_as_bytes(cloud) })?;
            counts.write_all(&((cloud.len() / 3) as u32).to_le_bytes())?;
            clouds += 1;
            points_appended += cloud.len() / 3;
        }
    }
    Ok((clouds, points_appended))
}

/// Writes every shard's first conformers into part pairs of `shards_per_part` input shards each,
/// resuming after the pairs an earlier run finished.
async fn download_conformers(
    cli: &Cli,
    client: &reqwest::Client,
    shards_per_part: usize,
    max_shards: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let source = cli.source.directory();
    let mut part_index = 0;
    while part_rows_of(&counts_part_path(&cli.output, part_index))?.is_some_and(|clouds| clouds != 0) {
        part_index += 1;
    }
    let first_shard = part_index * shards_per_part;
    if part_index > 0 {
        eprintln!("Keeping {part_index} finished parts, resuming at shard {first_shard}");
    }
    let progress = shard_progress(cli.limit.map(|_| max_shards));
    progress.set_position(first_shard as u64);

    let urls = (first_shard..max_shards).map(|shard_index| {
        (
            shard_index,
            shard_url(&cli.url_prefix, source, shard_index, ".3D.parquet"),
        )
    });
    let mut stream = stream::iter(urls)
        .map(|(shard_index, url)| {
            let client = client.clone();
            async move {
                let bytes = download_shard(&client, &url).await;
                (shard_index, url, bytes)
            }
        })
        .buffered(cli.download_concurrency);

    // Both files of a part open with zero rows; the points header lands first and the counts
    // header last, so an interrupted pair never passes for a finished one.
    let (mut total_clouds, mut total_points) = (0usize, 0usize);
    let mut part: Option<(BufWriter<File>, BufWriter<File>, usize, usize)> = None;
    let finish_pair = |(points, counts, clouds, points_count): (BufWriter<File>, BufWriter<File>, usize, usize)| {
        finish_part(points, points_count, 3)?;
        finish_part(counts, clouds, 1)
    };
    while let Some((shard_index, url, result)) = stream.next().await {
        let bytes = match result {
            Ok(Some(bytes)) => bytes,
            Ok(None) => {
                eprintln!("\n  shard {shard_index} missing (stopping): {url}");
                break;
            }
            Err(error) => return Err(format!("download failed for {url}: {error}").into()),
        };
        let (points, counts, clouds, points_count) = match &mut part {
            Some(part) => part,
            None => {
                let mut points = BufWriter::with_capacity(
                    OUTPUT_BUFFER_BYTES,
                    File::create(points_part_path(&cli.output, part_index))?,
                );
                let mut counts = BufWriter::with_capacity(
                    OUTPUT_BUFFER_BYTES,
                    File::create(counts_part_path(&cli.output, part_index))?,
                );
                write_bin_header_buf(&mut points, 0, 3)?;
                write_bin_header_buf(&mut counts, 0, 1)?;
                part.insert((points, counts, 0, 0))
            }
        };
        let (clouds_appended, points_appended) = extract_conformers(bytes, points, counts)?;
        *clouds += clouds_appended;
        *points_count += points_appended;
        if *points_count > u32::MAX as usize {
            return Err(
                format!("part {part_index} passed 2³² points at shard {shard_index}; lower --part-rows").into(),
            );
        }
        total_clouds += clouds_appended;
        total_points += points_appended;
        if (shard_index + 1).is_multiple_of(shards_per_part) {
            finish_pair(part.take().unwrap())?;
            part_index += 1;
        }
        progress.set_position((shard_index + 1) as u64);
        progress.set_message(format!(
            "{total_clouds} clouds, {total_points} points ({:.1} GiB)",
            (total_points * std::mem::size_of::<[numkong::f16; 3]>()) as f64 / (1u64 << 30) as f64
        ));
    }
    progress.finish_and_clear();
    if let Some(pair) = part.take() {
        finish_pair(pair)?;
        part_index += 1;
    }
    if total_clouds == 0 {
        return Err("no conformers downloaded — check --source / --url-prefix".into());
    }
    eprintln!("Downloaded {total_clouds} conformers of {total_points} points in {part_index} parts, this run");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli: Cli = retrieval::parse_cli();
    eprintln!("- Source: {}", cli.source);
    match cli.fingerprint {
        Some(fingerprint) => eprintln!("- Fingerprint: {fingerprint}"),
        None => eprintln!("- Conformers: first per embedded molecule"),
    }
    match cli.limit {
        Some(limit) => eprintln!("- Limit: {limit}"),
        None => eprintln!("- Limit: all"),
    }
    eprintln!("- Part rows: {}", cli.part_rows);
    eprintln!("- Query count: {}", cli.query_count);
    match cli.top_k {
        Some(top_k) => eprintln!("- Top k: {top_k}"),
        None => eprintln!("- Top k: none, no queries or ground truth"),
    }
    eprintln!("- Threads: {}", cli.threads);
    eprintln!("- Download concurrency: {}", cli.download_concurrency);
    eprintln!("- Seed: {}", cli.seed);
    eprintln!("- Keep Parquet: {}", cli.keep_parquet);
    eprintln!("- URL prefix: {}", cli.url_prefix);
    eprintln!("- Output: {}", cli.output.display());
    let source = cli.source.directory();

    std::fs::create_dir_all(&cli.output)?;
    let parquet_dir = cli.output.join("parquet");
    if cli.keep_parquet {
        std::fs::create_dir_all(&parquet_dir)?;
    }

    let client = reqwest::Client::builder().timeout(HTTP_REQUEST_TIME_LIMIT).build()?;

    if !cli.part_rows.is_multiple_of(SHARD_ROWS) || cli.part_rows > u32::MAX as usize {
        return Err(format!("--part-rows must be a multiple of {SHARD_ROWS} up to {}", u32::MAX).into());
    }
    let shards_per_part = cli.part_rows / SHARD_ROWS;

    // Request enough shards to cover `--limit`; without one the lazily pulled stream
    // runs until the bucket answers 403 or 404 past its last shard.
    let limit = cli.limit.unwrap_or(usize::MAX);
    let max_shards = limit.div_ceil(SHARD_ROWS);
    let Some(fingerprint) = cli.fingerprint else {
        return download_conformers(&cli, &client, shards_per_part, max_shards).await;
    };
    let info = fingerprint.info();

    // Parts an earlier run finished are kept, so a rerun fetches only the shards after them.
    let mut part_index = 0;
    while (part_index + 1) * cli.part_rows <= limit
        && part_rows_of(&part_path(&cli.output, part_index))? == Some(cli.part_rows)
    {
        part_index += 1;
    }
    let first_shard = part_index * shards_per_part;
    if part_index > 0 {
        eprintln!("Keeping {part_index} finished parts, resuming at shard {first_shard}");
    }

    let progress = shard_progress(cli.limit.map(|_| max_shards));
    progress.set_position(first_shard as u64);

    // Concurrent download with ordered processing: we need rows in shard order
    // for deterministic output. `buffered(n)` preserves request order.
    let urls = (first_shard..max_shards)
        .map(|shard_index| (shard_index, shard_url(&cli.url_prefix, source, shard_index, ".parquet")));

    // Each part is written with a zero row count, patched in once the part fills,
    // so an interrupted part never passes for a finished one.
    let mut total_rows = first_shard * SHARD_ROWS;
    let mut part: Option<(BufWriter<File>, usize)> = None;
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
        let (writer, part_rows) = match &mut part {
            Some(part) => part,
            None => {
                let file = File::create(part_path(&cli.output, part_index))?;
                let mut writer = BufWriter::with_capacity(OUTPUT_BUFFER_BYTES, file);
                write_bin_header_buf(&mut writer, 0, info.dimensions_bits as u32)?;
                part.insert((writer, 0))
            }
        };
        let appended = extract_and_append(bytes, &info, remaining, writer)?;
        *part_rows += appended;
        total_rows += appended;
        if *part_rows == cli.part_rows {
            let (writer, part_rows) = part.take().unwrap();
            finish_part(writer, part_rows, info.dimensions_bits)?;
            part_index += 1;
        }
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
    if let Some((writer, part_rows)) = part.take() {
        finish_part(writer, part_rows, info.dimensions_bits)?;
        part_index += 1;
    }

    if total_rows == 0 {
        return Err("no rows downloaded — check --source / --url-prefix".into());
    }
    eprintln!(
        "Downloaded {total_rows} molecules in {part_index} parts ({:.2} GB base data)",
        (total_rows * info.bytes_per_vector) as f64 / (1u64 << 30) as f64
    );
    let Some(top_k) = cli.top_k else {
        return Ok(());
    };

    // Memory-map the parts for query sampling + ground truth.
    let parts = (0..part_index)
        .map(|index| retrieval::Dataset::load(&part_path(&cli.output, index)))
        .collect::<std::io::Result<Vec<_>>>()?;
    let mut part_slices: Vec<&[u8]> = Vec::with_capacity(parts.len());
    for part in &parts {
        match part.slice(0, part.rows(), part.dimensions(), &mut []).data {
            retrieval::VectorSlice::B1x8(data) => part_slices.push(data),
            _ => return Err("unexpected non-binary base after load".into()),
        }
    }
    assert_eq!(parts.iter().map(retrieval::Dataset::rows).sum::<usize>(), total_rows);

    // Sample query indices without replacement.
    let query_count = cli.query_count.min(total_rows);
    let mut rng = StdRng::seed_from_u64(cli.seed.into());
    let query_indices = sample_without_replacement(&mut rng, total_rows, query_count).into_vec();

    // Copy queries into a contiguous buffer that doubles as the ground-truth input.
    // Every part but the last holds `part_rows` rows, so a row's part is its quotient.
    let mut query_buffer = {
        let mut values = Vec::new_in(std::alloc::System);
        values.resize(query_count * info.bytes_per_vector, 0u8);
        values
    };
    for (output_index, &base_index) in query_indices.iter().enumerate() {
        let source = part_slices[base_index / cli.part_rows];
        let source_offset = (base_index % cli.part_rows) * info.bytes_per_vector;
        let dest_offset = output_index * info.bytes_per_vector;
        query_buffer[dest_offset..dest_offset + info.bytes_per_vector]
            .copy_from_slice(&source[source_offset..source_offset + info.bytes_per_vector]);
    }

    let query_path = cli.output.join(format!("query.{query_count}.b1bin"));
    let mut query_file = File::create(&query_path)?;
    write_bin_header(&mut query_file, query_count as u32, info.dimensions_bits as u32)?;
    query_file.write_all(&query_buffer)?;
    eprintln!("Wrote {}", query_path.display());

    // Ground truth.
    eprintln!(
        "Computing brute-force hamming top-{} ground truth (NumKong + ForkUnion)...",
        top_k
    );
    // NumKong's u1x8 storage requires bit dimensions to be multiples of 8.
    // MACCS (166) and PubChem (881) fingerprints round up to full-byte counts;
    // trailing unused bits contribute 0 to Hamming (identical across vectors).
    let storage_bits = info.bytes_per_vector * 8;
    let base_views: Vec<_> = part_slices
        .iter()
        .map(|slice| binary_view(slice, slice.len() / info.bytes_per_vector, storage_bits))
        .collect();
    let query_view = binary_view(&query_buffer, query_count, storage_bits);
    let mut ground_truth_indices = {
        let mut values = Vec::new_in(std::alloc::System);
        values.resize(query_count * top_k, 0u32);
        values
    };
    let ground_truth_span = matrix_span(&mut ground_truth_indices, query_count, top_k);
    compute_top_k(&base_views, query_view, ground_truth_span, cli.threads)?;

    let gt_path = cli.output.join(format!("groundtruth.{query_count}.ibin"));
    let mut gt_file = File::create(&gt_path)?;
    write_bin_header(&mut gt_file, query_count as u32, top_k as u32)?;
    // SAFETY: `u32` is POD.
    gt_file.write_all(unsafe { retrieval::pod_slice_as_bytes(&ground_truth_indices) })?;
    eprintln!("Wrote {}", gt_path.display());

    if !cli.keep_parquet && parquet_dir.exists() {
        let _ = std::fs::remove_dir_all(&parquet_dir);
    }

    eprintln!("Done: {} base x {} queries, top-{}", total_rows, query_count, top_k);
    Ok(())
}
