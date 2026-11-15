//! Download a subset of the USearchMolecules dataset, extract one binary
//! fingerprint column into `.b1bin` base parts, and precompute
//! Hamming top-K ground truth with NumKong + ForkUnion; or extract the
//! lowest-energy conformer of every embedded molecule into point-cloud parts.
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
//! Every embedded PubChem molecule's lowest-energy conformer, hydrogens included, in 8 parts
//! of `f16` xyz points beside their per-cloud counts:
//! ```sh
//! retri-download-molecules \
//!     --source pubchem \
//!     --conformers \
//!     --part-rows 15000000 \
//!     --output datasets/pubchem-conformers-113m/
//! ```
//!
//! 256 conformers sampled out of the first 1M Enamine REAL molecules, as shape queries:
//! ```sh
//! retri-download-molecules \
//!     --source enamine \
//!     --conformers \
//!     --limit 1000000 \
//!     --query-count 256 \
//!     --output datasets/real-conformers-1m/
//! ```

use std::{
    fmt,
    fs::File,
    io::{BufWriter, Read, Seek, SeekFrom, Write},
    ops::Range,
    path::{Path, PathBuf},
};

use arrow_array::{
    cast::AsArray,
    types::{Float16Type, UInt16Type},
    Array, FixedSizeBinaryArray,
};
use bytes::Bytes;
use clap::Parser;
use futures::stream::{self, Stream, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use memmap2::Mmap;
use numkong as nk;
use parquet::arrow::{arrow_reader::ParquetRecordBatchReaderBuilder, ProjectionMask};
use rand::{rngs::StdRng, seq::index::sample as sample_without_replacement, SeedableRng};

use retrieval::{
    download::{download_shard, http_client, OUTPUT_BUFFER_BYTES},
    ground_truth::{write_bin_header, write_sampled_ground_truth},
    Dataset, Seed, Threads,
};

const SHARD_ROWS: usize = 1_000_000;
const BUCKET_URL_PREFIX: &str = "https://s3.us-west-2.amazonaws.com/usearch-molecules/data";

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

    /// Extract each embedded molecule's lowest-energy conformer from the `.3D.parquet` shards instead
    /// of a fingerprint: `base.part_N.f16bin` xyz points and `counts.part_N.ibin` points per cloud.
    /// With `--query-count`, also `query.N.f16bin` and `query.N.counts.ibin` of N sampled clouds.
    #[arg(long, conflicts_with_all = ["fingerprint", "top_k"])]
    conformers: bool,

    /// Maximum molecules to extract (default: all available shards).
    #[arg(long, value_parser = retrieval::parse_count_flag)]
    limit: Option<usize>,

    /// Rows per base part, `base.part_N.b1bin`, a whole number of the bucket's 1M-row shards.
    /// Keeps each part's `u32` row count in range; Enamine's 6.04B rows overflow a single file.
    #[arg(long, default_value_t = 512 * SHARD_ROWS, value_parser = retrieval::parse_count_flag)]
    part_rows: usize,

    /// Number of queries to randomly sample from the base set: fingerprints for `--top-k`, 10,000
    /// when unset; conformer clouds only when set.
    #[arg(long, value_parser = retrieval::parse_count_flag)]
    query_count: Option<usize>,

    /// Move the sampled queries out of the base parts before computing their truth, so the base
    /// holds none of them; the queries must fit in the last part.
    #[arg(long, requires = "top_k")]
    hold_out: bool,

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
        // Storage bits padded to a byte; the zero padding leaves Hamming and Jaccard unchanged.
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
    // Shards are `parquet/<start>-<end>.parquet` with 10-digit row bounds, `.3D.parquet` for conformers.
    let start = shard_index * SHARD_ROWS;
    let end = start + SHARD_ROWS;
    format!("{url_prefix}/{source}/parquet/{start:010}-{end:010}{suffix}")
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

/// Opens a part at `path` with a zero row count, so an interrupted part never passes for a
/// finished one.
fn create_part(path: &Path, width: usize) -> std::io::Result<BufWriter<File>> {
    let mut writer = BufWriter::with_capacity(OUTPUT_BUFFER_BYTES, File::create(path)?);
    write_bin_header(&mut writer, 0, width)?;
    Ok(writer)
}

/// Patches a written part's row count into its header and syncs it to disk.
fn finish_part(writer: BufWriter<File>, rows: usize, width: usize) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = writer
        .into_inner()
        .map_err(|error| format!("flush part writer: {error}"))?;
    file.seek(SeekFrom::Start(0))?;
    write_bin_header(&mut file, rows, width)?;
    file.sync_all()?;
    Ok(())
}

/// Shard download progress, with a length only when `--limit` bounds the shards.
fn shard_progress(shards: Option<usize>, first_shard: usize) -> ProgressBar {
    let progress = shards.map_or_else(ProgressBar::no_length, |shards| ProgressBar::new(shards as u64));
    progress.set_style(
        ProgressStyle::with_template("  {spinner} downloading shards: {pos}/{len} [{elapsed_precise}] {msg}").unwrap(),
    );
    progress.set_position(first_shard as u64);
    progress
}

/// Downloads `shards` in order, `--download-concurrency` at a time, as each shard's index, URL and
/// bytes; `None` bytes mean the bucket ended before that shard.
fn shard_downloads<'a>(
    client: &'a reqwest::Client,
    cli: &'a Cli,
    shards: Range<usize>,
    suffix: &'a str,
) -> impl Stream<Item = (usize, String, Result<Option<Bytes>, String>)> + 'a {
    stream::iter(shards)
        .map(move |shard_index| {
            let url = shard_url(&cli.url_prefix, cli.source.directory(), shard_index, suffix);
            async move {
                let bytes = download_shard(client, &url).await;
                (shard_index, url, bytes)
            }
        })
        .buffered(cli.download_concurrency)
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

/// Read one shard's Parquet bytes, extract the requested fingerprint column,
/// and append at most `take` rows to `output`. Returns the number of rows appended.
fn extract_and_append<W: Write>(
    shard_bytes: Bytes,
    info: &FingerprintInfo,
    take: usize,
    output: &mut W,
) -> Result<usize, Box<dyn std::error::Error>> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(shard_bytes)?;
    let mask = ProjectionMask::roots(builder.parquet_schema(), root_columns(&builder, [info.column_name])?);
    let mut appended = 0;
    for batch in builder.with_projection(mask).build()? {
        let batch = batch?;
        let fingerprints = batch
            .column(0)
            .as_any()
            .downcast_ref::<FixedSizeBinaryArray>()
            .ok_or("expected FixedSizeBinaryArray for fingerprint column")?;
        let width = fingerprints.value_length() as usize;
        if width != info.bytes_per_vector {
            return Err(format!(
                "fingerprint `{}` has width {width} in Parquet but expected {}",
                info.column_name, info.bytes_per_vector
            )
            .into());
        }
        // `.value(row)` applies the array's slice offset; the BufWriter makes the per-row calls cheap.
        let rows = fingerprints.len().min(take - appended);
        for row in 0..rows {
            output.write_all(fingerprints.value(row))?;
        }
        appended += rows;
        if appended == take {
            break;
        }
    }
    Ok(appended)
}

/// Appends the lowest-energy conformer of every embedded molecule in one `.3D.parquet` shard,
/// whose rows hold each molecule's conformers end to end in `points`, `n_atoms` atoms apiece and
/// lowest energy first: its xyz points to `points` and its point count to `counts`. Returns the
/// clouds and points appended.
fn extract_conformers<W: Write>(
    shard_bytes: Bytes,
    points: &mut W,
    counts: &mut W,
) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(shard_bytes)?;
    let columns = root_columns(&builder, ["n_atoms", "points"])?;
    let mask = ProjectionMask::roots(builder.parquet_schema(), columns);
    let (mut clouds, mut points_appended) = (0, 0);
    for batch in builder.with_projection(mask).build()? {
        let batch = batch?;
        let atoms = batch.column(0).as_primitive::<UInt16Type>();
        let conformers = batch.column(1).as_list::<i32>();
        let scalars = conformers
            .values()
            .as_fixed_size_list()
            .values()
            .as_primitive::<Float16Type>()
            .values();
        let offsets = conformers.value_offsets();
        for (row, &first) in offsets.iter().enumerate().take(batch.num_rows()) {
            let atoms = usize::from(atoms.value(row));
            let (first, stored) = (first as usize, conformers.value_length(row) as usize);
            // Molecules that failed to embed keep their rows with no atoms or no points.
            if atoms == 0 || stored == 0 {
                continue;
            }
            if stored < atoms {
                return Err(format!("a row holds {stored} points, fewer than its {atoms} atoms").into());
            }
            let cloud = &scalars[first * 3..(first + atoms) * 3];
            // SAFETY: `f16` is POD.
            points.write_all(unsafe { retrieval::pod_slice_as_bytes(cloud) })?;
            counts.write_all(&((cloud.len() / 3) as u32).to_le_bytes())?;
            clouds += 1;
            points_appended += cloud.len() / 3;
        }
    }
    Ok((clouds, points_appended))
}

/// Writes the fingerprints of `max_shards` shards, at most `limit` rows, into parts of
/// `shards_per_part` shards each, resuming after the parts an earlier run finished. Returns the
/// rows across every part.
async fn download_fingerprints(
    cli: &Cli,
    client: &reqwest::Client,
    info: &FingerprintInfo,
    shards_per_part: usize,
    max_shards: usize,
) -> Result<usize, Box<dyn std::error::Error>> {
    let limit = cli.limit.unwrap_or(usize::MAX);
    let parquet_directory = cli.output.join("parquet");
    if cli.keep_parquet {
        std::fs::create_dir_all(&parquet_directory)?;
    }
    let mut part_index = 0;
    while (part_index + 1) * cli.part_rows <= limit
        && part_rows_of(&part_path(&cli.output, part_index))? == Some(cli.part_rows)
    {
        part_index += 1;
    }
    if part_index > 0 {
        eprintln!(
            "Keeping {part_index} finished parts, resuming at shard {}",
            part_index * shards_per_part
        );
    }
    let progress = shard_progress(cli.limit.map(|_| max_shards), part_index * shards_per_part);

    let mut total_rows = part_index * cli.part_rows;
    let mut bucket_ended = false;
    while !bucket_ended && total_rows < limit {
        let first_shard = part_index * shards_per_part;
        let shards = first_shard..(first_shard + shards_per_part).min(max_shards);
        if shards.is_empty() {
            break;
        }
        let path = part_path(&cli.output, part_index);
        let mut writer = create_part(&path, info.dimensions_bits)?;
        let mut rows = 0;
        let mut downloads = std::pin::pin!(shard_downloads(client, cli, shards, ".parquet"));
        while let Some((shard_index, url, result)) = downloads.next().await {
            let Some(bytes) = result? else {
                eprintln!("\n  shard {shard_index} missing (stopping): {url}");
                bucket_ended = true;
                break;
            };
            if cli.keep_parquet {
                std::fs::write(
                    parquet_directory.join(format!("shard_{shard_index:03}.parquet")),
                    &bytes,
                )?;
            }
            rows += extract_and_append(bytes, info, limit - total_rows - rows, &mut writer)?;
            progress.set_position((shard_index + 1) as u64);
            progress.set_message(format!(
                "{} rows ({:.1} GB)",
                total_rows + rows,
                ((total_rows + rows) * info.bytes_per_vector) as f64 / (1u64 << 30) as f64
            ));
            if total_rows + rows == limit {
                break;
            }
        }
        match rows {
            0 => std::fs::remove_file(&path)?,
            _ => {
                finish_part(writer, rows, info.dimensions_bits)?;
                part_index += 1;
            }
        }
        total_rows += rows;
    }
    progress.finish_and_clear();
    if total_rows == 0 {
        return Err("no rows downloaded — check --source / --url-prefix".into());
    }
    eprintln!(
        "Downloaded {total_rows} molecules in {part_index} parts ({:.2} GB base data)",
        (total_rows * info.bytes_per_vector) as f64 / (1u64 << 30) as f64
    );
    Ok(total_rows)
}

/// Writes every shard's lowest-energy conformers into part pairs of `shards_per_part` input shards each,
/// resuming after the pairs an earlier run finished.
async fn download_conformers(
    cli: &Cli,
    client: &reqwest::Client,
    shards_per_part: usize,
    max_shards: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut part_index = 0;
    while part_rows_of(&counts_part_path(&cli.output, part_index))?.is_some_and(|clouds| clouds != 0) {
        part_index += 1;
    }
    if part_index > 0 {
        eprintln!(
            "Keeping {part_index} finished parts, resuming at shard {}",
            part_index * shards_per_part
        );
    }
    let progress = shard_progress(cli.limit.map(|_| max_shards), part_index * shards_per_part);

    // Both files of a pair open with zero rows; the points header lands first and the counts
    // header last, so an interrupted pair never passes for a finished one.
    let (mut total_clouds, mut total_points) = (0, 0);
    let mut bucket_ended = false;
    while !bucket_ended {
        let first_shard = part_index * shards_per_part;
        let shards = first_shard..(first_shard + shards_per_part).min(max_shards);
        if shards.is_empty() {
            break;
        }
        let (points_path, counts_path) = (
            points_part_path(&cli.output, part_index),
            counts_part_path(&cli.output, part_index),
        );
        let (mut points, mut counts) = (create_part(&points_path, 3)?, create_part(&counts_path, 1)?);
        let (mut clouds, mut points_count) = (0, 0);
        let mut downloads = std::pin::pin!(shard_downloads(client, cli, shards, ".3D.parquet"));
        while let Some((shard_index, url, result)) = downloads.next().await {
            let Some(bytes) = result? else {
                eprintln!("\n  shard {shard_index} missing (stopping): {url}");
                bucket_ended = true;
                break;
            };
            let (clouds_appended, points_appended) = extract_conformers(bytes, &mut points, &mut counts)?;
            clouds += clouds_appended;
            points_count += points_appended;
            if points_count > u32::MAX as usize {
                return Err(
                    format!("part {part_index} passed 2³² points at shard {shard_index}; lower --part-rows").into(),
                );
            }
            progress.set_position((shard_index + 1) as u64);
            progress.set_message(format!(
                "{} clouds, {} points ({:.1} GiB)",
                total_clouds + clouds,
                total_points + points_count,
                ((total_points + points_count) * std::mem::size_of::<[nk::f16; 3]>()) as f64 / (1u64 << 30) as f64
            ));
        }
        match clouds {
            0 => {
                std::fs::remove_file(&points_path)?;
                std::fs::remove_file(&counts_path)?;
            }
            _ => {
                finish_part(points, points_count, 3)?;
                finish_part(counts, clouds, 1)?;
                part_index += 1;
            }
        }
        total_clouds += clouds;
        total_points += points_count;
    }
    progress.finish_and_clear();
    if total_clouds == 0 {
        return Err("no conformers downloaded — check --source / --url-prefix".into());
    }
    eprintln!("Downloaded {total_clouds} conformers of {total_points} points in {part_index} parts, this run");
    Ok(())
}

/// Samples `query_count` clouds of every finished conformer part under `seed` into
/// `query.<count>.f16bin` points and `query.<count>.counts.ibin` points per cloud, in part order.
fn write_sampled_conformers(output: &Path, query_count: usize, seed: Seed) -> Result<(), Box<dyn std::error::Error>> {
    const HEADER_BYTES: usize = 8;
    const POINT_BYTES: usize = std::mem::size_of::<[nk::f16; 3]>();
    let mut parts = Vec::new();
    for part_index in 0.. {
        if !part_rows_of(&counts_part_path(output, part_index))?.is_some_and(|clouds| clouds != 0) {
            break;
        }
        // SAFETY: finished parts are never written again while mapped.
        let counts = unsafe { Mmap::map(&File::open(counts_part_path(output, part_index))?)? };
        let points = unsafe { Mmap::map(&File::open(points_part_path(output, part_index))?)? };
        parts.push((counts, points));
    }
    let clouds: usize = parts.iter().map(|(counts, _)| (counts.len() - HEADER_BYTES) / 4).sum();
    let query_count = query_count.min(clouds);
    let mut sampled =
        sample_without_replacement(&mut StdRng::seed_from_u64(seed.into()), clouds, query_count).into_vec();
    sampled.sort_unstable();

    let (points_path, counts_path) = (
        output.join(format!("query.{query_count}.f16bin")),
        output.join(format!("query.{query_count}.counts.ibin")),
    );
    let (mut points, mut counts) = (create_part(&points_path, 3)?, create_part(&counts_path, 1)?);
    let mut wanted = sampled.iter().copied().peekable();
    let (mut first_cloud, mut points_written) = (0, 0);
    for (part_counts, part_points) in &parts {
        let mut point_offset = 0;
        for (cloud, &count) in part_counts[HEADER_BYTES..].as_chunks::<4>().0.iter().enumerate() {
            let count = u32::from_le_bytes(count) as usize;
            if wanted.next_if_eq(&(first_cloud + cloud)).is_some() {
                let start = HEADER_BYTES + point_offset * POINT_BYTES;
                points.write_all(&part_points[start..start + count * POINT_BYTES])?;
                counts.write_all(&(count as u32).to_le_bytes())?;
                points_written += count;
            }
            point_offset += count;
        }
        first_cloud += (part_counts.len() - HEADER_BYTES) / 4;
    }
    finish_part(points, points_written, 3)?;
    finish_part(counts, query_count, 1)?;
    eprintln!(
        "Sampled {query_count} of {clouds} clouds, {points_written} points, into {}",
        points_path.display()
    );
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli: Cli = retrieval::parse_cli();
    eprintln!("- Source: {}", cli.source);
    match cli.fingerprint {
        Some(fingerprint) => eprintln!("- Fingerprint: {fingerprint}"),
        None => eprintln!("- Conformers: lowest energy per embedded molecule"),
    }
    match cli.limit {
        Some(limit) => eprintln!("- Limit: {limit}"),
        None => eprintln!("- Limit: all"),
    }
    eprintln!("- Part rows: {}", cli.part_rows);
    match cli.query_count {
        Some(query_count) => eprintln!("- Query count: {query_count}"),
        None => eprintln!("- Query count: default"),
    }
    match cli.top_k {
        Some(top_k) => eprintln!("- Top k: {top_k}"),
        None => eprintln!("- Top k: none, no queries or ground truth"),
    }
    eprintln!("- Hold out: {}", cli.hold_out);
    eprintln!("- Threads: {}", cli.threads);
    eprintln!("- Download concurrency: {}", cli.download_concurrency);
    eprintln!("- Seed: {}", cli.seed);
    eprintln!("- Keep Parquet: {}", cli.keep_parquet);
    eprintln!("- URL prefix: {}", cli.url_prefix);
    eprintln!("- Output: {}", cli.output.display());
    std::fs::create_dir_all(&cli.output)?;
    let client = http_client()?;

    if !cli.part_rows.is_multiple_of(SHARD_ROWS) || cli.part_rows > u32::MAX as usize {
        return Err(format!("--part-rows must be a multiple of {SHARD_ROWS} up to {}", u32::MAX).into());
    }
    let shards_per_part = cli.part_rows / SHARD_ROWS;
    // Enough shards to cover `--limit`; without one the lazily pulled stream runs until the
    // bucket answers 403 or 404 past its last shard.
    let max_shards = cli.limit.unwrap_or(usize::MAX).div_ceil(SHARD_ROWS);
    let Some(fingerprint) = cli.fingerprint else {
        download_conformers(&cli, &client, shards_per_part, max_shards).await?;
        return match cli.query_count {
            Some(query_count) => write_sampled_conformers(&cli.output, query_count, cli.seed),
            None => Ok(()),
        };
    };
    let total_rows = download_fingerprints(&cli, &client, &fingerprint.info(), shards_per_part, max_shards).await?;
    let Some(top_k) = cli.top_k else {
        return Ok(());
    };

    let base = Dataset::load(&cli.output.join("base.part_*.b1bin"))?;
    if base.rows() != total_rows {
        return Err(format!(
            "the parts hold {} rows, not the {total_rows} this run wrote; remove stale parts from {}",
            base.rows(),
            cli.output.display()
        )
        .into());
    }
    let query_count = cli.query_count.unwrap_or(10_000);
    write_sampled_ground_truth(
        base,
        query_count,
        top_k,
        cli.seed,
        cli.threads,
        cli.hold_out,
        &cli.output,
    )?;
    let query_count = query_count.min(total_rows);
    let base_rows = if cli.hold_out {
        total_rows - query_count
    } else {
        total_rows
    };
    eprintln!("Done: {base_rows} base x {query_count} queries, top-{top_k}");
    Ok(())
}
