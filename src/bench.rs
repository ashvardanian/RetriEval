//! Shared benchmark infrastructure for vector search engines.
//!
//! This is the library root. Backend binaries (`usearch.rs`, `faiss.rs`, etc.)
//! import from here and provide their own `main()`.

pub mod dataset;
#[cfg(feature = "tier2")]
pub mod docker;
#[cfg(feature = "download")]
pub mod download;
pub mod error;
pub mod eval;
#[cfg(feature = "ground-truth")]
pub mod ground_truth;
pub mod output;
#[cfg(feature = "ground-truth")]
pub mod packed_distance;
pub mod perf_counters;

use std::{
    alloc::{Allocator, System},
    collections::HashMap,
    fmt::{self, Write as _},
    hash::{BuildHasher, Hasher},
    num::{NonZeroU16, NonZeroUsize},
    ops::Range,
    path::{Path, PathBuf},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use forkunion::{self as fu, ThreadPool, Topology};
use indicatif::{ProgressBar, ProgressStyle};
use numkong::{self as nk, MatrixSpan, MatrixView};
use serde_json::Value;

pub use crate::{
    dataset::{Corpus, Dataset, GroundTruth, Keys, Permutation, Shuffle, Targets, TextCorpus, Texts, Truth},
    error::{DatasetError, GroundTruthError},
    output::{
        config_hash, write_report, ConfigReport, DatasetInfo, StepAddEntry, StepEntry, StepRefineEntry, StepSearchEntry,
    },
};

// #region Core types

/// A vector's identity: an integer widened to 128 bits, or a UUID's 16 bytes read most significant first.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Key(pub u128);

impl Key {
    /// Pads the result slots a backend leaves unfilled.
    pub const MISSING: Key = Key(u128::MAX);
}

impl From<u32> for Key {
    fn from(value: u32) -> Self {
        Key(u128::from(value))
    }
}

impl From<u64> for Key {
    fn from(value: u64) -> Self {
        Key(u128::from(value))
    }
}

impl TryFrom<Key> for u64 {
    type Error = std::num::TryFromIntError;

    fn try_from(key: Key) -> Result<Self, Self::Error> {
        u64::try_from(key.0)
    }
}

impl fmt::Display for Key {
    /// The lowercase dashed UUID spelling, which integer keys take too, zero-padded.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bits = self.0;
        write!(
            formatter,
            "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
            bits >> 96,
            (bits >> 80) & 0xFFFF,
            (bits >> 64) & 0xFFFF,
            (bits >> 48) & 0xFFFF,
            bits & 0xFFFF_FFFF_FFFF,
        )
    }
}

impl std::str::FromStr for Key {
    type Err = String;

    /// Parses 32 hexadecimal digits, dashes anywhere, as `Display` spells them.
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let digits = text.bytes().filter(|&byte| byte != b'-');
        let (count, bits) = digits.fold((0, Some(0u128)), |(count, bits), byte| {
            let digit = (byte as char).to_digit(16).map(u128::from);
            (count + 1, bits.zip(digit).map(|(bits, digit)| bits << 4 | digit))
        });
        match (count, bits) {
            (32, Some(bits)) => Ok(Key(bits)),
            _ => Err(format!("{text:?} is not a UUID of 32 hexadecimal digits")),
        }
    }
}

impl nk::StorageElement for Key {
    fn zero() -> Self {
        Key(0)
    }
    fn one() -> Self {
        Key(1)
    }
}

/// Keys generated when `--base-keys` is unset, one per base row and unique to it.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyType {
    U32,
    U64,
    /// Random UUIDs, scattered over the whole key space.
    #[value(name = "uuid-v4")]
    UuidV4,
    /// Time-ordered UUIDs: the row is the timestamp and counter, so keys sort as rows do.
    #[value(name = "uuid-v7")]
    UuidV7,
}

impl KeyType {
    /// The key of base `row`.
    pub fn encode(self, row: usize) -> Key {
        let row = row as u64;
        match self {
            Self::U32 => Key::from(row as u32),
            Self::U64 => Key::from(row),
            Self::UuidV4 => uuid(4, mix(row), mix(row ^ SPLITMIX64_GAMMA)),
            Self::UuidV7 => uuid(7, row, mix(row)),
        }
    }

    /// The row [`KeyType::encode`] could have named `key` for; the caller checks it round-trips.
    pub fn decode(self, key: Key) -> Option<usize> {
        let payload = (key.0 >> 80) << 16 | (key.0 >> 64 & 0xFFF) << 4 | (key.0 >> 58 & 0xF);
        let row = match self {
            Self::U32 | Self::U64 => u64::try_from(key.0).ok()?,
            Self::UuidV4 => unmix(payload as u64),
            Self::UuidV7 => payload as u64,
        };
        usize::try_from(row).ok()
    }

    /// Bytes per key: 4, 8 or 16.
    pub fn key_bytes(self) -> usize {
        match self {
            Self::U32 => 4,
            Self::U64 => 8,
            Self::UuidV4 | Self::UuidV7 => 16,
        }
    }
}

impl fmt::Display for KeyType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        spell_value(self, formatter)
    }
}

/// An RFC 9562 UUID of `version` that keeps all 64 `payload` bits, in order, around the version and
/// variant fields, so distinct payloads never collide; `filler` supplies the remaining 58 bits.
fn uuid(version: u128, payload: u64, filler: u64) -> Key {
    let (payload, filler) = (u128::from(payload), u128::from(filler));
    Key((payload >> 16) << 80
        | version << 76
        | (payload >> 4 & 0xFFF) << 64
        | 0b10 << 62
        | (payload & 0xF) << 58
        | filler & ((1 << 58) - 1))
}

/// Distance/similarity value returned by search operations.
pub type Distance = f32;

// #region Vector types

/// A batch of row-major vectors, one matrix row per vector; `B1x8` rows hold `dimensions` bits.
#[derive(Clone, Copy)]
pub enum Vectors<'a> {
    F32(MatrixView<'a, f32>),
    I8(MatrixView<'a, i8>),
    U8(MatrixView<'a, u8>),
    B1x8(MatrixView<'a, nk::u1x8>),
}

/// A batch's values end to end, the layout files and wire formats take.
pub enum VectorSlice<'a> {
    F32(&'a [f32]),
    I8(&'a [i8]),
    U8(&'a [u8]),
    /// Binary vectors: 1-bit values packed 8 per byte.
    B1x8(&'a [u8]),
}

impl<'a> Vectors<'a> {
    /// Shapes `values` into rows `dimensions` wide, counted in bits for `B1x8`.
    pub fn new(values: VectorSlice<'a>, dimensions: usize) -> Self {
        match values {
            VectorSlice::F32(values) => Self::F32(matrix_view(values, values.len() / dimensions, dimensions)),
            VectorSlice::I8(values) => Self::I8(matrix_view(values, values.len() / dimensions, dimensions)),
            VectorSlice::U8(values) => Self::U8(matrix_view(values, values.len() / dimensions, dimensions)),
            VectorSlice::B1x8(bytes) => Self::B1x8(binary_view(bytes, bytes.len() / (dimensions / 8), dimensions)),
        }
    }

    fn shape(&self) -> &[usize] {
        match self {
            Self::F32(view) => view.shape(),
            Self::I8(view) => view.shape(),
            Self::U8(view) => view.shape(),
            Self::B1x8(view) => view.shape(),
        }
    }

    pub fn len(&self) -> usize {
        self.shape()[0]
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Values per row, counted in bits for `B1x8`.
    pub fn dimensions(&self) -> usize {
        self.shape()[1]
    }

    /// The rows end to end.
    pub fn values(&self) -> VectorSlice<'_> {
        const WHOLE_ROWS: &str = "a batch holds whole contiguous rows";
        match self {
            Self::F32(view) => VectorSlice::F32(view.as_packed_slice().expect(WHOLE_ROWS)),
            Self::I8(view) => VectorSlice::I8(view.as_packed_slice().expect(WHOLE_ROWS)),
            Self::U8(view) => VectorSlice::U8(view.as_packed_slice().expect(WHOLE_ROWS)),
            // SAFETY: `u1x8` is a transparent byte.
            Self::B1x8(view) => {
                VectorSlice::B1x8(unsafe { pod_slice_as_bytes(view.as_packed_slice().expect(WHOLE_ROWS)) })
            }
        }
    }

    /// The values as `f32`, cast into `scratch` unless they already are.
    pub fn to_f32_in<'s, A: Allocator>(&'s self, scratch: &'s mut Vec<f32, A>) -> Result<&'s [f32], &'static str> {
        match self.values() {
            VectorSlice::F32(values) => Ok(values),
            values => cast_in(&values, scratch),
        }
    }
}

/// Casts every value of `source` into `target`, resized to fit, with NumKong's conversions.
fn cast_in<'a, T: nk::CastDType + Copy + Default, A: Allocator>(
    source: &VectorSlice<'_>,
    target: &'a mut Vec<T, A>,
) -> Result<&'a [T], &'static str> {
    let cast = match source {
        VectorSlice::F32(values) => cast_values(values, target),
        VectorSlice::I8(values) => cast_values(values, target),
        VectorSlice::U8(values) => cast_values(values, target),
        VectorSlice::B1x8(_) => return Err("Packed binary input cannot be cast to a dense type without a bit metric"),
    };
    cast.expect("matching conversion lengths");
    Ok(target)
}

fn cast_values<Source: nk::CastDType, T: nk::CastDType + Copy + Default, A: Allocator>(
    values: &[Source],
    target: &mut Vec<T, A>,
) -> Result<(), nk::Error> {
    target.clear();
    target.resize(values.len(), T::default());
    nk::cast(values, target)
}

/// A backend's wire buffer in its stored scalar type: input already in that type passes through as
/// bytes, and any other input is cast into the buffer.
pub enum CastScratch {
    F64(Vec<f64, System>),
    F32(Vec<f32, System>),
    BF16(Vec<nk::bf16, System>),
    F16(Vec<nk::f16, System>),
    I8(Vec<i8, System>),
    U8(Vec<u8, System>),
}

impl CastScratch {
    /// The bytes of `vectors` in this buffer's scalar type.
    pub fn encode<'a>(&'a mut self, vectors: &'a Vectors<'_>) -> Result<&'a [u8], &'static str> {
        let source = vectors.values();
        // SAFETY: every scalar type here is POD.
        Ok(match (self, source) {
            (Self::F32(_), VectorSlice::F32(values)) => unsafe { pod_slice_as_bytes(values) },
            (Self::I8(_), VectorSlice::I8(values)) => unsafe { pod_slice_as_bytes(values) },
            (Self::U8(_), VectorSlice::U8(values)) => values,
            (Self::F64(target), source) => unsafe { pod_slice_as_bytes(cast_in(&source, target)?) },
            (Self::F32(target), source) => unsafe { pod_slice_as_bytes(cast_in(&source, target)?) },
            (Self::BF16(target), source) => unsafe { pod_slice_as_bytes(cast_in(&source, target)?) },
            (Self::F16(target), source) => unsafe { pod_slice_as_bytes(cast_in(&source, target)?) },
            (Self::I8(target), source) => unsafe { pod_slice_as_bytes(cast_in(&source, target)?) },
            (Self::U8(target), source) => unsafe { pod_slice_as_bytes(cast_in(&source, target)?) },
        })
    }
}

/// A `[rows, bits]` view of bit-packed rows; `bits` must be a whole number of bytes.
pub fn binary_view(bytes: &[u8], rows: usize, bits: usize) -> MatrixView<'_, nk::u1x8> {
    assert!(
        bits.is_multiple_of(8) && bytes.len() == rows * (bits / 8),
        "{} bytes do not hold {rows} rows of {bits} bits",
        bytes.len()
    );
    // SAFETY: the bytes hold `rows` packed rows of `bits / 8` bytes, and `u1x8` is a transparent byte.
    unsafe { MatrixView::from_raw_parts(bytes.as_ptr().cast(), &[rows, bits], &[(bits / 8) as isize, 1]) }
        .expect("whole packed rows are a valid view")
}

/// A `[rows, columns]` view of row-major values.
pub fn matrix_view<T: nk::StorageElement>(values: &[T], rows: usize, columns: usize) -> MatrixView<'_, T> {
    assert_eq!(
        values.len(),
        rows * columns,
        "values do not hold {rows} rows of {columns}"
    );
    let element = std::mem::size_of::<T>() as isize;
    // SAFETY: the values hold `rows` rows of `columns` elements end to end.
    unsafe {
        MatrixView::from_raw_parts(
            values.as_ptr(),
            &[rows, columns],
            &[columns as isize * element, element],
        )
    }
    .expect("whole rows are a valid view")
}

/// This process's anonymous resident bytes, `RssAnon`, leaving out the mapped files a resident
/// set also counts, such as the base vectors; zero where the kernel does not report it.
pub fn process_anon_bytes() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    status
        .lines()
        .find_map(|line| line.strip_prefix("RssAnon:"))
        .and_then(|value| value.trim().strip_suffix("kB"))
        .and_then(|kibibytes| kibibytes.trim().parse::<u64>().ok())
        .map_or(0, |kibibytes| kibibytes << 10)
}

/// The base rows of one batch, global across shards, resolved once so the harness and the backend
/// read the same rows.
#[derive(Clone)]
pub enum BatchRows<'a> {
    /// File order: the rows themselves.
    Range(Range<usize>),
    /// Shuffled order: the rows the permutation produced for this batch.
    Shuffled(&'a [usize]),
}

/// One batch's keys, read through the run's [`Keys`] in insertion order, so the batch owns no buffer.
#[derive(Clone)]
pub struct BatchKeys<'a> {
    keys: &'a Keys,
    rows: BatchRows<'a>,
}

impl<'a> BatchKeys<'a> {
    pub fn new(keys: &'a Keys, rows: BatchRows<'a>) -> Self {
        Self { keys, rows }
    }

    pub fn len(&self) -> usize {
        match &self.rows {
            BatchRows::Range(rows) => rows.len(),
            BatchRows::Shuffled(rows) => rows.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bytes per key, for engines that size a key column once: 4, 8 or 16.
    pub fn key_bytes(&self) -> usize {
        self.keys.key_bytes()
    }

    /// The base rows of the batch, for engines that store rows rather than wide keys.
    pub fn rows(&self) -> impl ExactSizeIterator<Item = usize> + 'a {
        let rows = self.rows.clone();
        (0..self.len()).map(move |index| match &rows {
            BatchRows::Range(rows) => rows.start + index,
            BatchRows::Shuffled(rows) => rows[index],
        })
    }

    pub fn keys(&self) -> impl ExactSizeIterator<Item = Key> + 'a {
        let keys = self.keys;
        self.rows().map(move |row| keys.get(row))
    }
}

// #region Backend trait

/// Common trait for all search backends, each reading the rows of one kind of corpus.
pub trait Backend: Send {
    type Corpus: Corpus;
    fn metadata(&self) -> HashMap<String, Value>;
    fn add(&mut self, keys: BatchKeys<'_>, rows: Rows<'_, Self>) -> Result<(), String>;
    /// Answers every row of `queries` into the matching row of `hits`.
    fn search(&mut self, queries: Rows<'_, Self>, hits: Hits<'_>) -> Result<(), String>;
    fn memory_bytes(&self) -> usize;

    /// Waits until everything added is searchable as the index will finally serve it, for servers
    /// that index in the background; timed into the add phase.
    fn flush(&mut self) -> Result<(), String> {
        Ok(())
    }

    /// Completes the build after the last insert step, before its search: refinement passes, then
    /// any move to where the index serves from, reporting the passes when there were any.
    fn finish(&mut self) -> Result<Option<StepRefineEntry>, String> {
        Ok(None)
    }

    /// How many entries the backend holds, when it can say. A loaded index may
    /// cover a different slice of the base than this run's flags imply, and
    /// self-recall queries base rows assuming each one is indexed.
    fn indexed_count(&self) -> Option<usize> {
        None
    }

    /// Persist the index at `path`.
    fn save(&self, _path: &str) -> Result<(), String> {
        Err("--index: save not yet implemented for this backend".into())
    }
}

/// One batch of the rows backend `B` reads.
pub type Rows<'a, B> = <<B as Backend>::Corpus as Corpus>::Rows<'a>;

// #region Index configuration

/// The hyper-parameters every graph backend understands, in the shared
/// vocabulary. Backends take this by value and add their own engine-specific
/// knobs alongside; naming the fields is what stops `metric` and `data_type`,
/// or the run of `usize` graph knobs, from being transposed at a call site.
#[derive(Clone, Copy)]
pub struct IndexConfig<DataType, Metric> {
    pub dimensions: usize,
    pub data_type: DataType,
    pub metric: Metric,
    pub connectivity: usize,
    pub expansion_add: usize,
    pub expansion_search: usize,
}

/// Where a config's index comes from and goes to, decided once from `--index`.
#[derive(Clone, Copy)]
pub enum IndexFile<'a> {
    None,
    Save(&'a str),
    Load(&'a str),
}

impl<'a> IndexFile<'a> {
    /// Loads `path` when it exists and saves to it otherwise; a path serves a single config,
    /// since several would write to or read from the same file.
    pub fn new(path: Option<&'a str>, configs: usize) -> Self {
        match path {
            None => Self::None,
            Some(_) if configs > 1 => bail(&format!(
                "--index requires a single config; got {configs} configs from the sweep"
            )),
            Some(path) if Path::new(path).exists() => Self::Load(path),
            Some(path) => Self::Save(path),
        }
    }
}

impl fmt::Display for IndexFile<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => formatter.write_str("none"),
            Self::Save(path) => write!(formatter, "save to {path}"),
            Self::Load(path) => write!(formatter, "load {path}"),
        }
    }
}

// #region Result buffers

/// Every query's answers for one pass: a `[queries, count]` key matrix, its distances, and how
/// many hits each row holds.
pub struct SearchBuffers {
    keys: nk::Matrix<Key>,
    distances: nk::Matrix<Distance>,
    counts: Vec<usize, System>,
}

impl SearchBuffers {
    pub fn new(queries: usize, count: usize) -> Result<Self, String> {
        let error = |error| format!("result matrices for {queries} queries of {count} hits: {error:?}");
        let mut counts = Vec::with_capacity_in(queries, System);
        counts.resize(queries, 0);
        Ok(Self {
            keys: nk::Matrix::full(&[queries, count], Key::MISSING).map_err(error)?,
            distances: nk::Matrix::full(&[queries, count], Distance::INFINITY).map_err(error)?,
            counts,
        })
    }

    /// The rows of one batch, for a backend to answer into; `base_keys` names the rows an engine
    /// reports instead of keys.
    pub fn hits<'a>(&'a mut self, rows: Range<usize>, base_keys: &'a Keys) -> Hits<'a> {
        const WHOLE_ROWS: &str = "a batch is a run of whole rows";
        Hits {
            keys: self.keys.slice_mut((rows.clone(), ..)).expect(WHOLE_ROWS),
            distances: self.distances.slice_mut((rows.clone(), ..)).expect(WHOLE_ROWS),
            counts: &mut self.counts[rows],
            base_keys,
        }
    }

    /// Each query's keys, distances and hit count, in query order.
    pub fn rows(&self) -> impl Iterator<Item = (&[Key], &[Distance], usize)> {
        let count = self.keys.shape()[1];
        self.keys
            .as_slice()
            .chunks_exact(count)
            .zip(self.distances.as_slice().chunks_exact(count))
            .zip(&self.counts)
            .map(|((keys, distances), &found)| (keys, distances, found))
    }

    /// Each query's found keys, best first.
    pub fn found(&self) -> impl Iterator<Item = &[Key]> {
        self.rows().map(|(keys, _, found)| &keys[..found])
    }

    /// Each query's found keys and their distances, best first.
    pub fn found_hits(&self) -> impl Iterator<Item = (&[Key], &[Distance])> {
        self.rows()
            .map(|(keys, distances, found)| (&keys[..found], &distances[..found]))
    }
}

/// One batch's rows of a [`SearchBuffers`].
pub struct Hits<'a> {
    keys: MatrixSpan<'a, Key>,
    distances: MatrixSpan<'a, Distance>,
    counts: &'a mut [usize],
    base_keys: &'a Keys,
}

impl Hits<'_> {
    /// Hits requested per query.
    pub fn count(&self) -> usize {
        self.keys.shape()[1]
    }

    /// Each query's row, in query order.
    pub fn rows(&mut self) -> impl Iterator<Item = HitRow<'_>> {
        const WHOLE_ROWS: &str = "a batch is a run of whole rows";
        let count = self.count();
        let base_keys = self.base_keys;
        let keys = self.keys.as_contiguous_slice_mut().expect(WHOLE_ROWS);
        let distances = self.distances.as_contiguous_slice_mut().expect(WHOLE_ROWS);
        keys.chunks_exact_mut(count)
            .zip(distances.chunks_exact_mut(count))
            .zip(self.counts.iter_mut())
            .map(move |((keys, distances), found)| HitRow {
                keys,
                distances,
                found,
                base_keys,
            })
    }
}

/// One query's row of a [`Hits`] batch.
pub struct HitRow<'a> {
    keys: &'a mut [Key],
    distances: &'a mut [Distance],
    found: &'a mut usize,
    base_keys: &'a Keys,
}

impl HitRow<'_> {
    /// Writes the query's hits best first, pads the rest with `Key::MISSING` and infinity, and counts
    /// them. `eval` scores only the counted prefix, so backends drop their engine's misses rather
    /// than yield sentinels; more hits than slots is an error.
    pub fn write(self, hits: impl IntoIterator<Item = Result<(Key, Distance), String>>) -> Result<(), String> {
        let mut found = 0;
        for hit in hits {
            let (key, distance) = hit?;
            if found == self.keys.len() {
                return Err(format!("more than the {found} hits requested"));
            }
            self.keys[found] = key;
            self.distances[found] = distance;
            found += 1;
        }
        self.keys[found..].fill(Key::MISSING);
        self.distances[found..].fill(Distance::INFINITY);
        *self.found = found;
        Ok(())
    }

    /// Writes hits that an engine reports as base rows, as engines without wide ids store them,
    /// naming each by its key.
    pub fn write_rows(self, hits: impl IntoIterator<Item = Result<(usize, Distance), String>>) -> Result<(), String> {
        let base_keys = self.base_keys;
        self.write(hits.into_iter().map(|hit| {
            hit.and_then(|(row, distance)| {
                if row < base_keys.rows() {
                    Ok((base_keys.get(row), distance))
                } else {
                    Err(format!("a hit names row {row}, past the base's end"))
                }
            })
        }))
    }
}

// #region Utilities

/// Reinterpret a `&[T]` as the raw bytes that back it, without copying.
///
/// # Safety
///
/// `T` must be "plain old data": no padding, no destructor, no invalid bit patterns.
pub unsafe fn pod_slice_as_bytes<T: Copy>(slice: &[T]) -> &[u8] {
    std::slice::from_raw_parts(slice.as_ptr().cast::<u8>(), std::mem::size_of_val(slice))
}

/// A number spelled with thousand separators, `1,234,567`, written straight into the formatter.
pub struct Thousands(pub u64);

impl fmt::Display for Thousands {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut digits = [0u8; 20];
        let mut remaining = self.0;
        let mut count = 0;
        loop {
            digits[count] = b'0' + (remaining % 10) as u8;
            remaining /= 10;
            count += 1;
            if remaining == 0 {
                break;
            }
        }
        for index in (0..count).rev() {
            formatter.write_char(char::from(digits[index]))?;
            if index > 0 && index.is_multiple_of(3) {
                formatter.write_char(',')?;
            }
        }
        Ok(())
    }
}

/// Splits `range` into runs of `step`, the last one possibly shorter.
pub fn row_ranges(range: Range<usize>, step: usize) -> impl Iterator<Item = Range<usize>> {
    let end = range.end;
    range.step_by(step).map(move |start| start..(start + step).min(end))
}

// #region Common CLI args

/// Dense inputs, shared by every vector backend.
#[derive(clap::Args, Debug)]
pub struct CommonArgs {
    /// Base vectors to index (.fbin, .u8bin, .i8bin, .b1bin)
    #[arg(long, env = "RETRIEVAL_BASE_VECTORS")]
    pub base_vectors: PathBuf,

    /// Query vectors to search with
    #[arg(long, env = "RETRIEVAL_QUERY_VECTORS")]
    pub query_vectors: PathBuf,

    /// Ground-truth neighbor rows for those queries: `.ibin`, or `.u64bin` past 2^32 base rows;
    /// or one target row per query in `.txt`
    #[arg(long, env = "RETRIEVAL_QUERY_NEIGHBORS")]
    pub query_neighbors: PathBuf,

    /// Cap the number of base vectors used (for calibration on a slice of a larger file).
    /// Queries and ground truth are unaffected; only the add/permutation range shrinks.
    #[arg(long, env = "RETRIEVAL_MAX_BASE_VECTORS", value_parser = parse_count_flag)]
    pub max_base_vectors: Option<usize>,

    /// Matryoshka-style embedding-dimension truncations to evaluate
    /// (comma-separated). Empty → use the file's native dimensions. Each value must
    /// be ≤ the native dimensions; for `.b1bin` files each must be a multiple of 8.
    #[arg(long, env = "RETRIEVAL_DIMS", value_delimiter = ',', value_parser = parse_count_flag)]
    pub dims: Vec<usize>,

    #[command(flatten)]
    pub run: RunArgs,
}

/// Text inputs: UTF-8 files of one entry per line, as StringKilla's bases and tasks hold them.
#[derive(clap::Args, Debug)]
pub struct TextArgs {
    /// Base entries to index, one per line, the line number being the row
    #[arg(long, env = "RETRIEVAL_BASE_TEXTS")]
    pub base_texts: PathBuf,

    /// Query entries to search with, one per line
    #[arg(long, env = "RETRIEVAL_QUERY_TEXTS")]
    pub query_texts: PathBuf,

    /// Each query's target base row, one zero-based row per line
    #[arg(long, env = "RETRIEVAL_QUERY_TARGETS")]
    pub query_targets: PathBuf,

    /// Cap the number of base entries indexed, for calibration on a slice of a larger file
    #[arg(long, env = "RETRIEVAL_MAX_BASE_ENTRIES", value_parser = parse_count_flag)]
    pub max_base_entries: Option<usize>,

    #[command(flatten)]
    pub run: RunArgs,
}

/// Run flags every binary shares, whatever its rows.
#[derive(clap::Args, Debug)]
pub struct RunArgs {
    /// Keys for the base rows, one per row: `.i32bin`, `.u64bin` or `.uuidbin`
    #[arg(long, env = "RETRIEVAL_BASE_KEYS")]
    pub base_keys: Option<PathBuf>,

    /// Keys generated when `--base-keys` is unset: u32, u64, uuid-v4 or uuid-v7
    #[arg(
        long,
        env = "RETRIEVAL_KEY_TYPE",
        value_enum,
        default_value = "u32",
        conflicts_with = "base_keys"
    )]
    pub key_type: KeyType,

    /// Neighbors to request per query — the k every metric is taken at.
    /// Defaults to the ground-truth file's width and may not exceed it; under targets,
    /// defaults to 10 with no bound.
    #[arg(long, env = "RETRIEVAL_TOP_K", value_parser = parse_count_flag)]
    pub top_k: Option<usize>,

    /// Order in which base rows are inserted: `shuffled` by `--seed`, or the file's `original` order
    #[arg(long, env = "RETRIEVAL_INSERTION_ORDER", value_enum, default_value = "shuffled")]
    pub insertion_order: InsertionOrder,

    /// Seed of the shuffled insertion order, or `random` to draw one; 42 when unset
    #[arg(long, env = "RETRIEVAL_SEED", value_parser = parse_seed_flag)]
    pub seed: Option<Seed>,

    /// Number of measurement steps (dataset is split into this many equal parts)
    #[arg(long, env = "RETRIEVAL_STEPS", default_value_t = 10, value_parser = parse_count_flag)]
    pub steps: usize,

    /// Entries per add() call for each of the backend's worker threads
    #[arg(long, env = "RETRIEVAL_INSERTS_PER_CORE", default_value_t = 2048, value_parser = parse_count_flag)]
    pub inserts_per_core: usize,

    /// Queries per search() call for each of the backend's worker threads
    #[arg(long, env = "RETRIEVAL_QUERIES_PER_CORE", default_value_t = 2048, value_parser = parse_count_flag)]
    pub queries_per_core: usize,

    /// Output directory for JSON result files
    #[arg(long, env = "RETRIEVAL_OUTPUT")]
    pub output: Option<PathBuf>,

    /// Memory the engine may use, page cache included, like 16GB; at most this process's cgroup
    /// limit, which is the default
    #[arg(long, env = "RETRIEVAL_MEMORY_LIMIT", value_parser = parse_size_flag)]
    pub memory_limit: Option<u64>,

    /// Record hardware counters per phase; requires the perf-counters feature on Linux
    #[arg(long, env = "RETRIEVAL_PERF_COUNTERS", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new())]
    pub perf_counters: bool,

    /// Replay indexed entries as their own queries and report the fraction that
    /// retrieve themselves within the top-`--self-search-top-k` ("self-recall").
    /// Runs once, after the last insertion, and needs no ground truth — an entry
    /// in the index is its own nearest neighbor. Because it can sweep the whole
    /// base rather than a short query file, it is also the only phase that
    /// measures throughput under sustained load.
    #[arg(long, env = "RETRIEVAL_SELF_SEARCH", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new())]
    pub self_search: bool,

    /// Neighbors requested per query during the self-recall sweep (the `k` in
    /// self-recall@k). No effect without `--self-search`.
    #[arg(long, env = "RETRIEVAL_SELF_SEARCH_TOP_K", default_value_t = 10, value_parser = parse_count_flag)]
    pub self_search_top_k: usize,

    /// How many base entries the self-recall sweep replays. A bare integer ≥ 1
    /// is an absolute count; a value with a decimal point (≤ 1.0) is a fraction
    /// of the base, so `1.0` means all and `1` means a single entry. Omitted →
    /// all. A full sweep is the point on GPU backends, but 100M round trips
    /// through a server-style backend is not a measurement anyone waits for.
    ///
    /// The subset is the *leading* N rows of the base file, not a random draw —
    /// contiguity is what keeps the query slices zero-copy, so the throughput
    /// figure stays clean. On a base whose row order carries structure (sorted,
    /// clustered, or concatenated shards) a partial sweep is therefore not a
    /// representative sample; only the default full sweep is unbiased.
    #[arg(long, env = "RETRIEVAL_SELF_SEARCH_SAMPLE", value_name = "N|FRACTION", value_parser = parse_self_search_sample, requires = "self_search")]
    pub self_search_sample: Option<SelfSearchSample>,
}

/// The files and flags one kind of corpus is read with, so one loader serves dense and text runs.
pub trait Inputs {
    type Corpus: Corpus;
    fn run(&self) -> &RunArgs;
    /// The base, query and truth files, in that order.
    fn paths(&self) -> [&Path; 3];
    /// The cap on the base rows indexed.
    fn max_base_entries(&self) -> Option<usize>;

    /// Refuses views the loaded rows cannot serve and lists the ones they will.
    fn check_views(&self, _base: &Self::Corpus, _queries: &Self::Corpus) -> Result<(), DatasetError> {
        Ok(())
    }
}

impl Inputs for CommonArgs {
    type Corpus = Dataset;

    fn run(&self) -> &RunArgs {
        &self.run
    }

    fn paths(&self) -> [&Path; 3] {
        [
            self.base_vectors.as_path(),
            self.query_vectors.as_path(),
            self.query_neighbors.as_path(),
        ]
    }

    fn max_base_entries(&self) -> Option<usize> {
        self.max_base_vectors
    }

    fn check_views(&self, base: &Dataset, queries: &Dataset) -> Result<(), DatasetError> {
        for &truncated in &self.dims {
            base.check_dimensions(truncated)?;
            queries.check_dimensions(truncated)?;
        }
        eprintln!("- Dims: {}", spell_list(self.dimensions_sweep(base.dimensions())));
        Ok(())
    }
}

impl Inputs for TextArgs {
    type Corpus = TextCorpus;

    fn run(&self) -> &RunArgs {
        &self.run
    }

    fn paths(&self) -> [&Path; 3] {
        [
            self.base_texts.as_path(),
            self.query_texts.as_path(),
            self.query_targets.as_path(),
        ]
    }

    fn max_base_entries(&self) -> Option<usize> {
        self.max_base_entries
    }
}

/// Edits a query term may differ by and still match, as Elasticsearch spells it: `0`, `1`, `2`, or
/// `AUTO` for none up to 2 characters, one up to 5, and two beyond.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fuzziness {
    Edits(u8),
    Auto,
}

impl Fuzziness {
    /// Edits allowed for a term of `characters`.
    pub fn edits(self, characters: usize) -> u8 {
        match (self, characters) {
            (Self::Edits(edits), _) => edits,
            (Self::Auto, 0..=2) => 0,
            (Self::Auto, 3..=5) => 1,
            (Self::Auto, _) => 2,
        }
    }
}

impl std::str::FromStr for Fuzziness {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text {
            "0" | "1" | "2" => Ok(Self::Edits(text.as_bytes()[0] - b'0')),
            _ if text.eq_ignore_ascii_case("auto") => Ok(Self::Auto),
            _ => Err("expected 0, 1, 2 or AUTO".into()),
        }
    }
}

impl fmt::Display for Fuzziness {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Edits(edits) => write!(formatter, "{edits}"),
            Self::Auto => formatter.write_str("AUTO"),
        }
    }
}

/// Order in which base rows are inserted.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum InsertionOrder {
    Shuffled,
    Original,
}

impl fmt::Display for InsertionOrder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        spell_value(self, formatter)
    }
}

/// Seed of the shuffled insertion order when `--seed` is unset.
const DEFAULT_SEED: Seed = Seed(42);

impl CommonArgs {
    /// Resolve `--dims` into a sweep list. Empty CLI input expands to a
    /// single-element list at the file's native dimensions, so binaries can iterate
    /// uniformly without special-casing the no-truncation path.
    pub fn dimensions_sweep(&self, native: usize) -> Vec<usize, System> {
        let mut dimensions = Vec::new_in(System);
        dimensions.extend_from_slice(if self.dims.is_empty() {
            std::slice::from_ref(&native)
        } else {
            &self.dims
        });
        dimensions
    }
}

// #region BenchState

/// The `--self-search` pass: neighbors per replayed entry, and how many entries replay.
#[derive(Clone, Copy)]
pub struct SelfSearch {
    pub top_k: usize,
    pub sample: Option<SelfSearchSample>,
}

impl SelfSearch {
    /// Entries to replay out of the `entries_indexed` an index holds.
    fn queries(self, entries_indexed: usize) -> usize {
        resolve_self_search_sample(self.sample, entries_indexed)
    }
}

/// Pre-loaded benchmark state. Call `BenchState::load()` once, then `run()` per configuration.
pub struct BenchState<C: Corpus> {
    base: C,
    /// Effective base-entry count; equals `base.rows()` unless a cap shrank it.
    total_entries: usize,
    keys: Keys,
    queries: C,
    truth: Truth,
    permutation: Permutation,
    /// Resolved `--top-k`: the search width and the k every metric uses.
    top_k: usize,
    steps: usize,
    inserts_per_core: usize,
    queries_per_core: usize,
    self_search: Option<SelfSearch>,
    perf_counters: bool,
    output_dir: Option<PathBuf>,
    dataset_info: DatasetInfo,
    search_buffers: SearchBuffers,
    /// The rows of a shuffled batch, resolved once per batch.
    row_scratch: Vec<usize, System>,
    /// Rows copied for a batch that cannot borrow the mapped file: a gather, a truncation or a
    /// shard boundary.
    scratch: C::Scratch,
}

impl<C: Corpus> BenchState<C> {
    pub fn load<I: Inputs<Corpus = C>>(inputs: &I) -> Result<Self, Box<dyn std::error::Error>> {
        let args = inputs.run();
        let [base_path, query_path, truth_path] = inputs.paths();
        let seed = match (args.insertion_order, args.seed) {
            (InsertionOrder::Shuffled, seed) => Some(seed.unwrap_or(DEFAULT_SEED)),
            (InsertionOrder::Original, None) => None,
            (InsertionOrder::Original, Some(_)) => {
                return Err("--seed applies only to --insertion-order shuffled".into())
            }
        };

        if let Some(directory) = &args.output {
            if directory
                .extension()
                .is_some_and(|extension| extension == "json" || extension == "jsonl")
            {
                return Err(format!(
                    "--output should be a directory, not a file: {}. \
                     Each config produces its own JSON file inside this directory.",
                    directory.display()
                )
                .into());
            }
            std::fs::create_dir_all(directory)?;
        }

        eprintln!("Loading base: {}", base_path.display());
        let base = C::load(base_path)?;
        let file_entries = base.rows();
        let total_entries = inputs
            .max_base_entries()
            .map_or(file_entries, |cap| cap.min(file_entries));
        if total_entries < file_entries {
            eprintln!(
                "  {} entries in file, capping to {}",
                Thousands(file_entries as u64),
                Thousands(total_entries as u64),
            );
            eprintln!(
                "  note: ground truth still refers to all {} rows, so raw recall is bounded by \
                 ~{:.0}% — don't compare capped runs against uncapped ones.",
                Thousands(file_entries as u64),
                100.0 * total_entries as f64 / file_entries as f64,
            );
        } else {
            eprintln!(
                "  {} entries, {} bytes",
                Thousands(total_entries as u64),
                Thousands(base.bytes())
            );
        }

        let keys = match &args.base_keys {
            Some(path) => {
                eprintln!("Loading keys: {}", path.display());
                let keys = Keys::load(path)?;
                eprintln!("  {} keys", Thousands(keys.rows() as u64));
                // Ground truth names rows of the whole file, even past the cap.
                if keys.rows() < file_entries {
                    return Err(format!(
                        "--base-keys has {} keys but the base has {} entries; \
                         one key per entry is required",
                        Thousands(keys.rows() as u64),
                        Thousands(file_entries as u64),
                    )
                    .into());
                }
                keys
            }
            None if args.key_type == KeyType::U32 && file_entries as u64 > u64::from(u32::MAX) + 1 => {
                return Err(format!("--key-type u32 names at most 2^32 rows, the base has {file_entries}").into());
            }
            None => Keys::Generated {
                count: file_entries,
                key_type: args.key_type,
            },
        };

        eprintln!("Loading queries: {}", query_path.display());
        let queries = C::load(query_path)?;
        let query_count = queries.rows();
        eprintln!("  {} queries", Thousands(query_count as u64));

        eprintln!("Loading truth: {}", truth_path.display());
        let truth = Truth::load(truth_path)?;
        let neighbors_per_query = match &truth {
            Truth::Neighbors(_) if base.vectors().is_none() => {
                return Err(format!(
                    "{}: exact neighbors need vectors to rescore; pass targets as .txt",
                    truth_path.display()
                )
                .into());
            }
            Truth::Neighbors(neighbors) => {
                eprintln!(
                    "  {} queries, {} neighbors each",
                    Thousands(neighbors.queries() as u64),
                    neighbors.neighbors_per_query(),
                );
                Some(neighbors.neighbors_per_query())
            }
            Truth::Targets(targets) => {
                eprintln!("  {} queries, one target each", Thousands(targets.rows().len() as u64));
                None
            }
        };
        truth
            .check(query_count, file_entries)
            .map_err(|error| format!("{}: {error}", truth_path.display()))?;
        let top_k = truth.top_k(args.top_k)?;

        let self_search = args.self_search.then_some(SelfSearch {
            top_k: args.self_search_top_k,
            sample: args.self_search_sample,
        });

        match &args.base_keys {
            Some(path) => eprintln!("- Keys: {}", path.display()),
            None => eprintln!("- Key type: {}", args.key_type),
        }
        eprintln!("- Insertion order: {}", args.insertion_order);
        if let Some(seed) = seed {
            eprintln!("- Seed: {seed}");
        }
        eprintln!("- Top k: {top_k}");
        inputs.check_views(&base, &queries)?;
        eprintln!("- Steps: {}", args.steps);
        eprintln!("- Inserts per core: {}", args.inserts_per_core);
        eprintln!("- Queries per core: {}", args.queries_per_core);
        match inputs.max_base_entries() {
            Some(count) => eprintln!("- Max base entries: {count}"),
            None => eprintln!("- Max base entries: all"),
        }
        match self_search {
            Some(SelfSearch { top_k, sample }) => {
                eprintln!("- Self-search top k: {top_k}");
                match sample {
                    Some(sample) => eprintln!("- Self-search sample: {sample}"),
                    None => eprintln!("- Self-search sample: all"),
                }
            }
            None => eprintln!("- Self-search: false"),
        }
        match &args.output {
            Some(path) => eprintln!("- Output: {}", path.display()),
            None => eprintln!("- Output: none"),
        }
        eprintln!("- Perf counters: {}", args.perf_counters);
        let budget = output::Budget::resolve(args.memory_limit)?.install();
        eprintln!("- Cores: {} ({})", budget.cores, budget.cpus);
        eprintln!("- Memory limit: {}", spell_size(budget.memory_limit));

        let permutation = match seed {
            None => Permutation::Identity,
            Some(seed) => Permutation::Shuffled(Shuffle::new(total_entries, seed)),
        };

        let dataset_info = DatasetInfo {
            base_path: base_path.display().to_string(),
            query_path: query_path.display().to_string(),
            truth_path: truth_path.display().to_string(),
            entries_count: file_entries,
            base_bytes: base.bytes(),
            queries_count: query_count,
            dimensions: base.vectors().map(Dataset::dimensions),
            neighbors_per_query,
            truth: truth.name(),
        };

        Ok(Self {
            total_entries,
            permutation,
            top_k,
            steps: args.steps,
            inserts_per_core: args.inserts_per_core,
            queries_per_core: args.queries_per_core,
            self_search,
            perf_counters: args.perf_counters,
            output_dir: args.output.clone(),
            dataset_info,
            search_buffers: SearchBuffers::new(query_count, top_k)?,
            row_scratch: Vec::new_in(System),
            scratch: Default::default(),
            base,
            keys,
            queries,
            truth,
        })
    }
}

impl BenchState<Dataset> {
    /// Native dimensions of the base dataset; truncation is per config.
    pub fn dimensions(&self) -> usize {
        self.base.dimensions()
    }
}

// #region Benchmark loop

type BenchResult<T> = Result<T, Box<dyn std::error::Error>>;

/// Items per second, zero before any time has passed.
fn per_second(count: usize, elapsed: f64) -> u64 {
    if elapsed > 0.0 {
        (count as f64 / elapsed) as u64
    } else {
        0
    }
}

/// A progress bar over `total` items, drawn by `template`.
fn progress_bar(total: usize, template: &str) -> ProgressBar {
    let progress = ProgressBar::new(total as u64);
    progress.set_style(
        ProgressStyle::default_bar()
            .template(template)
            .expect("a valid progress template")
            .progress_chars("##-"),
    );
    progress
}

/// Arm perf counters before a phase; on failure disarm the whole capture so
/// later phases don't try again. Single place for the error-logging wording.
fn start_counter_capture(perf: &mut Option<perf_counters::PerfCounters>, phase: &str) {
    if let Some(counters) = perf.as_mut() {
        if let Err(error) = counters.reset_and_enable() {
            eprintln!("  perf counters: reset/enable before {phase} failed ({error})");
            *perf = None;
        }
    }
}

/// Read a sample at the end of a phase; same disarm-on-error policy.
fn stop_and_read_counters(
    perf: &mut Option<perf_counters::PerfCounters>,
    phase: &str,
) -> Option<perf_counters::CounterSample> {
    let counters = perf.as_mut()?;
    match counters.disable_and_read() {
        Ok(sample) => Some(sample),
        Err(error) => {
            eprintln!("  perf counters: read after {phase} failed ({error})");
            *perf = None;
            None
        }
    }
}

/// Inserts the base rows `step` in batches of `inserts_per_call`, at `view`.
fn run_add_phase<C: Corpus>(
    backend: &mut impl Backend<Corpus = C>,
    state: &mut BenchState<C>,
    perf: &mut Option<perf_counters::PerfCounters>,
    step: Range<usize>,
    view: C::View,
    inserts_per_call: usize,
) -> BenchResult<StepAddEntry> {
    let total_entries = state.total_entries;
    let message = |indexed: usize, throughput: u64| {
        format!(
            "{}/{} ({} add/s)",
            Thousands(indexed as u64),
            Thousands(total_entries as u64),
            Thousands(throughput),
        )
    };
    let progress = progress_bar(total_entries, "  add    [{elapsed_precise}] {bar:40.cyan/blue} {msg}");
    progress.set_position(step.start as u64);

    start_counter_capture(perf, "add");
    let started = Instant::now();
    for batch in row_ranges(step.clone(), inserts_per_call) {
        let (rows, entries) = match &state.permutation {
            Permutation::Identity => (
                BatchRows::Range(batch.clone()),
                state.base.slice(batch.start, batch.len(), view, &mut state.scratch),
            ),
            Permutation::Shuffled(shuffle) => {
                state.row_scratch.clear();
                state
                    .row_scratch
                    .extend(batch.clone().map(|position| shuffle.row(position)));
                let rows = state.row_scratch.as_slice();
                let entries = state.base.gather(rows.iter().copied(), view, &mut state.scratch);
                (BatchRows::Shuffled(rows), entries)
            }
        };
        backend.add(BatchKeys::new(&state.keys, rows), entries)?;
        progress.set_position(batch.end as u64);
        progress.set_message(message(
            batch.end,
            per_second(batch.end - step.start, started.elapsed().as_secs_f64()),
        ));
    }
    backend.flush()?;
    let elapsed = started.elapsed().as_secs_f64();
    let counters = stop_and_read_counters(perf, "add");
    let throughput = per_second(step.len(), elapsed);
    progress.set_message(message(step.end, throughput));
    progress.finish();
    Ok(StepAddEntry {
        elapsed,
        throughput,
        counters,
    })
}

/// Answers every query once into the search buffers.
fn search_pass<C: Corpus>(
    backend: &mut impl Backend<Corpus = C>,
    state: &mut BenchState<C>,
    view: C::View,
    queries_per_call: usize,
) -> BenchResult<()> {
    for batch in row_ranges(0..state.queries.rows(), queries_per_call) {
        let queries = state.queries.slice(batch.start, batch.len(), view, &mut state.scratch);
        backend.search(queries, state.search_buffers.hits(batch, &state.keys))?;
    }
    Ok(())
}

/// Runs all queries against an index holding `entries_indexed` entries and scores them: one untimed
/// pass to warm the index, then timed passes until at least `MIN_SEARCH_SECONDS` have passed.
/// `final_step` drops the `~` that marks metrics over a partial index.
fn run_search_phase<C: Corpus>(
    backend: &mut impl Backend<Corpus = C>,
    state: &mut BenchState<C>,
    perf: &mut Option<perf_counters::PerfCounters>,
    view: C::View,
    queries_per_call: usize,
    entries_indexed: usize,
    final_step: bool,
) -> BenchResult<StepSearchEntry> {
    const MIN_SEARCH_SECONDS: f64 = 5.0;
    let query_count = state.queries.rows();
    let count = state.top_k;
    let progress = progress_bar(query_count, "  search [{elapsed_precise}] {bar:40.green/blue} {msg}");
    progress.set_message("warmup");
    search_pass(backend, state, view, queries_per_call)?;

    start_counter_capture(perf, "search");
    let started = Instant::now();
    let mut passes = 0;
    while passes == 0 || started.elapsed().as_secs_f64() < MIN_SEARCH_SECONDS {
        search_pass(backend, state, view, queries_per_call)?;
        passes += 1;
        progress.set_position(query_count as u64);
        progress.set_message(format!(
            "{passes} passes ({} search/s)",
            Thousands(per_second(passes * query_count, started.elapsed().as_secs_f64())),
        ));
    }
    let elapsed = started.elapsed().as_secs_f64();
    let counters = stop_and_read_counters(perf, "search");
    let throughput = per_second(passes * query_count, elapsed);
    let found = state.search_buffers.found();
    let scores = match &state.truth {
        Truth::Targets(targets) => {
            let targets = targets.rows().iter().map(|&row| state.keys.get(row));
            eval::known_item_hits(found, targets, count).mean(query_count)
        }
        Truth::Neighbors(ground_truth) => {
            let label = backend
                .metadata()
                .get("metric")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned();
            let metric =
                eval::Metric::from_label(&label).ok_or_else(|| format!("no exact distance for metric {label:?}"))?;
            let rescored = state.base.vectors().zip(state.queries.vectors());
            let ((base, queries), dimensions) = rescored
                .zip(C::dimensions(view))
                .ok_or("exact neighbors need vectors to rescore")?;
            let truth = eval::ExactTruth {
                ground_truth,
                base,
                queries,
                base_keys: &state.keys,
                metric,
            };
            eval::score(found, &truth, dimensions, count)?
        }
    };

    // Raw metrics, since dividing out coverage would treat the indexed prefix as a uniform sample.
    let coverage = entries_indexed as f64 / state.base.rows().max(1) as f64;
    let approximate = if final_step { "" } else { "~" };
    // Printed rather than left on the bar, which draws nothing when stderr is a log file.
    progress.finish_and_clear();
    let quality = match (scores.intersection_at_k, scores.ndcg_at_k, scores.mrr_at_k) {
        (Some(intersection_at_k), Some(ndcg_at_k), _) => format!(
            "{approximate}1-recall@1={:.4}, {approximate}1-recall@{count}={:.4}, \
             {approximate}{count}-recall@{count}={intersection_at_k:.4}, {approximate}NDCG@{count}={ndcg_at_k:.4}",
            scores.recall_at_1, scores.recall_at_k,
        ),
        (_, _, mrr_at_k) => format!(
            "{approximate}success@1={:.4}, {approximate}success@{count}={:.4}, {approximate}MRR@{count}={:.4}",
            scores.recall_at_1,
            scores.recall_at_k,
            mrr_at_k.unwrap_or_default(),
        ),
    };
    eprintln!(
        "  searched at {} search/s, {quality} ({} entries, {:.0}% coverage)",
        Thousands(throughput),
        Thousands(entries_indexed as u64),
        coverage * 100.0
    );

    Ok(StepSearchEntry {
        queries: query_count,
        passes,
        top_k: count,
        elapsed,
        throughput,
        recall_at_1: scores.recall_at_1,
        recall_at_k: scores.recall_at_k,
        intersection_at_k: scores.intersection_at_k,
        ndcg_at_k: scores.ndcg_at_k,
        mrr_at_k: scores.mrr_at_k,
        counters,
    })
}

/// Replays the leading `query_count` indexed entries as their own queries: an entry in the index
/// is its own nearest neighbor, so identity is the ground truth, and any dataset can run it.
/// Batches are scored and dropped as they land, since a 100M-entry answer matrix would only
/// reduce to three sums.
fn run_self_search_phase<C: Corpus>(
    backend: &mut impl Backend<Corpus = C>,
    state: &mut BenchState<C>,
    perf: &mut Option<perf_counters::PerfCounters>,
    count: usize,
    query_count: usize,
    view: C::View,
    queries_per_call: usize,
) -> BenchResult<StepSearchEntry> {
    let batch_size = queries_per_call.min(query_count);
    let mut buffers = SearchBuffers::new(batch_size, count)?;
    let progress = progress_bar(query_count, "  self   [{elapsed_precise}] {bar:40.magenta/blue} {msg}");

    start_counter_capture(perf, "self-recall");
    let started = Instant::now();
    let mut hits = eval::KnownItemHits::default();
    for batch in row_ranges(0..query_count, batch_size) {
        let expected = batch.clone().map(|row| state.keys.get(row));
        let queries = state.base.slice(batch.start, batch.len(), view, &mut state.scratch);
        backend.search(queries, buffers.hits(0..batch.len(), &state.keys))?;
        hits = hits + eval::known_item_hits(buffers.found().take(batch.len()), expected, count);
        progress.set_position(batch.end as u64);
        progress.set_message(format!(
            "{}/{} ({} search/s)",
            Thousands(batch.end as u64),
            Thousands(query_count as u64),
            Thousands(per_second(batch.end, started.elapsed().as_secs_f64())),
        ));
    }
    let elapsed = started.elapsed().as_secs_f64();
    let counters = stop_and_read_counters(perf, "self-recall");
    let throughput = per_second(query_count, elapsed);
    let scores = hits.mean(query_count);
    progress.finish_and_clear();
    eprintln!(
        "  self-searched at {} search/s, recall@1={:.4}, recall@{count}={:.4}, MRR@{count}={:.4} ({} queries)",
        Thousands(throughput),
        scores.recall_at_1,
        scores.recall_at_k,
        scores.mrr_at_k.unwrap_or(0.0),
        Thousands(query_count as u64),
    );

    Ok(StepSearchEntry {
        queries: query_count,
        passes: 1,
        top_k: count,
        elapsed,
        throughput,
        recall_at_1: scores.recall_at_1,
        recall_at_k: scores.recall_at_k,
        intersection_at_k: scores.intersection_at_k,
        ndcg_at_k: scores.ndcg_at_k,
        mrr_at_k: scores.mrr_at_k,
        counters,
    })
}

/// Parsed form of `--self-search-sample`. A value containing a decimal point is
/// a fraction of the base (`≤ 1.0`); a bare integer is an absolute vector count.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SelfSearchSample {
    Fraction(f64),
    Absolute(usize),
}

impl fmt::Display for SelfSearchSample {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fraction(fraction) => write!(formatter, "{fraction:?}"),
            Self::Absolute(count) => write!(formatter, "{count}"),
        }
    }
}

/// Parse a `--self-search-sample` spec. The base size isn't known at CLI-parse
/// time, so resolution to a concrete count is deferred to `resolve_self_search_sample`.
fn parse_self_search_sample(spec: &str) -> Result<SelfSearchSample, String> {
    let sample = if spec.contains('.') {
        let digits = spec.bytes().all(|byte| byte.is_ascii_digit() || byte == b'.');
        digits
            .then(|| spec.parse().ok())
            .flatten()
            .filter(|fraction| *fraction > 0.0 && *fraction <= 1.0)
            .map(SelfSearchSample::Fraction)
    } else {
        parse_count(spec).map(SelfSearchSample::Absolute)
    };
    sample.ok_or_else(|| "expected a positive count like 1000 or a fraction like 0.1".into())
}

/// Resolve a sample spec against the number of indexed entries. `None` → all.
/// A fraction rounds to at least one entry; an absolute count is clamped to
/// what is actually in the index.
fn resolve_self_search_sample(sample: Option<SelfSearchSample>, entries_indexed: usize) -> usize {
    match sample {
        None => entries_indexed,
        Some(SelfSearchSample::Fraction(fraction)) => ((entries_indexed as f64 * fraction).round() as usize).max(1),
        Some(SelfSearchSample::Absolute(count)) => count.min(entries_indexed),
    }
}

/// Assemble a `ConfigReport` and write it to `<output_dir>/<backend>-<hash>.json`.
/// No-op when the state has no output directory configured.
fn save_report<C: Corpus>(
    state: &BenchState<C>,
    mut metadata: HashMap<String, Value>,
    view: C::View,
    started_unix_seconds: u64,
    steps: Vec<StepEntry>,
) -> BenchResult<()> {
    const SCHEMA_VERSION: u32 = 2;
    let Some(directory) = &state.output_dir else {
        return Ok(());
    };

    // Every harness flag that changes the measurement belongs in the hash; backends add only their own knobs.
    if let Some(dimensions) = C::dimensions(view) {
        metadata.insert("dimensions".into(), Value::from(dimensions));
    }
    metadata.insert("entries_count".into(), Value::from(state.total_entries));
    metadata.insert("seed".into(), Value::from(state.permutation.seed().map(|seed| seed.0)));
    if let Keys::Generated { key_type, .. } = state.keys {
        if key_type != KeyType::U32 {
            metadata.insert("key_type".into(), Value::from(key_type.to_string()));
        }
    }
    metadata.insert("steps".into(), Value::from(state.steps));
    metadata.insert("inserts_per_core".into(), Value::from(state.inserts_per_core));
    metadata.insert("queries_per_core".into(), Value::from(state.queries_per_core));
    metadata.insert("top_k".into(), Value::from(state.top_k));
    if let Some(entry) = steps.last().and_then(|step| step.self_search.as_ref()) {
        metadata.insert("self_search_top_k".into(), Value::from(entry.top_k));
        metadata.insert("self_search_sample".into(), Value::from(entry.queries));
    }
    let backend_name = metadata.get("backend").and_then(Value::as_str).unwrap_or("unknown");
    let path = directory.join(format!("{backend_name}-{}.json", config_hash(&metadata)));

    let report = ConfigReport {
        schema_version: SCHEMA_VERSION,
        started_unix_seconds,
        machine: output::collect_machine_info(),
        dataset: state.dataset_info.clone(),
        config: metadata,
        steps,
    };

    write_report(&path, &report)?;
    eprintln!("  → {}", path.display());
    Ok(())
}

/// Runs one configuration and writes its report. A fresh index fills step by step and is
/// measured after each, saved before the last step's search when `index_file` asks; a loaded
/// index is measured once.
pub fn run<C: Corpus>(
    backend: &mut impl Backend<Corpus = C>,
    state: &mut BenchState<C>,
    description: &str,
    view: C::View,
    index_file: IndexFile<'_>,
) -> BenchResult<()> {
    let started_unix_seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let metadata = backend.metadata();
    let cores = metadata.get("threads").and_then(Value::as_u64).unwrap_or(1) as usize;
    let inserts_per_call = state.inserts_per_core * cores;
    let queries_per_call = state.queries_per_core * cores;
    // Self-search replays base rows in query-sized batches, so both corpora size for either batch.
    let rows_per_call = inserts_per_call.max(queries_per_call);
    state.base.reserve(&mut state.scratch, rows_per_call, view);
    state.queries.reserve(&mut state.scratch, rows_per_call, view);
    state.row_scratch.reserve(inserts_per_call);
    eprintln!("\n── {description} ──");

    // Without CAP_PERFMON or on another OS, steps run without counters and their fields stay empty.
    let mut perf = state
        .perf_counters
        .then(|| {
            perf_counters::PerfCounters::new()
                .inspect(|_| eprintln!("  perf counters: system-wide per-CPU capture enabled"))
                .inspect_err(|error| eprintln!("  perf counters: unavailable ({error}); running without"))
                .ok()
        })
        .flatten();

    let mut steps = Vec::with_capacity(state.steps);
    match index_file {
        IndexFile::Load(_) => {
            let indexed = backend.indexed_count().unwrap_or(state.total_entries);
            if indexed != state.total_entries {
                eprintln!(
                    "  loaded index holds {} entries, dataset implies {} — using the index's count",
                    Thousands(indexed as u64),
                    Thousands(state.total_entries as u64),
                );
            }
            let search = run_search_phase(backend, state, &mut perf, view, queries_per_call, indexed, true)?;
            steps.push(StepEntry {
                entries_indexed: indexed,
                memory_bytes: backend.memory_bytes() as u64,
                process_anon_bytes: process_anon_bytes(),
                add: None,
                refine: None,
                ground_truth_search: Some(search),
                self_search: None,
            });
        }
        IndexFile::None | IndexFile::Save(_) => {
            let total_entries = state.total_entries;
            let mut add_seconds = 0.0;
            for step in row_ranges(0..total_entries, total_entries.div_ceil(state.steps)) {
                let add = run_add_phase(backend, state, &mut perf, step.clone(), view, inserts_per_call)?;
                add_seconds += add.elapsed;
                eprintln!(
                    "  added {}/{} entries in {:.1} s at {} entries/s, {:.1} s of adds so far",
                    Thousands(step.end as u64),
                    Thousands(total_entries as u64),
                    add.elapsed,
                    Thousands(add.throughput),
                    add_seconds,
                );
                let final_step = step.end == total_entries;
                let refine = if final_step { backend.finish()? } else { None };
                // Saved before searching, so a failure while searching or scoring keeps the build.
                if let (true, IndexFile::Save(path)) = (final_step, index_file) {
                    backend.save(path)?;
                    eprintln!("  saved index → {path}");
                }
                let search = run_search_phase(backend, state, &mut perf, view, queries_per_call, step.end, final_step)?;
                steps.push(StepEntry {
                    entries_indexed: step.end,
                    memory_bytes: backend.memory_bytes() as u64,
                    process_anon_bytes: process_anon_bytes(),
                    add: Some(add),
                    refine,
                    ground_truth_search: Some(search),
                    self_search: None,
                });
            }
        }
    }

    // Runs once against the finished index, so it lands on the last step.
    let entries_indexed = steps.last().map_or(0, |step| step.entries_indexed);
    if let Some(self_search) = state.self_search {
        let queries = self_search.queries(entries_indexed);
        if queries > 0 {
            let entry = run_self_search_phase(
                backend,
                state,
                &mut perf,
                self_search.top_k,
                queries,
                view,
                queries_per_call,
            )?;
            if let Some(last) = steps.last_mut() {
                last.self_search = Some(entry);
            }
        }
    }

    let peak_memory = steps.iter().map(|step| step.memory_bytes).max().unwrap_or(0);
    save_report(state, metadata, view, started_unix_seconds, steps)?;
    eprintln!("  peak memory: {:.2} GB", peak_memory as f64 / (1u64 << 30) as f64);
    eprintln!();
    Ok(())
}

// #region Sweep driver

/// How one config in a sweep ended up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigOutcome {
    /// Backend constructed and the benchmark ran to the end.
    Ran,
    /// Backend construction failed — config is invalid for this engine (e.g. non-default ef on
    /// FAISS binary HNSW).
    Skipped,
    /// Backend ran but the benchmark loop or the save errored mid-flight.
    Failed,
}

impl ConfigOutcome {
    fn skipped(description: &str, error: impl fmt::Display) -> Self {
        eprintln!("\n── {description} — skipped ──\n  {error}");
        Self::Skipped
    }

    fn failed(description: &str, error: impl fmt::Display) -> Self {
        eprintln!("\n── {description} — failed ──\n  {error}");
        Self::Failed
    }
}

/// Runs one opened config, measuring a loaded index once and saving a fresh one when
/// `index_file` asks; failures land in the outcome rather than ending the sweep.
pub fn run_config<C: Corpus>(
    description: &str,
    index_file: IndexFile<'_>,
    backend: Result<impl Backend<Corpus = C>, impl fmt::Display>,
    state: &mut BenchState<C>,
    view: C::View,
) -> ConfigOutcome {
    let mut backend = match backend {
        Ok(backend) => backend,
        Err(error) => return ConfigOutcome::skipped(description, error),
    };
    match run(&mut backend, state, description, view, index_file) {
        Ok(()) => ConfigOutcome::Ran,
        Err(error) => ConfigOutcome::failed(description, error),
    }
}

/// Aggregate outcome counter for a sweep — record per config, report once at the end.
#[derive(Debug, Default, Clone, Copy)]
pub struct SweepSummary {
    pub ran: usize,
    pub skipped: usize,
    pub failed: usize,
}

impl SweepSummary {
    pub fn record(&mut self, outcome: ConfigOutcome) {
        match outcome {
            ConfigOutcome::Ran => self.ran += 1,
            ConfigOutcome::Skipped => self.skipped += 1,
            ConfigOutcome::Failed => self.failed += 1,
        }
    }

    /// Prints the tally and exits with status 1 when a config failed or none ran; servers must
    /// stop their containers first, since the exit skips destructors.
    pub fn finish(self) {
        eprintln!(
            "Benchmark complete: {} config(s) ran, {} skipped, {} failed.",
            self.ran, self.skipped, self.failed
        );
        if self.failed != 0 || self.ran == 0 {
            std::process::exit(1);
        }
    }
}

/// Print a CLI validation error and exit with status 1. Backend binaries call this from their `main()`
/// when `--metric`/`--data-type`/etc. parsing fails — at that point we haven't started the sweep yet, so an
/// early exit is the right shape (vs. a per-config skip).
pub fn bail(message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(1);
}

/// Parses the command line; a bad value prints `--name="value" does not parse, expected ...` and exits with 1.
pub fn parse_cli<T: clap::Parser>() -> T {
    use clap::error::{ContextKind, ContextValue, ErrorKind};

    let error = match T::try_parse() {
        Ok(cli) => return cli,
        Err(error) if !error.use_stderr() => error.exit(),
        Err(error) => error,
    };
    let context = |kind| match error.get(kind) {
        Some(ContextValue::String(text)) => text.as_str(),
        _ => "",
    };
    let expected = match (
        error.kind(),
        std::error::Error::source(&error),
        error.get(ContextKind::ValidValue),
    ) {
        (ErrorKind::ValueValidation, Some(expected), _) => expected.to_string(),
        (ErrorKind::InvalidValue, _, Some(ContextValue::Strings(values))) if !values.is_empty() => {
            format!("expected one of {}", values.join(", "))
        }
        (ErrorKind::InvalidValue, _, _) => "expected a non-empty value".into(),
        _ => {
            eprint!("{error}");
            std::process::exit(1)
        }
    };
    let flag = context(ContextKind::InvalidArg).split(' ').next().unwrap_or_default();
    eprintln!(
        "{flag}=\"{}\" does not parse, {expected}",
        context(ContextKind::InvalidValue)
    );
    std::process::exit(1)
}

/// Parses a 32-bit unsigned integer, or `random` as 32 bits from the OS entropy source.
pub fn parse_seed(text: &str) -> Option<Seed> {
    if text == "random" {
        return Some(Seed(std::hash::RandomState::new().build_hasher().finish() as u32));
    }
    let digits = !text.is_empty() && text.bytes().all(|byte| byte.is_ascii_digit());
    digits.then(|| text.parse().ok().map(Seed)).flatten()
}

/// Parses a thread count like `8`, or `0` as `all_cores`.
pub fn parse_threads(text: &str, all_cores: NonZeroUsize) -> Option<Threads> {
    match text {
        "0" => Some(Threads(all_cores)),
        _ => parse_count(text).and_then(NonZeroUsize::new).map(Threads),
    }
}

/// Parses a count like `10000` or `10M`: `K`, `M`, `G`, `T` are powers of 1000 in any case, and zero is `None`.
pub fn parse_count(text: &str) -> Option<usize> {
    let split = text.find(|c: char| !c.is_ascii_digit()).unwrap_or(text.len());
    let exponent = ["", "k", "m", "g", "t"]
        .iter()
        .position(|unit| text[split..].eq_ignore_ascii_case(unit))?;
    let count = text[..split]
        .parse::<usize>()
        .ok()?
        .checked_mul(1000usize.pow(exponent as u32))?;
    (count != 0).then_some(count)
}

fn parse_digits(text: &str) -> Option<u64> {
    let digits = !text.is_empty() && text.bytes().all(|byte| byte.is_ascii_digit());
    digits.then(|| text.parse().ok()).flatten().filter(|&count| count != 0)
}

/// Parses a duration like `200ms` or `10s`; a bare number, a fraction or zero is `None`.
pub fn parse_duration(text: &str) -> Option<Duration> {
    match text.strip_suffix("ms") {
        Some(count) => parse_digits(count).map(Duration::from_millis),
        None => parse_digits(text.strip_suffix('s')?).map(Duration::from_secs),
    }
}

/// Parses a size like `4096`, `64KB` or `16GB`, in powers of 1024; zero is `None`.
pub fn parse_size(text: &str) -> Option<u64> {
    let split = text.find(|c: char| !c.is_ascii_digit()).unwrap_or(text.len());
    let level = ["", "kb", "mb", "gb", "tb"]
        .iter()
        .position(|unit| text[split..].eq_ignore_ascii_case(unit))?;
    let bytes = text[..split].parse::<u64>().ok()?.checked_mul(1 << (10 * level))?;
    (bytes != 0).then_some(bytes)
}

/// Spells a size the way `parse_size` reads it: `16GB`, `1000`.
pub fn spell_size(bytes: u64) -> String {
    let (mut count, mut unit) = (bytes, "");
    for larger in ["KB", "MB", "GB", "TB"] {
        if count == 0 || count % 1024 != 0 {
            break;
        }
        count /= 1024;
        unit = larger;
    }
    format!("{count}{unit}")
}

/// Spells a duration the way `parse_duration` reads it: `1s`, `1500ms`.
pub fn spell_duration(duration: Duration) -> String {
    let milliseconds = duration.as_millis();
    match milliseconds % 1000 {
        0 => format!("{}s", milliseconds / 1000),
        _ => format!("{milliseconds}ms"),
    }
}

/// A 32-bit run seed, an integer or drawn from the OS for `random`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Seed(pub u32);

impl From<Seed> for u64 {
    fn from(seed: Seed) -> u64 {
        u64::from(seed.0)
    }
}

impl fmt::Display for Seed {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

/// A thread count; `0` in the variable resolves to every core when read, so it is never zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Threads(pub NonZeroUsize);

impl Threads {
    pub const ONE: Threads = Threads(NonZeroUsize::MIN);

    /// Every core in this process's affinity mask, at least one.
    pub fn all_cores() -> Self {
        Self(std::thread::available_parallelism().unwrap_or(NonZeroUsize::MIN))
    }
}

/// A ForkUnion pool of `threads` workers over the machine's topology.
pub fn spawn_pool(threads: Threads) -> fu::Result<ThreadPool> {
    Topology::new().and_then(|topology| ThreadPool::spawn(&topology, threads.0.get()))
}

impl fmt::Display for Threads {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

/// SplitMix64's increment, the golden ratio in 64 bits.
const SPLITMIX64_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// SplitMix64's finalizer, a bijection that spreads every input bit over the whole output.
pub fn mix(value: u64) -> u64 {
    let value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// The inverse of [`mix`].
pub fn unmix(value: u64) -> u64 {
    let value = (value ^ (value >> 31) ^ (value >> 62)).wrapping_mul(0x3196_42B2_D24D_8EC3);
    let value = (value ^ (value >> 27) ^ (value >> 54)).wrapping_mul(0x96DE_1B17_3F11_9089);
    value ^ (value >> 30) ^ (value >> 60)
}

/// The key of stream `index` of `name`, two mixes away from `seed`, so neighboring seeds and indices never meet.
pub fn stream_key(seed: Seed, name: &str, index: u64) -> u64 {
    let hashed = name.bytes().fold(0xCBF2_9CE4_8422_2325, |hash: u64, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01B3)
    });
    mix(mix(u64::from(seed) ^ hashed).wrapping_add(index))
}

/// A SplitMix64 stream seeded with a `stream_key`, bit-identical to the C++ `splitmix64_t`.
pub struct SplitMix64 {
    pub state: u64,
}

impl SplitMix64 {
    /// The next 64 random bits.
    pub fn draw(&mut self) -> u64 {
        self.state = self.state.wrapping_add(SPLITMIX64_GAMMA);
        mix(self.state)
    }

    /// A draw below `bound`: the high half of a draw times `bound`, with a bias of at most `bound` over 2^64.
    pub fn below(&mut self, bound: u64) -> u64 {
        ((u128::from(self.draw()) * u128::from(bound)) >> 64) as u64
    }
}

/// A TCP port; `0` is rejected at parse time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Port(pub NonZeroU16);

impl From<Port> for u16 {
    fn from(port: Port) -> u16 {
        port.0.get()
    }
}

impl fmt::Display for Port {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

/// `parse_count` for clap.
pub fn parse_count_flag(text: &str) -> Result<usize, String> {
    parse_count(text).ok_or_else(|| "expected a positive count like 1000 or 10M".into())
}

/// `parse_count` for clap, with `0` passed on for the library to derive.
pub fn parse_count_or_zero_flag(text: &str) -> Result<usize, String> {
    match text {
        "0" => Ok(0),
        _ => parse_count_flag(text),
    }
}

/// `parse_threads` for clap, with `0` resolved to every available core.
pub fn parse_threads_flag(text: &str) -> Result<Threads, String> {
    parse_threads(text, Threads::all_cores().0).ok_or_else(|| "expected a count, 0 for all cores".into())
}

/// `parse_size` for clap.
pub fn parse_size_flag(text: &str) -> Result<u64, String> {
    parse_size(text).ok_or_else(|| "expected a size like 4096, 64KB or 16GB".into())
}

/// `parse_seed` for clap.
pub fn parse_seed_flag(text: &str) -> Result<Seed, String> {
    parse_seed(text).ok_or_else(|| "expected an unsigned integer or random".into())
}

/// `parse_duration` for clap.
pub fn parse_duration_flag(text: &str) -> Result<Duration, String> {
    parse_duration(text).ok_or_else(|| "expected a duration like 200ms or 10s".into())
}

/// Parses a TCP port from 1 to 65535.
pub fn parse_port(text: &str) -> Result<Port, String> {
    parse_digits(text)
        .and_then(|port| u16::try_from(port).ok())
        .and_then(NonZeroU16::new)
        .map(Port)
        .ok_or_else(|| "expected a port from 1 to 65535".into())
}

/// Writes a `clap::ValueEnum` value the way its flag spells it, for `Display` impls.
pub fn spell_value<T: clap::ValueEnum>(value: &T, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    let possible = value.to_possible_value().expect("every variant has a name");
    formatter.write_str(possible.get_name())
}

/// Spells a list the way a comma-separated sweep flag reads it: `16,32`.
pub fn spell_list<T: fmt::Display>(values: impl IntoIterator<Item = T>) -> String {
    let mut spelled = String::new();
    for (index, value) in values.into_iter().enumerate() {
        let separator = if index == 0 { "" } else { "," };
        fmt::Write::write_fmt(&mut spelled, format_args!("{separator}{value}")).expect("a String takes any write");
    }
    spelled
}

/// Sugar for the recurring `result.unwrap_or_else(|e| bail(&format!("{prefix}: {e}")))`
/// pattern that flows up out of `BenchState::load`, `state.check_dimensions(...)`,
/// and the various backend constructors. The error variant is rendered with
/// `Display` so anything that satisfies the trait works without ceremony.
pub trait UnwrapOrBail<T> {
    /// Unwrap the success variant or bail with `<prefix>: <error>`.
    fn unwrap_or_bail(self, prefix: &str) -> T;
}

impl<T, E: std::fmt::Display> UnwrapOrBail<T> for Result<T, E> {
    fn unwrap_or_bail(self, prefix: &str) -> T {
        match self {
            Ok(value) => value,
            Err(error) => bail(&format!("{prefix}: {error}")),
        }
    }
}

// #region Tests

#[cfg(test)]
mod tests {
    use std::alloc::System;

    use super::{
        parse_self_search_sample, resolve_self_search_sample, CastScratch, Distance, Key, KeyType, Keys, SearchBuffers,
        SelfSearchSample, Shuffle, SplitMix64, VectorSlice, Vectors,
    };

    #[test]
    fn hit_rows_pad_count_translate_rows_and_refuse_overflow() {
        let base_keys = Keys::Generated {
            count: 3,
            key_type: KeyType::UuidV4,
        };
        let mut buffers = SearchBuffers::new(3, 3).unwrap();
        {
            let mut hits = buffers.hits(0..3, &base_keys);
            let mut rows = hits.rows();
            let (seven, nine) = (Key::from(7u32), Key::from(9u32));
            rows.next().unwrap().write([Ok((seven, 0.5)), Ok((nine, 1.5))]).unwrap();
            rows.next().unwrap().write_rows([Ok((2, 0.0))]).unwrap();
            let overflow = rows.next().unwrap().write_rows((0..4).map(|row| Ok((row % 3, 0.0))));
            assert!(overflow.is_err());
        }
        let mut rows = buffers.rows();
        let (keys, distances, found) = rows.next().unwrap();
        assert_eq!(keys, [Key::from(7u32), Key::from(9u32), Key::MISSING]);
        assert_eq!(distances, [0.5, 1.5, Distance::INFINITY]);
        assert_eq!(found, 2);
        assert_eq!(rows.next().unwrap().0[0], base_keys.get(2));
    }

    #[test]
    fn sizes_round_trip_in_powers_of_1024() {
        for text in ["4095", "64KB", "1000", "3GB", "1TB"] {
            assert_eq!(super::spell_size(super::parse_size(text).unwrap()), text);
        }
        assert_eq!(super::parse_size("1gb"), Some(1 << 30));
        for text in ["0", "", "GB", "1PB", "-1", "1.5GB"] {
            assert_eq!(super::parse_size(text), None, "{text}");
        }
    }

    #[test]
    fn counts_take_powers_of_1000_and_durations_take_digits() {
        assert_eq!(super::parse_count("10M"), Some(10_000_000));
        assert_eq!(super::parse_count("2k"), Some(2000));
        for text in ["0", "", "M", "0K", "1.5M", "-1", "10MB"] {
            assert_eq!(super::parse_count(text), None, "{text}");
        }
        assert_eq!(super::parse_duration("10s"), Some(super::Duration::from_secs(10)));
        assert_eq!(super::parse_duration("10Ks"), None);
        assert_eq!(super::parse_count_or_zero_flag("0"), Ok(0));
    }

    #[test]
    fn thousands_group_from_the_right() {
        let spelled = |number| super::Thousands(number).to_string();
        assert_eq!(spelled(0), "0");
        assert_eq!(spelled(999), "999");
        assert_eq!(spelled(1000), "1,000");
        assert_eq!(spelled(1_234_567), "1,234,567");
        assert_eq!(spelled(u64::MAX), "18,446,744,073,709,551,615");
    }

    #[test]
    fn shuffles_permute_every_size_without_storing_rows() {
        for count in [1, 2, 3, 5, 1000, 65_537] {
            let shuffle = Shuffle::new(count, super::Seed(42));
            let mut rows: Vec<usize> = (0..count).map(|position| shuffle.row(position)).collect();
            let identity = rows.iter().enumerate().all(|(position, &row)| position == row);
            assert!(count <= 3 || !identity, "count {count} came back in file order");
            rows.sort_unstable();
            assert!(rows.iter().copied().eq(0..count), "count {count}");
        }
    }

    #[test]
    fn generated_uuids_are_unique_versioned_decodable_and_v7_sorts_by_row() {
        let rows = [0, 1, 2, 15, 16, 4095, 4096, 65_535, 65_536, 6_039_411_649];
        for (key_type, version) in [(KeyType::UuidV4, 4), (KeyType::UuidV7, 7)] {
            let keys: Vec<Key> = rows.iter().map(|&row| key_type.encode(row)).collect();
            for (key, &row) in keys.iter().zip(&rows) {
                assert_eq!(key.0 >> 76 & 0xF, version);
                assert_eq!(key.0 >> 62 & 0b11, 0b10);
                assert_eq!(key.to_string().parse::<Key>(), Ok(*key));
                assert_eq!(key_type.decode(*key), Some(row));
            }
            let mut unique = keys.clone();
            unique.sort_unstable();
            unique.dedup();
            assert_eq!(unique.len(), keys.len());
            assert_eq!(key_type == KeyType::UuidV7, unique == keys);
        }
    }

    #[test]
    fn conversions_reuse_allocations_and_pass_matching_types_through() {
        let mut scratch = Vec::new_in(System);
        let signed = Vectors::new(VectorSlice::I8(&[-2, 3]), 2);
        assert_eq!(signed.to_f32_in(&mut scratch).unwrap(), [-2.0, 3.0]);
        let pointer = scratch.as_ptr();
        let unsigned = Vectors::new(VectorSlice::U8(&[7]), 1);
        assert_eq!(unsigned.to_f32_in(&mut scratch).unwrap(), [7.0]);
        assert_eq!(pointer, scratch.as_ptr());
        let source = [2.0, 4.0];
        let floats = Vectors::new(VectorSlice::F32(&source), 2);
        assert_eq!(floats.to_f32_in(&mut scratch).unwrap().as_ptr(), source.as_ptr());
        let mut wire = CastScratch::F32(Vec::new_in(System));
        let pointer = wire
            .encode(&Vectors::new(VectorSlice::I8(&[1, -2, 3]), 3))
            .unwrap()
            .as_ptr();
        assert_eq!(
            wire.encode(&Vectors::new(VectorSlice::I8(&[4, 5]), 2))
                .unwrap()
                .as_ptr(),
            pointer
        );
        let bytes = [9u8, 8];
        let mut bytes_wire = CastScratch::U8(Vec::new_in(System));
        let unsigned = Vectors::new(VectorSlice::U8(&bytes), 2);
        let passed = bytes_wire.encode(&unsigned).unwrap();
        assert_eq!(passed.as_ptr(), bytes.as_ptr());
    }

    #[test]
    fn self_recall_sample_none_is_all() {
        assert_eq!(resolve_self_search_sample(None, 1000), 1000);
    }

    #[test]
    fn self_recall_sample_absolute_is_clamped() {
        let resolve = |spec| resolve_self_search_sample(Some(parse_self_search_sample(spec).unwrap()), 1000);
        assert_eq!(resolve("250"), 250);
        assert_eq!(resolve("5000"), 1000);
    }

    #[test]
    fn self_recall_sample_fraction_rounds_and_floors_at_one() {
        let resolve = |spec| resolve_self_search_sample(Some(parse_self_search_sample(spec).unwrap()), 1000);
        assert_eq!(resolve("0.1"), 100);
        assert_eq!(resolve("1.0"), 1000);
        // A fraction so small it would round to zero still yields one vector.
        assert_eq!(resolve("0.0001"), 1);
    }

    #[test]
    fn self_recall_sample_one_vs_one_point_zero_disambiguate() {
        // `1` is an absolute count; `1.0` is the whole base.
        assert_eq!(parse_self_search_sample("1").unwrap(), SelfSearchSample::Absolute(1));
        assert_eq!(
            parse_self_search_sample("1.0").unwrap(),
            SelfSearchSample::Fraction(1.0)
        );
        assert_eq!(SelfSearchSample::Fraction(1.0).to_string(), "1.0");
    }

    #[test]
    fn self_recall_sample_rejects_zero_signs_and_out_of_range() {
        for spec in ["0", "0.0", "1.5", "abc", "+5", "+0.5", "-0.5", "1e-3"] {
            assert!(parse_self_search_sample(spec).is_err(), "{spec}");
        }
    }

    #[test]
    fn splitmix64_matches_the_shared_test_vector() {
        assert_eq!(SplitMix64 { state: 42 }.draw(), 0xbdd7_3226_2feb_6e95);
    }
}
