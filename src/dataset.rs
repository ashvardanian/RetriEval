//! Memory-mapped vector datasets and text corpora, key mappings, and ground truth.

use std::{
    alloc::System,
    fs::{File, OpenOptions},
    io::{self, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

use memmap2::Mmap;

use crate::{error::DatasetError, mix, stream_key, Key, KeyType, Seed, Thousands, VectorSlice, Vectors};

/// Rows a backend reads, loaded once and handed to it in batches, borrowed in place where they can be.
pub trait Corpus: Sized {
    /// One batch, borrowed from the corpus or from its scratch.
    type Rows<'a>
    where
        Self: 'a;
    /// How every row of one config is exposed, such as a truncated width.
    type View: Copy;
    /// Where batches that cannot borrow the corpus are copied.
    type Scratch: Default;

    fn load(path: &Path) -> io::Result<Self>;
    fn rows(&self) -> usize;
    /// Bytes the rows occupy in their files.
    fn bytes(&self) -> u64;
    /// Grows `scratch` to hold a batch of `rows` at `view`.
    fn reserve(&self, scratch: &mut Self::Scratch, rows: usize, view: Self::View);
    /// The rows `start..start + count`, borrowed when they are contiguous in place.
    fn slice<'a>(
        &'a self,
        start: usize,
        count: usize,
        view: Self::View,
        scratch: &'a mut Self::Scratch,
    ) -> Self::Rows<'a>;
    /// The listed rows, copied into `scratch`.
    fn gather<'a>(
        &'a self,
        rows: impl IntoIterator<Item = usize>,
        view: Self::View,
        scratch: &'a mut Self::Scratch,
    ) -> Self::Rows<'a>;

    /// The vectors exact-neighbor truth is rescored over; `None` for rows without a distance.
    fn vectors(&self) -> Option<&Dataset> {
        None
    }

    /// The width `view` exposes, reported as `dimensions`; `None` for rows without one.
    fn dimensions(_view: Self::View) -> Option<usize> {
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScalarFormat {
    F32,
    U8,
    I8,
    B1x8,
}

/// One segment of a natural-sort key. Pure-text and integer segments alternate.
/// Variant order matters: text segments sort before numeric ones, so e.g.
/// `base.fbin` < `base.shard_0.fbin` regardless of what we'd otherwise compare.
#[derive(Debug, Eq, PartialEq, Ord, PartialOrd)]
enum NaturalSegment<'a> {
    Text(&'a [u8]),
    Number(u64),
}

/// Decompose a filename into alternating text/number runs for natural sort.
/// `shard_2.fbin` ranks before `shard_10.fbin` because the numeric runs
/// compare as 2 < 10 instead of `'1','0'` lex-comparing against `'2'`.
fn natural_key(name: &str) -> Vec<NaturalSegment<'_>, System> {
    let bytes = name.as_bytes();
    let mut segments = Vec::new_in(System);
    let mut position = 0;
    while position < bytes.len() {
        let start = position;
        let digits = bytes[start].is_ascii_digit();
        while position < bytes.len() && bytes[position].is_ascii_digit() == digits {
            position += 1;
        }
        let run = &bytes[start..position];
        // Saturating so an absurdly long digit run clamps instead of panicking; zero-padding is
        // not preserved, so `shard_007` and `shard_7` compare equal.
        segments.push(match digits {
            true => NaturalSegment::Number(run.iter().fold(0u64, |number, &digit| {
                number.saturating_mul(10).saturating_add(u64::from(digit - b'0'))
            })),
            false => NaturalSegment::Text(run),
        });
    }
    segments
}

/// Expand `pattern` into a sorted list of shard paths. A pattern with no
/// glob metacharacters (`*`, `?`, `[`) is returned as-is — single-file
/// callers stay on the zero-glob fast path. Globbed expansions are sorted
/// with `natural_key` so `shard_2.fbin` comes before `shard_10.fbin`.
fn expand_glob(pattern: &str) -> io::Result<Vec<PathBuf, System>> {
    if !pattern.contains(['*', '?', '[']) {
        let mut paths = Vec::with_capacity_in(1, System);
        paths.push(PathBuf::from(pattern));
        return Ok(paths);
    }
    let entries = glob::glob(pattern).map_err(|error| io::Error::other(format!("glob({pattern}): {error}")))?;
    let mut paths = Vec::new_in(System);
    for entry in entries {
        paths.push(entry.map_err(|error| io::Error::other(format!("glob({pattern}): {error}")))?);
    }
    if paths.is_empty() {
        return Err(io::Error::other(format!("glob({pattern}) matched no files")));
    }
    paths.sort_by(|left, right| natural_key(&left.to_string_lossy()).cmp(&natural_key(&right.to_string_lossy())));
    Ok(paths)
}

/// Bytes occupied by the per-shard header (rows u32 LE + secondary u32 LE).
/// Same shape across `.fbin` / `.u8bin` / `.i8bin` / `.b1bin` / `.ibin` /
/// `.i32bin` files: the secondary u32 is `dimensions` for vector files and
/// `neighbors_per_query` for ground-truth.
const HEADER_BYTES: usize = 8;

/// Multi-shard mmap substrate shared by `Dataset`, `Keys::Mapped`, and
/// `GroundTruth`. Each shard's body is row-major at byte offset
/// `HEADER_BYTES + within_shard_index * stride_bytes`. A single-file dataset
/// holds exactly one mmap; glob expansions hold many.
pub struct Shards {
    mmaps: Vec<Mmap, System>,
    /// Cumulative per-shard row-count prefix sum. `offsets[i]` rows live in
    /// shards `0..i`. Length = `mmaps.len() + 1`, last element = total rows.
    offsets: Vec<usize, System>,
    /// Byte stride between consecutive rows within each shard.
    stride_bytes: usize,
}

impl Shards {
    /// Build from already-validated mmaps and their per-shard row counts.
    fn new(mmaps: Vec<Mmap, System>, shard_rows: &[usize], stride_bytes: usize) -> Self {
        debug_assert_eq!(mmaps.len(), shard_rows.len());
        let mut offsets = Vec::with_capacity_in(shard_rows.len() + 1, System);
        offsets.push(0);
        offsets.extend(shard_rows.iter().scan(0, |total, &rows| {
            *total += rows;
            Some(*total)
        }));
        Self {
            mmaps,
            offsets,
            stride_bytes,
        }
    }

    fn rows(&self) -> usize {
        *self.offsets.last().unwrap_or(&0)
    }

    /// Locate the shard owning a global `row`. Returns
    /// `(shard_index, within_shard_index)`.
    fn locate(&self, row: usize) -> (usize, usize) {
        // `partition_point` counts the offsets at or below `row`, one past the shard holding it.
        let shard_index = self.offsets.partition_point(|&offset| offset <= row) - 1;
        (shard_index, row - self.offsets[shard_index])
    }

    /// Borrow the leading `len` bytes of one row's storage. Used for full
    /// rows (`len == stride_bytes`) and truncated prefixes (`len < stride_bytes`).
    fn row_bytes(&self, row: usize, length: usize) -> &[u8] {
        let (shard, within) = self.locate(row);
        let offset = HEADER_BYTES + within * self.stride_bytes;
        &self.mmaps[shard][offset..offset + length]
    }

    /// Borrow a contiguous range of `count` rows iff it fits within a single
    /// shard — i.e. callers can take the zero-copy fast path. Returns `None`
    /// for cross-shard ranges (caller must fall back to copying via
    /// `row_bytes`) and for empty ranges.
    fn zero_copy_range(&self, start: usize, count: usize) -> Option<&[u8]> {
        if count == 0 {
            return None;
        }
        let (shard, within) = self.locate(start);
        let shard_len = self.offsets[shard + 1] - self.offsets[shard];
        if within + count > shard_len {
            return None;
        }
        let offset = HEADER_BYTES + within * self.stride_bytes;
        Some(&self.mmaps[shard][offset..offset + count * self.stride_bytes])
    }
}

/// Maps every shard `pattern` names and checks each header against its body: the row count, then
/// a width that every shard must share, `dimensions` for vectors and `neighbors` for ground truth.
/// `stride_bytes` turns that width into the bytes of one row. Returns the shards and the width.
fn load_shards(
    pattern: &Path,
    kind: &'static str,
    stride_bytes: impl Fn(usize) -> usize,
) -> io::Result<(Shards, usize)> {
    let shard_paths = expand_glob(&pattern.to_string_lossy())?;
    let mut mmaps = Vec::with_capacity_in(shard_paths.len(), System);
    let mut shard_rows = Vec::with_capacity_in(shard_paths.len(), System);
    let mut shared_width = None;
    for shard_path in &shard_paths {
        let path = shard_path.to_path_buf();
        // SAFETY: the inputs are read-only files nothing else writes during a run.
        let mmap = unsafe { Mmap::map(&File::open(shard_path)?)? };
        if mmap.len() < HEADER_BYTES {
            let got = mmap.len() as u64;
            return Err(io::Error::other(DatasetError::HeaderTooSmall { path, kind, got }));
        }
        let rows = u32::from_le_bytes(mmap[0..4].try_into().unwrap()) as usize;
        let width = u32::from_le_bytes(mmap[4..8].try_into().unwrap()) as usize;
        if let Some(expected) = shared_width.filter(|&expected| expected != width) {
            return Err(io::Error::other(DatasetError::ShardMismatch {
                path,
                expected_width: expected as u32,
                got_width: width as u32,
            }));
        }
        shared_width = Some(width);
        let expected = HEADER_BYTES + rows * stride_bytes(width);
        if mmap.len() < expected {
            return Err(io::Error::other(DatasetError::BodyTooSmall {
                path,
                expected: expected as u64,
                rows: rows as u32,
                dimensions: width as u32,
                got: mmap.len() as u64,
            }));
        }
        mmaps.push(mmap);
        shard_rows.push(rows);
    }
    let width = shared_width.expect("a pattern that matched nothing already failed");
    Ok((Shards::new(mmaps, &shard_rows, stride_bytes(width)), width))
}

/// A memory-mapped binary vector dataset.
///
/// File format (all little-endian):
/// - bytes 0..4: number of rows (u32)
/// - bytes 4..8: number of dimensions (u32)
/// - bytes 8..: row-major vector data
///
/// For `.b1bin` files, dimensions is the number of bits per vector.
/// Each vector occupies `ceil(dimensions / 8)` bytes.
pub struct Dataset {
    shards: Shards,
    /// Native (file-encoded) per-vector dimensionality. Immutable. Truncation
    /// is a per-call concern of `slice` / `gather` — pass the desired dimension
    /// through their `dimensions` argument.
    dimensions: usize,
    format: ScalarFormat,
    /// The path or glob the shards were mapped from, in the order `expand_glob` lists them.
    pattern: PathBuf,
}

/// Bytes occupied by a vector of the given format and dimensionality.
fn bytes_for(format: ScalarFormat, dimensions: usize) -> usize {
    match format {
        ScalarFormat::F32 => dimensions * 4,
        ScalarFormat::U8 | ScalarFormat::I8 => dimensions,
        ScalarFormat::B1x8 => dimensions.div_ceil(8),
    }
}

/// Ground-truth neighbor rows from one or more `.ibin` or `.u64bin` files (single path or glob expansion).
pub struct GroundTruth {
    shards: Shards,
    neighbors_per_query: usize,
    /// Bytes per neighbor row id: 4 in `.ibin`, 8 in `.u64bin`.
    row_bytes: usize,
}

impl Dataset {
    /// Load a dataset from `path`, a single file or a glob of shards that share their native
    /// dimensions; the extension names the scalar format.
    pub fn load(path: &Path) -> io::Result<Self> {
        let extension = path.extension().and_then(|extension| extension.to_str()).unwrap_or("");
        let format = match extension {
            "fbin" => ScalarFormat::F32,
            "u8bin" => ScalarFormat::U8,
            "i8bin" => ScalarFormat::I8,
            "b1bin" => ScalarFormat::B1x8,
            _ => {
                return Err(io::Error::other(DatasetError::UnsupportedExtension {
                    path: path.to_path_buf(),
                    extension: extension.to_string(),
                }))
            }
        };
        let (shards, dimensions) = load_shards(path, "dataset", |dimensions| bytes_for(format, dimensions))?;
        Ok(Self {
            shards,
            dimensions,
            format,
            pattern: path.to_path_buf(),
        })
    }

    /// Moves the distinct `rows` out of the files in place, about one row write each: a listed row
    /// before the last `rows.len()` takes the bytes of an unlisted row from among them, then the
    /// last shard drops that tail, which must lie within it. Returns the dataset reopened.
    pub fn without_rows(self, rows: &[usize]) -> io::Result<Self> {
        let total = self.rows();
        let last_shard = self.shards.mmaps.len() - 1;
        let first_of_last = self.shards.offsets[last_shard];
        let kept = total
            .checked_sub(rows.len())
            .filter(|&kept| kept >= first_of_last)
            .ok_or_else(|| io::Error::other(format!("{} rows to remove reach past the last shard", rows.len())))?;
        let mut removed = Vec::with_capacity_in(rows.len(), System);
        removed.extend_from_slice(rows);
        removed.sort_unstable();
        let mut fillers = (kept..total).filter(|row| removed.binary_search(row).is_err());
        let paths = expand_glob(&self.pattern.to_string_lossy())?;
        let mut writers: Vec<Option<File>, System> = Vec::with_capacity_in(paths.len(), System);
        writers.resize_with(paths.len(), || None);
        let stride = self.shards.stride_bytes;
        for &row in removed.iter().take_while(|&&row| row < kept) {
            let filler = fillers
                .next()
                .expect("an unlisted tail row for every listed row before the tail");
            let (shard, within) = self.shards.locate(row);
            let writer = match &mut writers[shard] {
                Some(writer) => writer,
                slot => slot.insert(OpenOptions::new().write(true).open(&paths[shard])?),
            };
            writer.seek(SeekFrom::Start((HEADER_BYTES + within * stride) as u64))?;
            writer.write_all(self.shards.row_bytes(filler, stride))?;
        }
        for writer in writers.iter().flatten() {
            writer.sync_all()?;
        }
        let last_rows = kept - first_of_last;
        let mut last = OpenOptions::new().write(true).open(&paths[last_shard])?;
        last.write_all(&(last_rows as u32).to_le_bytes())?;
        let pattern = self.pattern.clone();
        // Unmapped before the shrink, so no mapping outlives the bytes it covers.
        drop(self);
        last.set_len((HEADER_BYTES + last_rows * stride) as u64)?;
        last.sync_all()?;
        Self::load(&pattern)
    }

    pub fn rows(&self) -> usize {
        self.shards.rows()
    }

    /// Native (file-encoded) per-vector dimensionality. Truncation is
    /// per-call — pass the desired `dimensions` through `slice` / `gather`.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Bytes per row at the file's native dimensions — equivalently, the on-disk
    /// stride between consecutive rows. Useful for sizing scratch buffers
    /// to an upper bound that fits any truncation.
    pub fn vector_bytes(&self) -> usize {
        self.shards.stride_bytes
    }

    /// Validate that `dimensions` is a legal exposure for this dataset:
    /// `0 < dimensions ≤ native dimensions`, and for `b1bin` a multiple of 8.
    pub fn check_dimensions(&self, dimensions: usize) -> Result<(), DatasetError> {
        let byte_aligned = self.format != ScalarFormat::B1x8 || dimensions.is_multiple_of(8);
        if dimensions == 0 || dimensions > self.dimensions || !byte_aligned {
            return Err(DatasetError::InvalidTruncation {
                requested: dimensions,
                native: self.dimensions,
            });
        }
        Ok(())
    }

    /// Shapes rows already packed at `dimensions` into `Vectors`.
    fn wrap_bytes<'a>(&self, bytes: &'a [u8], dimensions: usize) -> Vectors<'a> {
        let values = match self.format {
            // SAFETY: the mapped rows start past an 8-byte header on a page boundary, and scratch
            // rows are copies of them, so `f32` alignment holds; `i8` has no alignment.
            ScalarFormat::F32 => VectorSlice::F32(unsafe {
                std::slice::from_raw_parts(bytes.as_ptr().cast(), bytes.len() / std::mem::size_of::<f32>())
            }),
            ScalarFormat::U8 => VectorSlice::U8(bytes),
            ScalarFormat::I8 => {
                VectorSlice::I8(unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast(), bytes.len()) })
            }
            ScalarFormat::B1x8 => VectorSlice::B1x8(bytes),
        };
        Vectors::new(values, dimensions)
    }

    /// Borrow the rows `start..start + count` at `dimensions`. Zero-copy when the
    /// dimensions are native and the range sits in one shard; otherwise the rows
    /// are copied through `scratch`, which must hold `count` rows at `dimensions`.
    pub fn slice<'a>(&'a self, start: usize, count: usize, dimensions: usize, scratch: &'a mut [u8]) -> Vectors<'a> {
        let count = count.min(self.shards.rows() - start);
        if dimensions == self.dimensions {
            if let Some(bytes) = self.shards.zero_copy_range(start, count) {
                return self.wrap_bytes(bytes, dimensions);
            }
        }
        self.gather(start..start + count, dimensions, scratch)
    }

    /// Each mapped shard's rows as zero-copy `Vectors` at native dimensions.
    pub fn shards(&self) -> impl Iterator<Item = Vectors<'_>> {
        let stride_bytes = self.shards.stride_bytes;
        self.shards
            .mmaps
            .iter()
            .zip(self.shards.offsets.windows(2))
            .map(move |(mmap, rows)| {
                let body = &mmap[HEADER_BYTES..HEADER_BYTES + (rows[1] - rows[0]) * stride_bytes];
                self.wrap_bytes(body, self.dimensions)
            })
    }

    /// Copy the rows at `indices` into `scratch` at `dimensions`, keeping each
    /// row's leading bytes when `dimensions` is below native.
    pub fn gather<'a>(
        &self,
        indices: impl IntoIterator<Item = usize>,
        dimensions: usize,
        scratch: &'a mut [u8],
    ) -> Vectors<'a> {
        let bytes_per_row = bytes_for(self.format, dimensions);
        let mut rows = 0;
        for (destination, source_index) in scratch.chunks_exact_mut(bytes_per_row).zip(indices) {
            destination.copy_from_slice(self.shards.row_bytes(source_index, bytes_per_row));
            rows += 1;
        }
        self.wrap_bytes(&scratch[..rows * bytes_per_row], dimensions)
    }
}

impl Corpus for Dataset {
    type Rows<'a> = Vectors<'a>;
    /// The truncated dimensions.
    type View = usize;
    type Scratch = Vec<u8>;

    fn load(path: &Path) -> io::Result<Self> {
        Dataset::load(path)
    }

    fn rows(&self) -> usize {
        Dataset::rows(self)
    }

    fn bytes(&self) -> u64 {
        (self.rows() * self.vector_bytes()) as u64
    }

    fn reserve(&self, scratch: &mut Self::Scratch, rows: usize, _view: usize) {
        let bytes = rows * self.vector_bytes();
        if scratch.len() < bytes {
            scratch.resize(bytes, 0);
        }
    }

    fn slice<'a>(&'a self, start: usize, count: usize, view: usize, scratch: &'a mut Self::Scratch) -> Vectors<'a> {
        Dataset::slice(self, start, count, view, scratch)
    }

    fn gather<'a>(
        &'a self,
        rows: impl IntoIterator<Item = usize>,
        view: usize,
        scratch: &'a mut Self::Scratch,
    ) -> Vectors<'a> {
        Dataset::gather(self, rows, view, scratch)
    }

    fn vectors(&self) -> Option<&Dataset> {
        Some(self)
    }

    fn dimensions(view: usize) -> Option<usize> {
        Some(view)
    }
}

/// A memory-mapped UTF-8 file of one entry per line, the line number being its row, as StringKilla's
/// bases and queries are laid out.
pub struct TextCorpus {
    mmap: Mmap,
    /// Where each line starts, then the file's end: one more offset than lines.
    starts: Vec<u64, System>,
}

/// A batch of text entries: line bytes and one more offset into them than entries, as USearch's
/// `Tape` lays entries out.
#[derive(Clone, Copy)]
pub struct Texts<'a> {
    bytes: &'a [u8],
    offsets: &'a [u64],
}

impl<'a> Texts<'a> {
    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Entry `row`, without its line ending.
    pub fn get(&self, row: usize) -> &'a str {
        let line = &self.bytes[self.offsets[row] as usize..self.offsets[row + 1] as usize];
        let line = line.strip_suffix(b"\n").unwrap_or(line);
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        // SAFETY: the file was validated as UTF-8 once, and lines split at ASCII bytes.
        unsafe { std::str::from_utf8_unchecked(line) }
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = &'a str> + 'a {
        let texts = *self;
        (0..self.len()).map(move |row| texts.get(row))
    }
}

/// The lines of a shuffled batch, copied end to end.
pub struct TextScratch {
    bytes: Vec<u8, System>,
    offsets: Vec<u64, System>,
}

impl Default for TextScratch {
    fn default() -> Self {
        Self {
            bytes: Vec::new_in(System),
            offsets: Vec::new_in(System),
        }
    }
}

impl Corpus for TextCorpus {
    type Rows<'a> = Texts<'a>;
    type View = ();
    type Scratch = TextScratch;

    fn load(path: &Path) -> io::Result<Self> {
        // SAFETY: the inputs are read-only files nothing else writes during a run.
        let mmap = unsafe { Mmap::map(&File::open(path)?)? };
        std::str::from_utf8(&mmap)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, format!("{}: {error}", path.display())))?;
        let lines = mmap.iter().filter(|&&byte| byte == b'\n').count();
        let mut starts = Vec::with_capacity_in(lines + 2, System);
        starts.push(0);
        starts.extend(
            mmap.iter()
                .enumerate()
                .filter(|&(_, &byte)| byte == b'\n')
                .map(|(position, _)| position as u64 + 1),
        );
        // A last line without a final newline still counts.
        if starts.last() != Some(&(mmap.len() as u64)) {
            starts.push(mmap.len() as u64);
        }
        Ok(Self { mmap, starts })
    }

    fn rows(&self) -> usize {
        self.starts.len() - 1
    }

    fn bytes(&self) -> u64 {
        self.mmap.len() as u64
    }

    fn reserve(&self, scratch: &mut TextScratch, rows: usize, _view: ()) {
        scratch.offsets.clear();
        scratch.offsets.reserve(rows + 1);
    }

    fn slice<'a>(&'a self, start: usize, count: usize, _view: (), _scratch: &'a mut TextScratch) -> Texts<'a> {
        let count = count.min(self.rows() - start);
        Texts {
            bytes: &self.mmap,
            offsets: &self.starts[start..=start + count],
        }
    }

    fn gather<'a>(
        &'a self,
        rows: impl IntoIterator<Item = usize>,
        _view: (),
        scratch: &'a mut TextScratch,
    ) -> Texts<'a> {
        scratch.bytes.clear();
        scratch.offsets.clear();
        scratch.offsets.push(0);
        for row in rows {
            let line = &self.mmap[self.starts[row] as usize..self.starts[row + 1] as usize];
            scratch.bytes.extend_from_slice(line);
            scratch.offsets.push(scratch.bytes.len() as u64);
        }
        Texts {
            bytes: &scratch.bytes,
            offsets: &scratch.offsets,
        }
    }
}

/// The order rows are inserted in: the file's own, which reads rows in place, or a seeded shuffle.
pub enum Permutation {
    Identity,
    Shuffled(Shuffle),
}

impl Permutation {
    /// The base row inserted at `position`.
    pub fn row(&self, position: usize) -> usize {
        match self {
            Self::Identity => position,
            Self::Shuffled(shuffle) => shuffle.row(position),
        }
    }

    /// The seed of a shuffled order.
    pub fn seed(&self) -> Option<Seed> {
        match self {
            Self::Identity => None,
            Self::Shuffled(shuffle) => Some(shuffle.seed),
        }
    }
}

/// Rounds of the shuffle's Feistel network; four make it a strong pseudorandom permutation.
const SHUFFLE_ROUNDS: usize = 4;

/// A seeded bijection on `[0, count)`, computed per position, so a shuffle of any size stores nothing
/// per row. A balanced Feistel network permutes the smallest even-width power of two holding
/// `count`, and positions it sends past the end walk the cycle until they land inside, which takes
/// under four steps on average, since that domain is less than four times `count`.
pub struct Shuffle {
    seed: Seed,
    count: u64,
    half_bits: u32,
    round_keys: [u64; SHUFFLE_ROUNDS],
}

impl Shuffle {
    pub fn new(count: usize, seed: Seed) -> Self {
        let bits = u64::BITS - (count as u64).saturating_sub(1).leading_zeros();
        Self {
            seed,
            count: count as u64,
            half_bits: bits.div_ceil(2).max(1),
            round_keys: std::array::from_fn(|round| stream_key(seed, "permutation", round as u64)),
        }
    }

    pub fn row(&self, position: usize) -> usize {
        debug_assert!(
            (position as u64) < self.count,
            "position {position} past {}",
            self.count
        );
        let mut value = position as u64;
        loop {
            value = self.encrypt(value);
            if value < self.count {
                return value as usize;
            }
        }
    }

    fn encrypt(&self, value: u64) -> u64 {
        let mask = (1 << self.half_bits) - 1;
        let (left, right) = self
            .round_keys
            .iter()
            .fold((value >> self.half_bits, value & mask), |(left, right), &key| {
                (right, left ^ (mix(right ^ key) & mask))
            });
        left << self.half_bits | right
    }
}

/// Vector keys: memory-mapped from `.i32bin`, `.u64bin` or `.uuidbin` shards at the file's width,
/// or generated per row, so no run stores keys wider than its input.
pub enum Keys {
    /// One `key_bytes`-wide key per row after each shard's `rows`, `1` header: little-endian
    /// integers, or UUIDs most significant byte first.
    Mapped {
        shards: Shards,
        key_bytes: usize,
        /// Every row, ordered by its key, so a key finds its row by binary search.
        rows_by_key: Vec<usize, System>,
    },
    Generated {
        count: usize,
        key_type: KeyType,
    },
}

/// The key mapped `shards` of `key_bytes`-wide keys hold for `row`.
fn mapped_key(shards: &Shards, key_bytes: usize, row: usize) -> Key {
    let bytes = shards.row_bytes(row, key_bytes);
    match key_bytes {
        4 => Key::from(u32::from_le_bytes(bytes.try_into().unwrap())),
        8 => Key::from(u64::from_le_bytes(bytes.try_into().unwrap())),
        _ => Key(u128::from_be_bytes(bytes.try_into().unwrap())),
    }
}

impl Keys {
    /// Load keys from a path or glob of shards, as wide as the extension says.
    pub fn load(path: &Path) -> io::Result<Self> {
        let key_bytes = match path.extension().and_then(|extension| extension.to_str()) {
            Some("i32bin") => 4,
            Some("u64bin") => 8,
            Some("uuidbin") => 16,
            _ => {
                return Err(io::Error::other(format!(
                    "{}: keys are .i32bin, .u64bin or .uuidbin",
                    path.display()
                )))
            }
        };
        let (shards, _) = load_shards(path, "keys", |_| key_bytes)?;
        let mut rows_by_key = Vec::with_capacity_in(shards.rows(), System);
        rows_by_key.extend(0..shards.rows());
        rows_by_key.sort_unstable_by_key(|&row| mapped_key(&shards, key_bytes, row));
        Ok(Keys::Mapped {
            shards,
            key_bytes,
            rows_by_key,
        })
    }

    /// The key of base `row`.
    pub fn get(&self, row: usize) -> Key {
        match self {
            Keys::Mapped { shards, key_bytes, .. } => mapped_key(shards, *key_bytes, row),
            Keys::Generated { key_type, .. } => key_type.encode(row),
        }
    }

    /// The base row holding `key`, when one does.
    pub fn row_of(&self, key: Key) -> Option<usize> {
        let row = match self {
            Keys::Mapped { rows_by_key, .. } => {
                rows_by_key[rows_by_key.binary_search_by_key(&key, |&row| self.get(row)).ok()?]
            }
            Keys::Generated { key_type, .. } => key_type.decode(key)?,
        };
        (row < self.rows() && self.get(row) == key).then_some(row)
    }

    pub fn rows(&self) -> usize {
        match self {
            Keys::Mapped { shards, .. } => shards.rows(),
            Keys::Generated { count, .. } => *count,
        }
    }

    /// Bytes per key, for engines that size a key column once: 4, 8 or 16.
    pub fn key_bytes(&self) -> usize {
        match self {
            Keys::Mapped { key_bytes, .. } => *key_bytes,
            Keys::Generated { key_type, .. } => key_type.key_bytes(),
        }
    }
}

impl GroundTruth {
    pub fn load(path: &Path) -> io::Result<Self> {
        let row_bytes = match path.extension().and_then(|extension| extension.to_str()) {
            Some("ibin" | "i32bin") => 4,
            Some("u64bin") => 8,
            _ => {
                return Err(io::Error::other(format!(
                    "{}: ground truth is .ibin, .i32bin or .u64bin",
                    path.display()
                )))
            }
        };
        let (shards, neighbors_per_query) = load_shards(path, "ground_truth", |neighbors| neighbors * row_bytes)?;
        Ok(Self {
            shards,
            neighbors_per_query,
            row_bytes,
        })
    }

    pub fn queries(&self) -> usize {
        self.shards.rows()
    }

    pub fn neighbors_per_query(&self) -> usize {
        self.neighbors_per_query
    }

    /// The base rows of a query's true neighbors, nearest first.
    pub fn neighbors(&self, query_index: usize) -> impl ExactSizeIterator<Item = usize> + '_ {
        let bytes = self
            .shards
            .row_bytes(query_index, self.neighbors_per_query * self.row_bytes);
        bytes.chunks_exact(self.row_bytes).map(|row| match row.len() {
            4 => u32::from_le_bytes(row.try_into().unwrap()) as usize,
            _ => u64::from_le_bytes(row.try_into().unwrap()) as usize,
        })
    }
}

/// Each query's one target base row, from a text file of one zero-based row per line, as
/// StringKilla's tasks and USearch's `--query-targets` hold them.
pub struct Targets {
    rows: Vec<usize, System>,
}

impl Targets {
    pub fn load(path: &Path) -> io::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let mut rows = Vec::new_in(System);
        for (line, row) in text.lines().enumerate() {
            let row = row.parse::<usize>().map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("{}:{}: {row:?} is not a base row", path.display(), line + 1),
                )
            })?;
            rows.push(row);
        }
        Ok(Self { rows })
    }

    /// The target base rows, in query order.
    pub fn rows(&self) -> &[usize] {
        &self.rows
    }
}

/// What a run's searches are judged against, chosen by the truth file's extension.
pub enum Truth {
    /// Each query's nearest base rows from `.ibin`, `.i32bin` or `.u64bin`, rescored by exact distance.
    Neighbors(GroundTruth),
    /// Each query's one target row from `.txt`, credited only where that row comes back.
    Targets(Targets),
}

impl Truth {
    pub fn load(path: &Path) -> io::Result<Self> {
        match path.extension().and_then(|extension| extension.to_str()) {
            Some("txt") => Targets::load(path).map(Self::Targets),
            _ => GroundTruth::load(path).map(Self::Neighbors),
        }
    }

    /// The judge's name in reports: `neighbors` or `targets`.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Neighbors(_) => "neighbors",
            Self::Targets(_) => "targets",
        }
    }

    /// The `--top-k` a run searches and scores at: under targets the request, else USearch's default of 10;
    /// under neighbors the request, else the truth's width, which it may not exceed.
    pub fn top_k(&self, requested: Option<usize>) -> Result<usize, String> {
        const TARGETS_TOP_K: usize = 10;
        match self {
            Self::Targets(_) => Ok(requested.unwrap_or(TARGETS_TOP_K)),
            Self::Neighbors(neighbors) => {
                let width = neighbors.neighbors_per_query();
                let top_k = requested.unwrap_or(width);
                if top_k > width {
                    return Err(format!(
                        "--top-k {top_k} exceeds the ground truth's {width} neighbors per query"
                    ));
                }
                Ok(top_k)
            }
        }
    }

    /// Refuses targets for another number of queries than `queries`, and any row past the base's `base_rows`.
    pub fn check(&self, queries: usize, base_rows: usize) -> Result<(), String> {
        let past_base = match self {
            Self::Neighbors(neighbors) => (0..neighbors.queries()).find_map(|query| {
                neighbors
                    .neighbors(query)
                    .find(|&row| row >= base_rows)
                    .map(|row| (query, row))
            }),
            Self::Targets(targets) => {
                if targets.rows.len() != queries {
                    return Err(format!("{} targets for {queries} queries", targets.rows.len()));
                }
                targets
                    .rows
                    .iter()
                    .position(|&row| row >= base_rows)
                    .map(|query| (query, targets.rows[query]))
            }
        };
        match past_base {
            Some((query, row)) => Err(format!(
                "query {query} names base row {row}, past the {} base rows",
                Thousands(base_rows as u64)
            )),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{io::Write, path::PathBuf};

    use super::{Corpus, TextCorpus, TextScratch, Truth};

    /// A file under the temporary directory holding `contents`, named for the test that wrote it.
    fn written(name: &str, contents: &[u8]) -> PathBuf {
        let path = std::env::temp_dir().join(format!("retrieval-{}-{name}", std::process::id()));
        std::fs::File::create(&path).unwrap().write_all(contents).unwrap();
        path
    }

    #[test]
    fn text_lines_drop_crlf_keep_a_last_line_without_newline_and_an_empty_file_has_none() {
        let corpus = TextCorpus::load(&written("lines.txt", b"alpha\r\n\nbeta\ngamma")).unwrap();
        let mut scratch = Default::default();
        let texts = corpus.slice(0, 4, (), &mut scratch);
        assert_eq!(texts.iter().collect::<Vec<_>>(), ["alpha", "", "beta", "gamma"]);
        assert_eq!(corpus.bytes(), 18);
        let empty = TextCorpus::load(&written("empty.txt", b"")).unwrap();
        assert_eq!(empty.rows(), 0);
        assert!(empty.slice(0, 1, (), &mut scratch).is_empty());
    }

    #[test]
    fn text_slices_borrow_the_mapping_and_gathers_copy_the_same_entries() {
        let corpus = TextCorpus::load(&written("gather.txt", b"one\ntwo\r\nthree\n")).unwrap();
        let (mut borrowed, mut copied) = (TextScratch::default(), TextScratch::default());
        let slice: Vec<&str> = corpus.slice(1, 2, (), &mut borrowed).iter().collect();
        let gather: Vec<&str> = corpus.gather([1, 2], (), &mut copied).iter().collect();
        assert_eq!(slice, gather);
        assert_eq!(slice, ["two", "three"]);
        assert!(borrowed.bytes.is_empty());
        assert_eq!(copied.bytes.as_slice(), b"two\r\nthree\n");
    }

    #[test]
    fn targets_refuse_a_count_mismatch_and_a_row_past_the_base() {
        let truth = Truth::load(&written("targets.txt", b"2\n0\r\n7\n")).unwrap();
        assert_eq!(truth.name(), "targets");
        assert!(truth.check(3, 8).is_ok());
        assert!(truth.check(2, 8).unwrap_err().contains("3 targets for 2 queries"));
        assert!(truth.check(3, 7).unwrap_err().contains("query 2 names base row 7"));
    }

    #[test]
    fn top_k_is_free_under_targets_and_bounded_by_the_neighbors_width() {
        let targets = Truth::load(&written("top-k.txt", b"0\n")).unwrap();
        assert_eq!(targets.top_k(None), Ok(10));
        assert_eq!(targets.top_k(Some(100)), Ok(100));
        let header_and_rows: Vec<u8> = [1u32, 3, 0, 1, 2]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let neighbors = Truth::load(&written("top-k.ibin", &header_and_rows)).unwrap();
        assert_eq!(neighbors.top_k(None), Ok(3));
        assert!(neighbors.top_k(Some(4)).is_err());
    }
}
