//! Memory-mapped vector datasets, key mappings, and ground truth.

use std::{alloc::System, fs::File, io, path::Path};

use memmap2::Mmap;

use crate::{error::DatasetError, mix, stream_key, Key, KeyType, Seed, VectorSlice, Vectors};

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
fn expand_glob(pattern: &str) -> io::Result<Vec<std::path::PathBuf, System>> {
    if !pattern.contains(['*', '?', '[']) {
        let mut paths = Vec::with_capacity_in(1, System);
        paths.push(std::path::PathBuf::from(pattern));
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
        })
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
    },
    Generated {
        count: usize,
        key_type: KeyType,
    },
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
        Ok(Keys::Mapped { shards, key_bytes })
    }

    /// The key of base `row`.
    pub fn get(&self, row: usize) -> Key {
        match self {
            Keys::Mapped { shards, key_bytes } => {
                let bytes = shards.row_bytes(row, *key_bytes);
                match *key_bytes {
                    4 => Key::from(u32::from_le_bytes(bytes.try_into().unwrap())),
                    8 => Key::from(u64::from_le_bytes(bytes.try_into().unwrap())),
                    _ => Key(u128::from_be_bytes(bytes.try_into().unwrap())),
                }
            }
            Keys::Generated { key_type, .. } => key_type.encode(row),
        }
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
