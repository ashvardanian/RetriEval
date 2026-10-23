//! Memory-mapped vector datasets, key mappings, and ground truth.

use std::{alloc::System, fs::File, io, path::Path};

use forkunion::{IntoParallelIterator, ParallelIteratorExt};
use memmap2::Mmap;

use crate::{error::DatasetError, spawn_pool, stream_key, Key, Seed, SplitMix64, Threads, VectorSlice, Vectors};

/// Below this many elements the thread-pool spin-up cost outweighs any
/// parallelism benefit; the shuffle runs serially as one chunk.
/// Measured at ~70 µs pool launch on a 192-thread Xeon 6776P, which
/// is slower than shuffling 65 K elements serially.
const PERMUTATION_SERIAL_THRESHOLD: usize = 65_536;

/// Chunks the parallel shuffle splits into; fixed, so the permutation never depends on the core count.
const PERMUTATION_CHUNKS: usize = 256;

/// In-place Fisher–Yates over chunk `chunk_index`'s own SplitMix64 stream, so
/// the permutation is the same on every platform and `rand` version.
fn fisher_yates_shuffle(slice: &mut [usize], seed: Seed, chunk_index: usize) {
    let mut generator = SplitMix64 {
        state: stream_key(seed, "permutation", chunk_index as u64),
    };
    for swap_target_ceiling in (1..slice.len()).rev() {
        let swap_target = generator.below(swap_target_ceiling as u64 + 1) as usize;
        slice.swap(swap_target_ceiling, swap_target);
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
fn expand_glob(pattern: &str) -> io::Result<Vec<std::path::PathBuf, System>> {
    if !pattern.contains(['*', '?', '[']) {
        let mut paths = Vec::with_capacity_in(1, System);
        paths.push(std::path::PathBuf::from(pattern));
        return Ok(paths);
    }
    let entries = glob::glob(pattern).map_err(|error| io::Error::other(format!("glob({pattern}): {error}")))?;
    let mut paths = Vec::new_in(System);
    paths.extend(entries.filter_map(Result::ok));
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
            let (expected_dims, got_dims) = (expected as u32, width as u32);
            return Err(io::Error::other(DatasetError::ShardMismatch {
                path,
                expected_dims,
                got_dims,
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

/// Ground-truth neighbor indices loaded from one or more `.ibin` files
/// (single path or glob expansion).
pub struct GroundTruth {
    shards: Shards,
    neighbors_per_query: usize,
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

/// The order rows are inserted in: the file's own, which needs no indices and reads rows in place,
/// or a permutation of `[0..n]` shuffled by `seed`.
pub enum Permutation {
    Identity,
    Shuffled { seed: Seed, indices: Vec<usize, System> },
}

impl Permutation {
    /// Create a shuffled permutation of `[0..n]` with a deterministic seed,
    /// parallelized across the ForkUnion thread pool.
    ///
    /// The output is uniformly random *within each chunk* of `ceil(n/256)`
    /// consecutive positions. Across chunks, element ranges remain ordered
    /// (chunk 0 holds some subset of `[0, n/256)`, chunk 1 holds a subset
    /// of `[n/256, 2n/256)`, etc.). This is sufficient for ANN
    /// benchmarking — it eliminates natural orderings (e.g. sorted molecules)
    /// and gives graph construction ~`chunk`-sized random windows to see a
    /// representative sample before moving on.
    pub fn shuffled(n: usize, seed: Seed) -> Self {
        let mut indices = Vec::with_capacity_in(n, System);
        indices.extend(0..n);
        if n < PERMUTATION_SERIAL_THRESHOLD {
            fisher_yates_shuffle(&mut indices, seed, 0);
            return Self::Shuffled { seed, indices };
        }
        let mut chunks = Vec::with_capacity_in(PERMUTATION_CHUNKS, System);
        chunks.extend(indices.chunks_mut(n.div_ceil(PERMUTATION_CHUNKS)));
        let shuffle = |chunk: &mut &mut [usize], chunk_index: usize| fisher_yates_shuffle(chunk, seed, chunk_index);
        let shuffled = match spawn_pool(Threads::all_cores()) {
            Ok(mut pool) => chunks
                .as_mut_slice()
                .into_par_iter()
                .with_pool(&mut pool)
                .for_each(|chunk, chunk_index, _| shuffle(chunk, chunk_index)),
            Err(error) => Err(error),
        };
        if shuffled.is_err() {
            chunks
                .iter_mut()
                .enumerate()
                .for_each(|(chunk_index, chunk)| shuffle(chunk, chunk_index));
        }
        Self::Shuffled { seed, indices }
    }

    /// The seed of a shuffled order.
    pub fn seed(&self) -> Option<Seed> {
        match self {
            Self::Identity => None,
            Self::Shuffled { seed, .. } => Some(*seed),
        }
    }
}

/// Vector keys, either loaded from one or more `.i32bin` files (single path
/// or glob expansion) or generated sequentially.
pub enum Keys {
    /// Memory-mapped keys, one or more shards (file format per shard:
    /// `rows u32`, `dimensions u32` (always 1), then `Key` data).
    Mapped(Shards),
    /// Sequential keys 0..count, generated on the fly.
    Sequential { count: usize },
}

impl Keys {
    /// Load keys from a `.i32bin` path or glob of shards.
    pub fn load(path: &Path) -> io::Result<Self> {
        let (shards, _) = load_shards(path, "keys", |_| std::mem::size_of::<Key>())?;
        Ok(Keys::Mapped(shards))
    }

    /// Get the key at a specific index.
    pub fn get(&self, index: usize) -> Key {
        debug_assert!(
            index < self.rows(),
            "key index {index} out of bounds (rows={})",
            self.rows()
        );
        match self {
            Keys::Mapped(shards) => {
                let bytes = shards.row_bytes(index, std::mem::size_of::<Key>());
                Key::from_le_bytes(bytes.try_into().unwrap())
            }
            Keys::Sequential { .. } => index as Key,
        }
    }

    pub fn rows(&self) -> usize {
        match self {
            Keys::Mapped(shards) => shards.rows(),
            Keys::Sequential { count } => *count,
        }
    }

    /// Borrow a contiguous slice of keys. Zero-copy when the range fits in
    /// a single mapped shard; otherwise (cross-shard or `Sequential`) the
    /// scratch buffer is filled and returned.
    pub fn slice<'a>(&'a self, start: usize, count: usize, scratch: &'a mut [Key]) -> &'a [Key] {
        let count = count.min(self.rows() - start);
        match self {
            Keys::Mapped(shards) => {
                if let Some(bytes) = shards.zero_copy_range(start, count) {
                    // SAFETY: `bytes` spans `count` keys and the page-aligned mapping keeps 4-byte alignment.
                    return unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const Key, count) };
                }
                for (offset, slot) in scratch[..count].iter_mut().enumerate() {
                    let bytes = shards.row_bytes(start + offset, std::mem::size_of::<Key>());
                    *slot = Key::from_le_bytes(bytes.try_into().unwrap());
                }
                &scratch[..count]
            }
            Keys::Sequential { .. } => {
                for (offset, slot) in scratch[..count].iter_mut().enumerate() {
                    *slot = (start + offset) as Key;
                }
                &scratch[..count]
            }
        }
    }
}

impl GroundTruth {
    pub fn load(path: &Path) -> io::Result<Self> {
        let (shards, neighbors_per_query) =
            load_shards(path, "ground_truth", |neighbors| neighbors * std::mem::size_of::<Key>())?;
        Ok(Self {
            shards,
            neighbors_per_query,
        })
    }

    pub fn queries(&self) -> usize {
        self.shards.rows()
    }

    pub fn neighbors_per_query(&self) -> usize {
        self.neighbors_per_query
    }

    /// Get the ground-truth neighbor IDs for a specific query.
    pub fn neighbors(&self, query_index: usize) -> &[Key] {
        let stride_bytes = self.neighbors_per_query * std::mem::size_of::<Key>();
        let bytes = self.shards.row_bytes(query_index, stride_bytes);
        // SAFETY: `bytes` spans `neighbors_per_query` keys and the page-aligned mapping keeps 4-byte alignment.
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const Key, self.neighbors_per_query) }
    }
}
