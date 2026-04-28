use std::fs::File;
use std::path::Path;

use fork_union::{SyncMutPtr, ThreadPool};
use memmap2::Mmap;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{RngCore, SeedableRng};

use crate::error::DatasetError;
use crate::{Key, VectorSlice, Vectors};

/// Below this many elements the thread-pool spin-up cost outweighs any
/// parallelism benefit; the shuffle falls back to the serial `SliceRandom`
/// path. Measured at ~70 µs pool launch on a 192-thread Xeon 6776P, which
/// is slower than shuffling 65 K elements serially.
const PERMUTATION_SERIAL_THRESHOLD: usize = 65_536;

/// Mixes the user's seed with the thread index to give each worker its own
/// deterministic `SmallRng`. Value is the fractional part of the golden ratio
/// times 2^64 — a standard choice (same constant fxhash et al. use) whose
/// avalanche properties won't cluster seeds for adjacent thread indices.
const THREAD_SEED_MIXER: u64 = 0x9E3779B97F4A7C15;

/// In-place Fisher–Yates on a `&mut [usize]`. Pure safe code — the
/// unsafe-slice-reconstruction lives at the caller (see
/// [`Permutation::shuffled`]). Kept out of line so the parallel shuffle
/// closure reads as three lines instead of ten.
fn fisher_yates_shuffle(slice: &mut [usize], rng: &mut SmallRng) {
    for swap_target_ceiling in (1..slice.len()).rev() {
        let swap_target = (rng.next_u64() as usize) % (swap_target_ceiling + 1);
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
fn natural_key(s: &str) -> Vec<NaturalSegment<'_>> {
    let bytes = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_digit() {
            let start = i;
            let mut n: u64 = 0;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                n = n.saturating_mul(10).saturating_add((bytes[i] - b'0') as u64);
                i += 1;
            }
            // saturating_mul guarantees the parse never panics on absurdly long
            // digit runs; the original byte slice doubles as a tiebreaker for
            // leading-zero variants like `shard_007` vs `shard_7`.
            let _ = start;
            out.push(NaturalSegment::Number(n));
        } else {
            let start = i;
            while i < bytes.len() && !bytes[i].is_ascii_digit() {
                i += 1;
            }
            out.push(NaturalSegment::Text(&bytes[start..i]));
        }
    }
    out
}

/// Expand `pattern` into a sorted list of shard paths. A pattern with no
/// glob metacharacters (`*`, `?`, `[`) is returned as-is — single-file
/// callers stay on the zero-glob fast path. Globbed expansions are sorted
/// with `natural_key` so `shard_2.fbin` comes before `shard_10.fbin`.
fn expand_glob(pattern: &str) -> std::io::Result<Vec<std::path::PathBuf>> {
    if !pattern.contains(['*', '?', '[']) {
        return Ok(vec![std::path::PathBuf::from(pattern)]);
    }
    let entries = glob::glob(pattern).map_err(|e| std::io::Error::other(format!("glob({pattern}): {e}")))?;
    let mut paths: Vec<std::path::PathBuf> = entries.filter_map(Result::ok).collect();
    if paths.is_empty() {
        return Err(std::io::Error::other(format!("glob({pattern}) matched no files")));
    }
    paths.sort_by(|a, b| {
        let aa = a.to_string_lossy();
        let bb = b.to_string_lossy();
        natural_key(&aa).cmp(&natural_key(&bb))
    });
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
    mmaps: Vec<Mmap>,
    /// Cumulative per-shard row-count prefix sum. `offsets[i]` rows live in
    /// shards `0..i`. Length = `mmaps.len() + 1`, last element = total rows.
    offsets: Vec<usize>,
    /// Byte stride between consecutive rows within each shard.
    stride_bytes: usize,
}

impl Shards {
    /// Build from already-validated mmaps and their per-shard row counts.
    fn new(mmaps: Vec<Mmap>, shard_rows: &[usize], stride_bytes: usize) -> Self {
        debug_assert_eq!(mmaps.len(), shard_rows.len());
        let mut offsets = Vec::with_capacity(shard_rows.len() + 1);
        let mut acc = 0usize;
        offsets.push(0);
        for &n in shard_rows {
            acc += n;
            offsets.push(acc);
        }
        Self {
            mmaps,
            offsets,
            stride_bytes,
        }
    }

    fn rows(&self) -> usize {
        *self.offsets.last().unwrap_or(&0)
    }

    fn stride_bytes(&self) -> usize {
        self.stride_bytes
    }

    /// Locate the shard owning a global `row`. Returns
    /// `(shard_index, within_shard_index)`.
    fn locate(&self, row: usize) -> (usize, usize) {
        // Largest i where offsets[i] <= row. partition_point returns the
        // count of elements where the predicate is true, which is i + 1.
        let shard_index = self.offsets.partition_point(|&off| off <= row) - 1;
        (shard_index, row - self.offsets[shard_index])
    }

    /// Borrow the leading `len` bytes of one row's storage. Used for full
    /// rows (`len == stride_bytes`) and truncated prefixes (`len < stride_bytes`).
    fn row_bytes(&self, row: usize, len: usize) -> &[u8] {
        let (shard, within) = self.locate(row);
        let off = HEADER_BYTES + within * self.stride_bytes;
        &self.mmaps[shard][off..off + len]
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
        let mmap = &self.mmaps[shard];
        let off = HEADER_BYTES + within * self.stride_bytes;
        let len = count * self.stride_bytes;
        Some(&mmap[off..off + len])
    }
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
    /// Load a dataset from `path`. If the path's string form contains glob
    /// metacharacters (`*`, `?`, `[`), it expands to a sorted list of shards;
    /// otherwise the single file is loaded as before. All shards in a glob
    /// expansion must share the same native dimensions and scalar format, otherwise
    /// loading errors.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let pattern = path.to_string_lossy();
        let shard_paths = expand_glob(&pattern)?;

        let mut mmaps: Vec<Mmap> = Vec::with_capacity(shard_paths.len());
        let mut shard_rows: Vec<usize> = Vec::with_capacity(shard_paths.len());
        let mut format: Option<ScalarFormat> = None;
        let mut native_dimensions: Option<usize> = None;

        for shard_path in &shard_paths {
            let file = File::open(shard_path)?;
            let mmap = unsafe { Mmap::map(&file)? };

            if mmap.len() < HEADER_BYTES {
                return Err(std::io::Error::other(DatasetError::HeaderTooSmall {
                    path: shard_path.to_path_buf(),
                    kind: "dataset",
                    got: mmap.len() as u64,
                }));
            }

            let rows = u32::from_le_bytes(mmap[0..4].try_into().unwrap()) as usize;
            let dimensions = u32::from_le_bytes(mmap[4..8].try_into().unwrap()) as usize;

            let ext = shard_path.extension().and_then(|e| e.to_str()).unwrap_or("");
            let shard_format = match ext {
                "fbin" => ScalarFormat::F32,
                "u8bin" => ScalarFormat::U8,
                "i8bin" => ScalarFormat::I8,
                "b1bin" => ScalarFormat::B1x8,
                _ => {
                    return Err(std::io::Error::other(DatasetError::UnsupportedExtension {
                        path: shard_path.to_path_buf(),
                        extension: ext.to_string(),
                    }))
                }
            };

            // First shard pins the format + dimensions; subsequent shards must match.
            match (format, native_dimensions) {
                (None, None) => {
                    format = Some(shard_format);
                    native_dimensions = Some(dimensions);
                }
                (Some(f), Some(d)) if f == shard_format && d == dimensions => {}
                (Some(_), Some(d)) => {
                    return Err(std::io::Error::other(DatasetError::ShardMismatch {
                        path: shard_path.to_path_buf(),
                        expected_dims: d as u32,
                        got_dims: dimensions as u32,
                    }));
                }
                _ => unreachable!(),
            }

            let stride_bytes = bytes_for(shard_format, dimensions);
            let expected = HEADER_BYTES + rows * stride_bytes;
            if mmap.len() < expected {
                return Err(std::io::Error::other(DatasetError::BodyTooSmall {
                    path: shard_path.to_path_buf(),
                    expected: expected as u64,
                    rows: rows as u32,
                    dimensions: dimensions as u32,
                    got: mmap.len() as u64,
                }));
            }

            mmaps.push(mmap);
            shard_rows.push(rows);
        }

        let format = format.expect("shard_paths is non-empty so format was set");
        let native_dimensions = native_dimensions.expect("shard_paths is non-empty so dimensions was set");
        let stride_bytes = bytes_for(format, native_dimensions);

        Ok(Self {
            shards: Shards::new(mmaps, &shard_rows, stride_bytes),
            dimensions: native_dimensions,
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
        self.shards.stride_bytes()
    }

    /// Validate that `dimensions` is a legal exposure for this dataset:
    /// `0 < dimensions ≤ native dimensions`, and for `b1bin` a multiple of 8.
    pub fn check_dimensions(&self, dimensions: usize) -> Result<(), DatasetError> {
        if dimensions == 0 || dimensions > self.dimensions {
            return Err(DatasetError::InvalidTruncation {
                requested: dimensions,
                native: self.dimensions,
            });
        }
        if matches!(self.format, ScalarFormat::B1x8) && !dimensions.is_multiple_of(8) {
            return Err(DatasetError::InvalidTruncation {
                requested: dimensions,
                native: self.dimensions,
            });
        }
        Ok(())
    }

    /// Reinterpret a raw byte slice as a typed `Vectors` at the requested
    /// `dimensions`. The byte slice must already be packed at the requested dimension
    /// (caller's responsibility — `slice` / `gather` produce such slices).
    fn wrap_bytes<'a>(&self, bytes: &'a [u8], num_vectors: usize, dimensions: usize) -> Vectors<'a> {
        let bytes_per_vector = bytes_for(self.format, dimensions);
        let total_bytes = num_vectors * bytes_per_vector;
        let data = match self.format {
            ScalarFormat::F32 => VectorSlice::F32(unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const f32, num_vectors * dimensions)
            }),
            ScalarFormat::U8 => VectorSlice::U8(&bytes[..num_vectors * dimensions]),
            ScalarFormat::I8 => VectorSlice::I8(unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const i8, num_vectors * dimensions)
            }),
            ScalarFormat::B1x8 => VectorSlice::B1x8(&bytes[..total_bytes]),
        };
        Vectors { data, dimensions }
    }

    /// Copy `count` row prefixes of length `bytes_per_row` starting at
    /// `start_row` into `destination`. Used by both the truncated-slice
    /// path and any cross-shard fallback — `Shards::row_bytes` hides the
    /// shard lookup.
    fn copy_row_prefixes(&self, start_row: usize, count: usize, bytes_per_row: usize, destination: &mut [u8]) {
        debug_assert!(destination.len() >= count * bytes_per_row);
        for row_offset in 0..count {
            let source = self.shards.row_bytes(start_row + row_offset, bytes_per_row);
            let destination_offset = row_offset * bytes_per_row;
            destination[destination_offset..destination_offset + bytes_per_row].copy_from_slice(source);
        }
    }

    /// Borrow a contiguous slice of vectors at `dimensions`. Zero-copy iff
    /// (a) `dimensions` equals the native dimensions and (b) the range falls within a
    /// single shard; otherwise rows are copied through `scratch`. `scratch`
    /// must be at least `count * bytes_for(format, dimensions)` bytes — unused on
    /// the zero-copy path but kept in the signature for type/lifetime
    /// uniformity.
    pub fn slice<'a>(&'a self, start: usize, count: usize, dimensions: usize, scratch: &'a mut [u8]) -> Vectors<'a> {
        let count = count.min(self.shards.rows() - start);
        let bytes_per_row = bytes_for(self.format, dimensions);
        if dimensions == self.dimensions {
            if let Some(bytes) = self.shards.zero_copy_range(start, count) {
                return self.wrap_bytes(bytes, count, dimensions);
            }
        }
        self.copy_row_prefixes(start, count, bytes_per_row, scratch);
        self.wrap_bytes(&scratch[..count * bytes_per_row], count, dimensions)
    }

    /// Gather vectors at the given indices into the provided buffer at
    /// `dimensions`. `buf` must be at least `indices.len() * bytes_for(format, dimensions)`
    /// bytes. When `dimensions` is below native, only the leading bytes of each
    /// source row are copied; the trailing tail is dropped.
    pub fn gather<'a>(&self, indices: &[usize], dimensions: usize, buf: &'a mut [u8]) -> Vectors<'a> {
        let bytes_per_row = bytes_for(self.format, dimensions);
        let num_vectors = indices.len();
        debug_assert!(buf.len() >= num_vectors * bytes_per_row);
        for (output_index, &source_index) in indices.iter().enumerate() {
            let source = self.shards.row_bytes(source_index, bytes_per_row);
            let destination_offset = output_index * bytes_per_row;
            buf[destination_offset..destination_offset + bytes_per_row].copy_from_slice(source);
        }
        self.wrap_bytes(&buf[..num_vectors * bytes_per_row], num_vectors, dimensions)
    }
}

/// A shuffled permutation of `[0..n]` for randomizing insertion order.
pub struct Permutation {
    indices: Vec<usize>,
}

impl Permutation {
    /// Create a shuffled permutation of `[0..n]` with a deterministic seed,
    /// parallelized across the ForkUnion thread pool.
    ///
    /// The output is uniformly random *within each chunk* of `ceil(n/threads)`
    /// consecutive positions. Across chunks, element ranges remain ordered
    /// (chunk 0 holds some subset of `[0, n/threads)`, chunk 1 holds a subset
    /// of `[n/threads, 2n/threads)`, etc.). This is sufficient for ANN
    /// benchmarking — it eliminates natural orderings (e.g. sorted molecules)
    /// and gives HNSW construction ~`chunk`-sized random windows to see a
    /// representative sample before moving on. A fully-uniform parallel
    /// shuffle would require parallel sort by random keys, which we skip to
    /// avoid adding a rayon/sort dep.
    pub fn shuffled(n: usize, seed: u64) -> Self {
        let mut indices: Vec<usize> = (0..n).collect();
        if n < PERMUTATION_SERIAL_THRESHOLD {
            let mut rng = SmallRng::seed_from_u64(seed);
            indices.shuffle(&mut rng);
            return Self { indices };
        }

        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .max(1);
        let mut pool = match ThreadPool::try_spawn(threads) {
            Ok(pool) => pool,
            Err(_) => {
                let mut rng = SmallRng::seed_from_u64(seed);
                indices.shuffle(&mut rng);
                return Self { indices };
            }
        };

        let chunk = n.div_ceil(threads);
        let shared_base = SyncMutPtr::new(indices.as_mut_ptr());

        // Per-chunk Fisher–Yates. Each thread only writes to its own
        // `[thread_index * chunk, (thread_index+1) * chunk)` range, so
        // there are no cross-thread races. Seeds per thread are derived
        // from `seed ^ mixer(thread_index)` for reproducibility.
        //
        // ForkUnion's `for_threads` takes a shared-ref closure (`Fn`), so
        // we can't capture `&mut [usize]` directly. We pass the disjoint
        // ranges via a `SyncMutPtr` and rebuild each thread's own
        // `&mut [usize]` inside the closure — one `from_raw_parts_mut`
        // call per thread, then safe `slice.swap(i, j)` in the hot loop.
        pool.for_threads(|thread_index, _| {
            let range_start = thread_index * chunk;
            if range_start >= n {
                return;
            }
            let range_end = (range_start + chunk).min(n);
            // SAFETY: the chunks are disjoint by construction
            // (`thread_index * chunk` is a monotonic multiple), so the
            // `&mut [usize]` rebuilt here never aliases with any other
            // thread's slice.
            let chunk_slice: &mut [usize] = unsafe {
                std::slice::from_raw_parts_mut(shared_base.as_ptr().add(range_start), range_end - range_start)
            };
            let mut rng = SmallRng::seed_from_u64(seed ^ (thread_index as u64).wrapping_mul(THREAD_SEED_MIXER));
            fisher_yates_shuffle(chunk_slice, &mut rng);
        });

        Self { indices }
    }

    /// Identity permutation (no shuffle).
    pub fn identity(n: usize) -> Self {
        Self {
            indices: (0..n).collect(),
        }
    }

    /// Get the permuted indices for a contiguous range `[start..start+count]`.
    pub fn range(&self, start: usize, count: usize) -> &[usize] {
        &self.indices[start..start + count]
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
    /// Load keys from a `.i32bin` path. Glob patterns expand to a sorted
    /// list of shards; `slice` falls back to a copy when a range crosses
    /// shard boundaries.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let pattern = path.to_string_lossy();
        let shard_paths = expand_glob(&pattern)?;

        let stride_bytes = std::mem::size_of::<Key>();
        let mut mmaps: Vec<Mmap> = Vec::with_capacity(shard_paths.len());
        let mut shard_rows: Vec<usize> = Vec::with_capacity(shard_paths.len());

        for shard_path in &shard_paths {
            let file = File::open(shard_path)?;
            let mmap = unsafe { Mmap::map(&file)? };

            if mmap.len() < HEADER_BYTES {
                return Err(std::io::Error::other(DatasetError::HeaderTooSmall {
                    path: shard_path.to_path_buf(),
                    kind: "keys",
                    got: mmap.len() as u64,
                }));
            }

            let count = u32::from_le_bytes(mmap[0..4].try_into().unwrap()) as usize;
            let expected = HEADER_BYTES + count * stride_bytes;
            if mmap.len() < expected {
                return Err(std::io::Error::other(DatasetError::BodyTooSmall {
                    path: shard_path.to_path_buf(),
                    expected: expected as u64,
                    rows: count as u32,
                    dimensions: 1,
                    got: mmap.len() as u64,
                }));
            }

            mmaps.push(mmap);
            shard_rows.push(count);
        }

        Ok(Keys::Mapped(Shards::new(mmaps, &shard_rows, stride_bytes)))
    }

    /// Create sequential keys 0..count.
    pub fn sequential(count: usize) -> Self {
        Keys::Sequential { count }
    }

    /// Get the key at a specific index.
    pub fn get(&self, index: usize) -> Key {
        debug_assert!(
            index < self.count(),
            "key index {index} out of bounds (count={})",
            self.count()
        );
        match self {
            Keys::Mapped(shards) => {
                let bytes = shards.row_bytes(index, std::mem::size_of::<Key>());
                Key::from_le_bytes(bytes.try_into().unwrap())
            }
            Keys::Sequential { .. } => index as Key,
        }
    }

    pub fn count(&self) -> usize {
        match self {
            Keys::Mapped(shards) => shards.rows(),
            Keys::Sequential { count } => *count,
        }
    }

    /// Borrow a contiguous slice of keys. Zero-copy when the range fits in
    /// a single mapped shard; otherwise (cross-shard or `Sequential`) the
    /// scratch buffer is filled and returned.
    pub fn slice<'a>(&'a self, start: usize, count: usize, scratch: &'a mut [Key]) -> &'a [Key] {
        let count = count.min(self.count() - start);
        match self {
            Keys::Mapped(shards) => {
                if let Some(bytes) = shards.zero_copy_range(start, count) {
                    // SAFETY: bytes spans `count * size_of::<Key>()` and the
                    // mmap base is page-aligned (so 4-byte alignment holds).
                    return unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const Key, count) };
                }
                for offset in 0..count {
                    let bytes = shards.row_bytes(start + offset, std::mem::size_of::<Key>());
                    scratch[offset] = Key::from_le_bytes(bytes.try_into().unwrap());
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
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let pattern = path.to_string_lossy();
        let shard_paths = expand_glob(&pattern)?;

        let mut mmaps: Vec<Mmap> = Vec::with_capacity(shard_paths.len());
        let mut shard_queries: Vec<usize> = Vec::with_capacity(shard_paths.len());
        let mut neighbors_per_query: Option<usize> = None;

        for shard_path in &shard_paths {
            let file = File::open(shard_path)?;
            let mmap = unsafe { Mmap::map(&file)? };

            if mmap.len() < HEADER_BYTES {
                return Err(std::io::Error::other(DatasetError::HeaderTooSmall {
                    path: shard_path.to_path_buf(),
                    kind: "ground_truth",
                    got: mmap.len() as u64,
                }));
            }

            let queries = u32::from_le_bytes(mmap[0..4].try_into().unwrap()) as usize;
            let neighbors = u32::from_le_bytes(mmap[4..8].try_into().unwrap()) as usize;

            // First shard pins neighbors_per_query; subsequent shards must match.
            match neighbors_per_query {
                None => neighbors_per_query = Some(neighbors),
                Some(prior) if prior == neighbors => {}
                Some(prior) => {
                    return Err(std::io::Error::other(DatasetError::ShardMismatch {
                        path: shard_path.to_path_buf(),
                        expected_dims: prior as u32,
                        got_dims: neighbors as u32,
                    }));
                }
            }

            let expected = HEADER_BYTES + queries * neighbors * std::mem::size_of::<Key>();
            if mmap.len() < expected {
                return Err(std::io::Error::other(DatasetError::BodyTooSmall {
                    path: shard_path.to_path_buf(),
                    expected: expected as u64,
                    rows: queries as u32,
                    dimensions: neighbors as u32,
                    got: mmap.len() as u64,
                }));
            }

            mmaps.push(mmap);
            shard_queries.push(queries);
        }

        let neighbors_per_query = neighbors_per_query.expect("shard_paths non-empty so neighbors_per_query was set");
        let stride_bytes = neighbors_per_query * std::mem::size_of::<Key>();

        Ok(Self {
            shards: Shards::new(mmaps, &shard_queries, stride_bytes),
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
        // SAFETY: bytes spans `neighbors_per_query * size_of::<Key>()` and
        // the mmap base is page-aligned, so 4-byte alignment for `Key = u32`
        // holds.
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const Key, self.neighbors_per_query) }
    }
}
