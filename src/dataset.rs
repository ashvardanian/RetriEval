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
    mmap: Mmap,
    rows: usize,
    dimensions: usize,
    bytes_per_vector: usize,
    format: ScalarFormat,
}

/// Ground-truth neighbor indices loaded from `.ibin` files.
pub struct GroundTruth {
    mmap: Mmap,
    queries: usize,
    neighbors_per_query: usize,
}

impl Dataset {
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < 8 {
            return Err(std::io::Error::other(DatasetError::HeaderTooSmall {
                path: path.to_path_buf(),
                kind: "dataset",
                got: mmap.len() as u64,
            }));
        }

        let rows = u32::from_le_bytes(mmap[0..4].try_into().unwrap()) as usize;
        let dimensions = u32::from_le_bytes(mmap[4..8].try_into().unwrap()) as usize;

        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

        let (bytes_per_vector, format) = match ext {
            "fbin" => (dimensions * 4, ScalarFormat::F32),
            "u8bin" => (dimensions, ScalarFormat::U8),
            "i8bin" => (dimensions, ScalarFormat::I8),
            "b1bin" => (dimensions.div_ceil(8), ScalarFormat::B1x8),
            _ => {
                return Err(std::io::Error::other(DatasetError::UnsupportedExtension {
                    path: path.to_path_buf(),
                    extension: ext.to_string(),
                }))
            }
        };

        let expected = 8 + rows * bytes_per_vector;
        if mmap.len() < expected {
            return Err(std::io::Error::other(DatasetError::BodyTooSmall {
                path: path.to_path_buf(),
                expected: expected as u64,
                rows: rows as u32,
                dims: dimensions as u32,
                got: mmap.len() as u64,
            }));
        }

        Ok(Self {
            mmap,
            rows,
            dimensions,
            bytes_per_vector,
            format,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Reinterpret a raw byte slice as a typed `Vectors` based on this dataset's scalar format.
    fn wrap_bytes<'a>(&self, bytes: &'a [u8], num_vectors: usize) -> Vectors<'a> {
        let dimensions = self.dimensions;
        let total_bytes = num_vectors * self.bytes_per_vector;
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

    /// Borrow a contiguous slice of vectors directly from the mmap. Zero-copy.
    pub fn slice(&self, start: usize, count: usize) -> Vectors<'_> {
        let count = count.min(self.rows - start);
        let offset = 8 + start * self.bytes_per_vector;
        let byte_len = count * self.bytes_per_vector;
        self.wrap_bytes(&self.mmap[offset..offset + byte_len], count)
    }

    /// Borrow all vectors from the mmap.
    pub fn all(&self) -> Vectors<'_> {
        self.slice(0, self.rows)
    }

    /// Gather vectors at the given indices into the provided buffer.
    /// The buffer must be at least `indices.len() * vector_bytes()` bytes.
    pub fn gather<'a>(&self, indices: &[usize], buf: &'a mut [u8]) -> Vectors<'a> {
        let bytes_per_vector = self.vector_bytes();
        let num_vectors = indices.len();
        debug_assert!(buf.len() >= num_vectors * bytes_per_vector);

        for (out_idx, &src_idx) in indices.iter().enumerate() {
            let src_offset = 8 + src_idx * bytes_per_vector;
            let dst_offset = out_idx * bytes_per_vector;
            buf[dst_offset..dst_offset + bytes_per_vector]
                .copy_from_slice(&self.mmap[src_offset..src_offset + bytes_per_vector]);
        }

        self.wrap_bytes(&buf[..num_vectors * bytes_per_vector], num_vectors)
    }

    /// Bytes per vector. For B1x8, this is `ceil(dimensions / 8)`.
    pub fn vector_bytes(&self) -> usize {
        self.bytes_per_vector
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
                std::slice::from_raw_parts_mut(
                    shared_base.as_ptr().add(range_start),
                    range_end - range_start,
                )
            };
            let mut rng = SmallRng::seed_from_u64(
                seed ^ (thread_index as u64).wrapping_mul(THREAD_SEED_MIXER),
            );
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

/// Vector keys, either loaded from a `.i32bin` file or generated sequentially.
pub enum Keys {
    /// Memory-mapped keys from a file (same format as `.ibin`: rows u32, dims u32, then Key data).
    Mapped { mmap: Mmap, count: usize },
    /// Sequential keys 0..count, generated on the fly.
    Sequential { count: usize },
}

impl Keys {
    /// Load keys from a `.i32bin` file.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < 8 {
            return Err(std::io::Error::other(DatasetError::HeaderTooSmall {
                path: path.to_path_buf(),
                kind: "keys",
                got: mmap.len() as u64,
            }));
        }

        let count = u32::from_le_bytes(mmap[0..4].try_into().unwrap()) as usize;
        let expected = 8 + count * std::mem::size_of::<Key>();
        if mmap.len() < expected {
            return Err(std::io::Error::other(DatasetError::BodyTooSmall {
                path: path.to_path_buf(),
                expected: expected as u64,
                rows: count as u32,
                dims: 1,
                got: mmap.len() as u64,
            }));
        }

        Ok(Keys::Mapped { mmap, count })
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
            Keys::Mapped { mmap, .. } => {
                let offset = 8 + index * std::mem::size_of::<Key>();
                let end = offset + std::mem::size_of::<Key>();
                assert!(end <= mmap.len(), "key index {index} out of mmap bounds");
                unsafe { *(mmap[offset..].as_ptr() as *const Key) }
            }
            Keys::Sequential { .. } => index as Key,
        }
    }

    pub fn count(&self) -> usize {
        match self {
            Keys::Mapped { count, .. } => *count,
            Keys::Sequential { count } => *count,
        }
    }

    /// Borrow a contiguous slice of keys. Zero-copy for mapped keys.
    /// For sequential keys, fills the provided scratch buffer and returns a slice into it.
    pub fn slice<'a>(&'a self, start: usize, count: usize, scratch: &'a mut [Key]) -> &'a [Key] {
        let count = count.min(self.count() - start);
        match self {
            Keys::Mapped { mmap, .. } => {
                let offset = 8 + start * std::mem::size_of::<Key>();
                unsafe { std::slice::from_raw_parts(mmap[offset..].as_ptr() as *const Key, count) }
            }
            Keys::Sequential { .. } => {
                for (i, slot) in scratch[..count].iter_mut().enumerate() {
                    *slot = (start + i) as Key;
                }
                &scratch[..count]
            }
        }
    }
}

impl GroundTruth {
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < 8 {
            return Err(std::io::Error::other(DatasetError::HeaderTooSmall {
                path: path.to_path_buf(),
                kind: "ground_truth",
                got: mmap.len() as u64,
            }));
        }

        let queries = u32::from_le_bytes(mmap[0..4].try_into().unwrap()) as usize;
        let neighbors_per_query = u32::from_le_bytes(mmap[4..8].try_into().unwrap()) as usize;

        let expected = 8 + queries * neighbors_per_query * std::mem::size_of::<Key>();
        if mmap.len() < expected {
            return Err(std::io::Error::other(DatasetError::BodyTooSmall {
                path: path.to_path_buf(),
                expected: expected as u64,
                rows: queries as u32,
                dims: neighbors_per_query as u32,
                got: mmap.len() as u64,
            }));
        }

        Ok(Self {
            mmap,
            queries,
            neighbors_per_query,
        })
    }

    pub fn queries(&self) -> usize {
        self.queries
    }

    pub fn neighbors_per_query(&self) -> usize {
        self.neighbors_per_query
    }

    /// Get the ground-truth neighbor IDs for a specific query.
    pub fn neighbors(&self, query_idx: usize) -> &[Key] {
        let offset = 8 + query_idx * self.neighbors_per_query * std::mem::size_of::<Key>();
        unsafe { std::slice::from_raw_parts(self.mmap[offset..].as_ptr() as *const Key, self.neighbors_per_query) }
    }
}
