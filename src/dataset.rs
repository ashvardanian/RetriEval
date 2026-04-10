use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::{Key, VectorSlice, Vectors};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScalarFormat {
    F32,
    U8,
    I8,
}

/// A memory-mapped binary vector dataset.
///
/// File format (all little-endian):
/// - bytes 0..4: number of rows (u32)
/// - bytes 4..8: number of dimensions (u32)
/// - bytes 8..: row-major vector data
pub struct Dataset {
    mmap: Mmap,
    rows: usize,
    dimensions: usize,
    scalar_size: usize,
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
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "file too small for header",
            ));
        }

        let rows = u32::from_le_bytes(mmap[0..4].try_into().unwrap()) as usize;
        let dimensions = u32::from_le_bytes(mmap[4..8].try_into().unwrap()) as usize;

        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

        let (scalar_size, format) = match ext {
            "fbin" => (4, ScalarFormat::F32),
            "u8bin" => (1, ScalarFormat::U8),
            "i8bin" => (1, ScalarFormat::I8),
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("unsupported file extension: .{ext}"),
                ))
            }
        };

        let expected = 8 + rows * dimensions * scalar_size;
        if mmap.len() < expected {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "file too small: expected {expected} bytes for {rows}x{dimensions}, got {}",
                    mmap.len()
                ),
            ));
        }

        Ok(Self {
            mmap,
            rows,
            dimensions,
            scalar_size,
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
        let elements = num_vectors * dimensions;
        let data = match self.format {
            ScalarFormat::F32 => VectorSlice::F32(unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const f32, elements)
            }),
            ScalarFormat::U8 => VectorSlice::U8(&bytes[..elements]),
            ScalarFormat::I8 => VectorSlice::I8(unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const i8, elements)
            }),
        };
        Vectors { data, dimensions }
    }

    /// Borrow a contiguous slice of vectors directly from the mmap. Zero-copy.
    pub fn slice(&self, start: usize, count: usize) -> Vectors<'_> {
        let count = count.min(self.rows - start);
        let offset = 8 + start * self.dimensions * self.scalar_size;
        let byte_len = count * self.dimensions * self.scalar_size;
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

    /// Bytes per vector (dimensions * scalar_size).
    pub fn vector_bytes(&self) -> usize {
        self.dimensions * self.scalar_size
    }
}

/// A shuffled permutation of `[0..n]` for randomizing insertion order.
pub struct Permutation {
    indices: Vec<usize>,
}

impl Permutation {
    /// Create a shuffled permutation of `[0..n]` with a deterministic seed.
    pub fn shuffled(n: usize, seed: u64) -> Self {
        let mut indices: Vec<usize> = (0..n).collect();
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        indices.shuffle(&mut rng);
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
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "keys file too small for header",
            ));
        }

        let count = u32::from_le_bytes(mmap[0..4].try_into().unwrap()) as usize;
        let expected = 8 + count * std::mem::size_of::<Key>();
        if mmap.len() < expected {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("keys file too small: expected {expected} bytes for {count} keys"),
            ));
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
                for i in 0..count {
                    scratch[i] = (start + i) as Key;
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
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "ground truth file too small for header",
            ));
        }

        let queries = u32::from_le_bytes(mmap[0..4].try_into().unwrap()) as usize;
        let neighbors_per_query = u32::from_le_bytes(mmap[4..8].try_into().unwrap()) as usize;

        let expected = 8 + queries * neighbors_per_query * std::mem::size_of::<Key>();
        if mmap.len() < expected {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "ground truth too small: expected {expected} bytes for {queries}x{neighbors_per_query}"
                ),
            ));
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
        unsafe {
            std::slice::from_raw_parts(
                self.mmap[offset..].as_ptr() as *const Key,
                self.neighbors_per_query,
            )
        }
    }
}
