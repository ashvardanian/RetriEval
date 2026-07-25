//! cuVS CAGRA GPU benchmark binary.
//!
//! CAGRA is a GPU-accelerated graph-based ANN algorithm from NVIDIA,
//! comparable to HNSW but designed for GPU parallelism.
//!
//! ## Prerequisites
//!
//! Requires `libcuvs` (CUDA 12), CMake >= 3.30, and the CUDA toolkit.
//! RAPIDS does not publish apt packages — install via conda or pip.
//!
//! **Option A — Conda** (cmake/linker find everything automatically):
//!
//! ```sh
//! conda install -c rapidsai -c conda-forge -c nvidia libcuvs cuda-version=12
//! ```
//!
//! **Option B — pip** (needs symlinks so cmake/linker discover the libraries):
//!
//! ```sh
//! pip install libcuvs-cu12
//! # Symlink into /opt/rapids so relative cmake configs resolve correctly.
//! # See: https://docs.rapids.ai/install
//! ```
//!
//! Then create `.cargo/config.toml` (git-ignored) pointing at your CUDA and
//! RAPIDS install prefixes:
//!
//! ```toml
//! [env]
//! BINDGEN_EXTRA_CLANG_ARGS = "-I/usr/local/cuda/targets/x86_64-linux/include"
//! CMAKE_PREFIX_PATH = "/opt/rapids/lib64/cmake"  # or conda env prefix
//!
//! [target.x86_64-unknown-linux-gnu]
//! rustflags = ["-L", "/opt/rapids/lib64", "-L", "/usr/local/cuda/lib64"]
//! ```
//!
//! ## Build & Run
//!
//! ```sh
//! cargo install --path . --features cuvs-backend
//! ```
//!
//! Quick data_type sweep:
//! ```sh
//! retri-eval-cuvs \
//!     --base-vectors datasets/wiki_1M/base.1M.fbin \
//!     --query-vectors datasets/wiki_1M/query.public.100K.fbin \
//!     --query-neighbors datasets/wiki_1M/groundtruth.public.100K.ibin \
//!     --data-type f32,f16 --metric ip \
//!     --output results/
//! ```
//!
//! Turing 10M at ~99% recall (M=64, ef=128/256, search_width=32):
//! ```sh
//! retri-eval-cuvs \
//!     --base-vectors datasets/turing_10M/base.10M.fbin \
//!     --query-vectors datasets/turing_10M/query.public.100K.fbin \
//!     --query-neighbors datasets/turing_10M/groundtruth.public.100K.ibin \
//!     --data-type f32,f16 --metric l2 \
//!     --connectivity 64 \
//!     --expansion-add 128 \
//!     --expansion-search 256 \
//!     --search-width 32 \
//!     --build-algo nn_descent \
//!     --output results/turing_10M
//! ```

use std::cell::Cell;
use std::cell::UnsafeCell;
use std::ptr::NonNull;

use clap::Parser;
use cuvs::distance_type::DistanceType;
use itertools::iproduct;
use retrieval::{
    run_config, Backend, BenchState, CommonArgs, Distance, IndexConfig, Key, SweepSummary, UnwrapOrBail, Vectors,
};
use serde_json::json;

// #region CudaAllocator

/// RMM-backed CUDA device memory allocator for NumKong tensors.
///
/// GPU-backed tensors must only use `as_ptr()` / `as_mut_ptr()` — host-side
/// accessors like `as_slice()` would dereference GPU pointers from the CPU.
#[derive(Clone)]
struct CudaAllocator(cuvs_sys::cuvsResources_t);

unsafe impl numkong::Allocator for CudaAllocator {
    fn allocate(&self, layout: std::alloc::Layout) -> Option<NonNull<u8>> {
        if layout.size() == 0 {
            return Some(NonNull::dangling());
        }
        unsafe {
            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            let err = cuvs_sys::cuvsRMMAlloc(self.0, &mut ptr, layout.size());
            if err != cuvs_sys::cuvsError_t::CUVS_SUCCESS || ptr.is_null() {
                return None;
            }
            NonNull::new(ptr as *mut u8)
        }
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: std::alloc::Layout) {
        if layout.size() > 0 {
            let _ = cuvs_sys::cuvsRMMFree(self.0, ptr.as_ptr() as *mut std::ffi::c_void, layout.size());
        }
    }
}

type GpuTensor<T> = numkong::Tensor<T, CudaAllocator>;

// #region DLPack

/// Build a non-owning `DLManagedTensor` descriptor.
///
/// The caller must keep the underlying data and `shape` alive for the
/// lifetime of the returned struct.  `deleter` is always `None` — ownership
/// stays with the caller (either a host buffer or a `GpuTensor`).
unsafe fn dl_tensor(
    data: *mut std::ffi::c_void,
    shape: *mut i64,
    ndim: i32,
    data_type: cuvs_sys::DLDataType,
    on_gpu: bool,
) -> cuvs_sys::DLManagedTensor {
    let device_type = if on_gpu {
        cuvs_sys::DLDeviceType::kDLCUDA
    } else {
        cuvs_sys::DLDeviceType::kDLCPU
    };
    cuvs_sys::DLManagedTensor {
        dl_tensor: cuvs_sys::DLTensor {
            data,
            device: cuvs_sys::DLDevice {
                device_type,
                device_id: 0,
            },
            ndim,
            dtype: data_type,
            shape,
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        },
        manager_ctx: std::ptr::null_mut(),
        deleter: None,
    }
}

// #region Dtype

/// Supported CAGRA scalar quantization types (matches the C API).
#[derive(Debug, Clone, Copy)]
enum CuvsDataType {
    F32,
    F16,
    I8,
    U8,
}

impl CuvsDataType {
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "f32" => Ok(Self::F32),
            "f16" => Ok(Self::F16),
            "i8" => Ok(Self::I8),
            "u8" => Ok(Self::U8),
            _ => Err(format!("unknown CAGRA data_type: {s}. supported: f32, f16, i8, u8")),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::I8 => "i8",
            Self::U8 => "u8",
        }
    }

    fn dl_type(self) -> cuvs_sys::DLDataType {
        let (code, bits) = match self {
            Self::F32 => (cuvs_sys::DLDataTypeCode::kDLFloat, 32),
            Self::F16 => (cuvs_sys::DLDataTypeCode::kDLFloat, 16),
            Self::I8 => (cuvs_sys::DLDataTypeCode::kDLInt, 8),
            Self::U8 => (cuvs_sys::DLDataTypeCode::kDLUInt, 8),
        };
        cuvs_sys::DLDataType {
            code: code as u8,
            bits,
            lanes: 1,
        }
    }

    fn bytes_per_element(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::I8 | Self::U8 => 1,
        }
    }

    /// Convert f32 values to the target data_type, appending raw bytes to `output`.
    fn convert_from_f32(self, source: &[f32], output: &mut Vec<u8>) {
        match self {
            Self::F32 => {
                let bytes = unsafe { std::slice::from_raw_parts(source.as_ptr() as *const u8, source.len() * 4) };
                output.extend_from_slice(bytes);
            }
            Self::F16 => {
                output.reserve(source.len() * 2);
                for &value in source {
                    output.extend_from_slice(&numkong::f16::from_f32(value).0.to_ne_bytes());
                }
            }
            Self::I8 => {
                output.reserve(source.len());
                for &value in source {
                    output.push(value.clamp(-128.0, 127.0) as i8 as u8);
                }
            }
            Self::U8 => {
                output.reserve(source.len());
                for &value in source {
                    output.push(value.clamp(0.0, 255.0) as u8);
                }
            }
        }
    }
}

fn dl_key() -> cuvs_sys::DLDataType {
    cuvs_sys::DLDataType {
        code: cuvs_sys::DLDataTypeCode::kDLUInt as u8,
        bits: 32,
        lanes: 1,
    }
}

fn dl_distance() -> cuvs_sys::DLDataType {
    cuvs_sys::DLDataType {
        code: cuvs_sys::DLDataTypeCode::kDLFloat as u8,
        bits: 32,
        lanes: 1,
    }
}

// #region CagraIndex
//
// Local thin wrapper around `cuvs_sys::cuvsCagraIndex_t`. We can't use
// `cuvs::cagra::Index` because its inner pointer field is private and the
// crate exposes no `serialize` / `deserialize` methods (CAGRA save/load
// landed in the C API but the Rust 26.4 wrapper hasn't surfaced it yet).
// Building on cuvs-sys directly lets us reach `cuvsCagraSerialize` /
// `cuvsCagraDeserialize` while keeping `IndexParams` / `SearchParams` from
// the high-level crate (their `.0` fields are public).

struct CagraIndex(cuvs_sys::cuvsCagraIndex_t);

fn cuvs_check(err: cuvs_sys::cuvsError_t, ctx: &str) -> Result<(), String> {
    if err == cuvs_sys::cuvsError_t::CUVS_SUCCESS {
        Ok(())
    } else {
        Err(format!("{ctx}: cuvsError_t = {err:?}"))
    }
}

impl CagraIndex {
    fn new() -> Result<Self, String> {
        let mut handle = std::mem::MaybeUninit::<cuvs_sys::cuvsCagraIndex_t>::uninit();
        unsafe {
            cuvs_check(
                cuvs_sys::cuvsCagraIndexCreate(handle.as_mut_ptr()),
                "cuvsCagraIndexCreate",
            )?;
            Ok(Self(handle.assume_init()))
        }
    }

    /// Build the index from a host- or device-resident DLPack tensor. Caller
    /// keeps `dataset_dl` alive for the duration of the call.
    fn build(
        &self,
        res: &cuvs::Resources,
        params: cuvs_sys::cuvsCagraIndexParams_t,
        dataset_dl: *mut cuvs_sys::DLManagedTensor,
    ) -> Result<(), String> {
        unsafe {
            cuvs_check(
                cuvs_sys::cuvsCagraBuild(res.0, params, dataset_dl, self.0),
                "cuvsCagraBuild",
            )
        }
    }

    /// Search the index. All three tensors must reference live memory
    /// (typically GPU buffers wrapped in `DLManagedTensor` views).
    fn search(
        &self,
        res: &cuvs::Resources,
        params: cuvs_sys::cuvsCagraSearchParams_t,
        queries: *mut cuvs_sys::DLManagedTensor,
        neighbors: *mut cuvs_sys::DLManagedTensor,
        distances: *mut cuvs_sys::DLManagedTensor,
    ) -> Result<(), String> {
        let prefilter = cuvs_sys::cuvsFilter {
            addr: 0,
            type_: cuvs_sys::cuvsFilterType::NO_FILTER,
        };
        unsafe {
            cuvs_check(
                cuvs_sys::cuvsCagraSearch(res.0, params, self.0, queries, neighbors, distances, prefilter),
                "cuvsCagraSearch",
            )
        }
    }

    fn serialize(&self, res: &cuvs::Resources, path: &str, include_dataset: bool) -> Result<(), String> {
        let c_path = std::ffi::CString::new(path).map_err(|e| format!("path contains NUL: {e}"))?;
        unsafe {
            cuvs_check(
                cuvs_sys::cuvsCagraSerialize(res.0, c_path.as_ptr(), self.0, include_dataset),
                "cuvsCagraSerialize",
            )
        }
    }

    fn deserialize(&self, res: &cuvs::Resources, path: &str) -> Result<(), String> {
        let c_path = std::ffi::CString::new(path).map_err(|e| format!("path contains NUL: {e}"))?;
        unsafe {
            cuvs_check(
                cuvs_sys::cuvsCagraDeserialize(res.0, c_path.as_ptr(), self.0),
                "cuvsCagraDeserialize",
            )
        }
    }
}

impl Drop for CagraIndex {
    fn drop(&mut self) {
        unsafe {
            // Ignore destroy errors during drop — there's nothing useful to do
            // and the Rust crate does the same dance.
            let _ = cuvs_sys::cuvsCagraIndexDestroy(self.0);
        }
    }
}

// SAFETY: cuvsCagraIndex_t is a thin pointer to a cuVS-managed struct;
// the existing CuvsBackend already declares Send/Sync via its res/cuda_alloc.
unsafe impl Send for CagraIndex {}
unsafe impl Sync for CagraIndex {}

/// Path of the host-keys sidecar. CAGRA's serializer persists the device-side
/// dataset and graph but knows nothing about our row-index → user-key mapping
/// (it uses sequential IDs internally), so we write the keys ourselves next
/// to the index file.
fn keys_sidecar_path(handle: &str) -> String {
    format!("{handle}.keys")
}

/// Sidecar layout: `magic | graph_degree: u64 | count: u64 | keys`, all
/// little-endian. Widths are fixed rather than `usize` because this is a file
/// read back on whatever machine opens the index; the counts are `u64` to match
/// the key count the format already carried. The magic is what makes a
/// pre-graph-degree file a loud error instead of one that parses with a
/// nonsense degree, since the old layout opened with a bare `u64`.
const KEYS_SIDECAR_MAGIC: &[u8; 8] = b"RETRIKY1";
const KEYS_SIDECAR_HEADER_BYTES: usize = 24;

/// `graph_degree` rides along because CAGRA's deserializer cannot report the
/// build parameters, and `memory_bytes()` needs the degree to size the graph.
fn write_host_keys(path: &str, keys: &[Key], graph_degree: usize) -> Result<(), String> {
    let mut bytes = Vec::with_capacity(KEYS_SIDECAR_HEADER_BYTES + keys.len() * std::mem::size_of::<Key>());
    bytes.extend_from_slice(KEYS_SIDECAR_MAGIC);
    bytes.extend_from_slice(&(graph_degree as u64).to_le_bytes());
    bytes.extend_from_slice(&(keys.len() as u64).to_le_bytes());
    for &key in keys {
        bytes.extend_from_slice(&key.to_le_bytes());
    }
    std::fs::write(path, &bytes).map_err(|e| format!("write {path}: {e}"))
}

fn read_host_keys(path: &str) -> Result<(Vec<Key>, usize), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
    if bytes.len() < KEYS_SIDECAR_HEADER_BYTES || &bytes[0..8] != KEYS_SIDECAR_MAGIC {
        return Err(format!(
            "{path}: not a versioned key sidecar — it predates graph-degree persistence, rebuild the index"
        ));
    }
    let graph_degree = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    let count = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
    let expected = KEYS_SIDECAR_HEADER_BYTES + count * std::mem::size_of::<Key>();
    if bytes.len() < expected {
        return Err(format!("{path}: expected {expected} bytes, got {}", bytes.len()));
    }
    let mut keys = Vec::with_capacity(count);
    for key_index in 0..count {
        let offset = KEYS_SIDECAR_HEADER_BYTES + key_index * std::mem::size_of::<Key>();
        keys.push(Key::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()));
    }
    Ok((keys, graph_degree))
}

// #region GpuQueries

/// Typed GPU query buffer, matching the `VectorSlice` enum pattern.
enum GpuQueries {
    F32(GpuTensor<f32>),
    F16(GpuTensor<numkong::f16>),
    I8(GpuTensor<i8>),
    U8(GpuTensor<u8>),
}

impl GpuQueries {
    fn allocate(data_type: CuvsDataType, shape: &[usize], allocator: CudaAllocator) -> Result<Self, String> {
        let error = |e| format!("GPU query alloc failed: {e}");
        unsafe {
            Ok(match data_type {
                CuvsDataType::F32 => Self::F32(GpuTensor::try_empty_in(shape, allocator).map_err(error)?),
                CuvsDataType::F16 => Self::F16(GpuTensor::try_empty_in(shape, allocator).map_err(error)?),
                CuvsDataType::I8 => Self::I8(GpuTensor::try_empty_in(shape, allocator).map_err(error)?),
                CuvsDataType::U8 => Self::U8(GpuTensor::try_empty_in(shape, allocator).map_err(error)?),
            })
        }
    }

    fn as_mut_ptr(&mut self) -> *mut std::ffi::c_void {
        match self {
            Self::F32(tensor) => tensor.as_mut_ptr() as _,
            Self::F16(tensor) => tensor.as_mut_ptr() as _,
            Self::I8(tensor) => tensor.as_mut_ptr() as _,
            Self::U8(tensor) => tensor.as_mut_ptr() as _,
        }
    }
}

// #region SearchBuffers

/// Pre-allocated GPU + host buffers for search, reused across calls.
///
/// Sized for `capacity_queries` rows and exactly `neighbor_count` neighbors.
/// A batch shorter than the capacity is a contiguous row-major prefix of the
/// same allocation, so it reuses the buffers and only the leading extent of
/// the DLPack shapes changes — see `set_batch_rows`.
struct SearchBuffers {
    queries: GpuQueries,
    neighbors: GpuTensor<Key>,
    distances: GpuTensor<Distance>,

    /// Rows the buffers were sized for. Batches at or below this reuse them.
    capacity_queries: usize,
    /// Neighbors per query the buffers — and `search_params` — were sized for.
    neighbor_count: usize,

    queries_shape: [i64; 2],
    neighbors_shape: [i64; 2],
    distances_shape: [i64; 2],

    neighbors_host: Vec<Key>,
    distances_host: Vec<Distance>,

    /// Reusable host-side buffer for data_type conversion before H2D copy.
    query_staging: Vec<u8>,

    /// Search params, cached for as long as `neighbor_count` holds — the itopk
    /// floor is derived from it, so a different neighbor count needs new params.
    search_params: cuvs::cagra::SearchParams,
}

impl SearchBuffers {
    fn allocate(
        allocator: CudaAllocator,
        capacity_queries: usize,
        dimensions: usize,
        neighbor_count: usize,
        data_type: CuvsDataType,
        search_params: cuvs::cagra::SearchParams,
    ) -> Result<Self, String> {
        let error = |e| format!("GPU alloc failed: {e}");
        let queries = GpuQueries::allocate(data_type, &[capacity_queries, dimensions], allocator.clone())?;
        let neighbors = unsafe {
            GpuTensor::<Key>::try_empty_in(&[capacity_queries, neighbor_count], allocator.clone()).map_err(error)?
        };
        let distances = unsafe {
            GpuTensor::<Distance>::try_empty_in(&[capacity_queries, neighbor_count], allocator).map_err(error)?
        };
        Ok(Self {
            queries,
            neighbors,
            distances,
            capacity_queries,
            neighbor_count,
            queries_shape: [capacity_queries as i64, dimensions as i64],
            neighbors_shape: [capacity_queries as i64, neighbor_count as i64],
            distances_shape: [capacity_queries as i64, neighbor_count as i64],
            neighbors_host: vec![Key::default(); capacity_queries * neighbor_count],
            distances_host: vec![Distance::default(); capacity_queries * neighbor_count],
            query_staging: Vec::new(),
            search_params,
        })
    }

    /// Whether this allocation can serve a `num_queries` × `neighbor_count` batch.
    fn fits(&self, num_queries: usize, neighbor_count: usize) -> bool {
        num_queries <= self.capacity_queries && neighbor_count == self.neighbor_count
    }

    /// Point the DLPack shapes at the current batch, leaving the allocation
    /// alone. Without this a trailing partial batch would keep the previous
    /// batch's row count and make CAGRA search stale rows.
    fn set_batch_rows(&mut self, num_queries: usize) {
        self.queries_shape[0] = num_queries as i64;
        self.neighbors_shape[0] = num_queries as i64;
        self.distances_shape[0] = num_queries as i64;
    }
}

// #region CLI

#[derive(Parser, Debug)]
#[command(name = "retri-eval-cuvs", about = "Benchmark cuVS CAGRA (GPU)")]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,

    /// Quantization types (comma-separated): f32, f16, i8, u8
    #[arg(long, value_delimiter = ',', default_value = "f32")]
    data_type: Vec<String>,

    /// Distance metric: l2, ip, cos
    #[arg(long, value_delimiter = ',', default_value = "l2")]
    metric: Vec<String>,

    /// CAGRA output graph degree — the shared-vocabulary name for HNSW's M.
    #[arg(long, value_delimiter = ',', default_value = "32")]
    connectivity: Vec<usize>,

    /// CAGRA intermediate graph degree before pruning — analogous to
    /// `ef_construction`.
    #[arg(long, value_delimiter = ',', default_value = "64")]
    expansion_add: Vec<usize>,

    /// CAGRA internal top-i list retained during search — analogous to
    /// `ef_search`. Higher values improve recall at the cost of speed.
    #[arg(long, value_delimiter = ',', default_value = "64")]
    expansion_search: Vec<usize>,

    /// Number of graph nodes used as starting points per search iteration.
    /// Higher values improve recall. 0 = auto.
    #[arg(long, value_delimiter = ',', default_value = "0")]
    search_width: Vec<usize>,

    /// Minimum search iterations (prevents early termination). 0 = auto.
    #[arg(long, default_value_t = 0)]
    min_iterations: usize,

    /// Maximum search iterations. 0 = auto.
    #[arg(long, default_value_t = 0)]
    max_iterations: usize,

    /// Number of random seed sampling rounds for initial search points. 0 = auto.
    #[arg(long, default_value_t = 0)]
    num_random_samplings: u32,

    /// Graph build algorithm: auto, nn_descent
    #[arg(long, default_value = "auto")]
    build_algo: String,
}

// #region Metric

fn parse_metric(s: &str) -> Result<DistanceType, String> {
    match s {
        "l2" | "l2sq" => Ok(DistanceType::L2Expanded),
        "ip" => Ok(DistanceType::InnerProduct),
        "cos" => Ok(DistanceType::CosineExpanded),
        _ => Err(format!("unknown metric: {s}. supported: l2, ip, cos")),
    }
}

fn metric_label(s: &str) -> &str {
    match s {
        "ip" => "ip",
        "cos" => "cos",
        _ => "l2",
    }
}

fn parse_build_algo(s: &str) -> Result<cuvs_sys::cuvsCagraGraphBuildAlgo, String> {
    match s {
        "auto" => Ok(cuvs_sys::cuvsCagraGraphBuildAlgo::AUTO_SELECT),
        "nn_descent" => Ok(cuvs_sys::cuvsCagraGraphBuildAlgo::NN_DESCENT),
        _ => Err(format!("unknown build algo: {s}. supported: auto, nn_descent")),
    }
}

// #region Backend

pub struct CuvsBackend {
    // GPU resources that depend on `res` are declared BEFORE `res`
    // so they drop first (Rust drops fields in declaration order).
    search_buffers: UnsafeCell<Option<SearchBuffers>>,
    index: UnsafeCell<Option<CagraIndex>>,

    res: cuvs::Resources,
    cuda_alloc: CudaAllocator,
    dimensions: usize,
    metric: DistanceType,
    data_type: CuvsDataType,
    connectivity: usize,
    expansion_add: usize,
    build_algo: cuvs_sys::cuvsCagraGraphBuildAlgo,
    expansion_search: usize,
    search_width: usize,
    min_iterations: usize,
    max_iterations: usize,
    num_random_samplings: u32,

    host_vectors: Vec<u8>,
    host_keys: Vec<Key>,
    dirty: Cell<bool>,

    description: String,
    metadata: std::collections::HashMap<String, serde_json::Value>,
}

unsafe impl Send for CuvsBackend {}
unsafe impl Sync for CuvsBackend {}

/// CAGRA knobs with no counterpart in the shared vocabulary.
#[derive(Clone, Copy)]
pub struct CagraTuning<'a> {
    pub build_algo: &'a str,
    pub search_width: usize,
    pub min_iterations: usize,
    pub max_iterations: usize,
    pub num_random_samplings: u32,
}

impl CuvsBackend {
    pub fn new(config: IndexConfig<'_>, tuning: CagraTuning<'_>) -> Result<Self, String> {
        let IndexConfig {
            dimensions,
            data_type: data_type_name,
            metric: metric_name,
            connectivity,
            expansion_add,
            expansion_search,
        } = config;
        let CagraTuning {
            build_algo: build_algo_name,
            search_width,
            min_iterations,
            max_iterations,
            num_random_samplings,
        } = tuning;

        let metric = parse_metric(metric_name)?;
        let data_type = CuvsDataType::from_str(data_type_name)?;
        let build_algo = parse_build_algo(build_algo_name)?;
        let res = cuvs::Resources::new().map_err(|e| format!("failed to create cuVS resources: {e}"))?;
        let cuda_alloc = CudaAllocator(res.0);

        let description = format!(
            "cuvs-cagra \u{b7} {} \u{b7} {metric_name} \u{b7} M={connectivity} \u{b7} \
             ef={expansion_add}/{expansion_search} \u{b7} sw={search_width}",
            data_type.as_str(),
        );

        let mut metadata = std::collections::HashMap::new();
        metadata.insert("backend".into(), json!("cuvs-cagra"));
        metadata.insert("data_type".into(), json!(data_type.as_str()));
        metadata.insert("metric".into(), json!(metric_label(metric_name)));
        metadata.insert("connectivity".into(), json!(connectivity));
        metadata.insert("expansion_add".into(), json!(expansion_add));
        metadata.insert("expansion_search".into(), json!(expansion_search));
        metadata.insert("search_width".into(), json!(search_width));
        metadata.insert("build_algo".into(), json!(build_algo_name));
        metadata.insert("min_iterations".into(), json!(min_iterations));
        metadata.insert("max_iterations".into(), json!(max_iterations));
        metadata.insert("num_random_samplings".into(), json!(num_random_samplings));

        Ok(Self {
            search_buffers: UnsafeCell::new(None),
            index: UnsafeCell::new(None),
            res,
            cuda_alloc,
            dimensions,
            metric,
            data_type,
            connectivity,
            expansion_add,
            build_algo,
            expansion_search,
            search_width,
            min_iterations,
            max_iterations,
            num_random_samplings,
            host_vectors: Vec::new(),
            host_keys: Vec::new(),
            dirty: Cell::new(false),
            description,
            metadata,
        })
    }

    /// Open a saved CAGRA index. CAGRA's deserializer reports no build params,
    /// so the graph degree comes from the key sidecar written alongside the
    /// index; `config`'s build-time fields are ignored on this path.
    pub fn load(handle: &str, config: IndexConfig<'_>, tuning: CagraTuning<'_>) -> Result<Self, String> {
        let IndexConfig {
            dimensions,
            data_type: data_type_name,
            metric: metric_name,
            expansion_search,
            ..
        } = config;
        let CagraTuning {
            search_width,
            min_iterations,
            max_iterations,
            num_random_samplings,
            ..
        } = tuning;

        let metric = parse_metric(metric_name)?;
        let data_type = CuvsDataType::from_str(data_type_name)?;
        let res = cuvs::Resources::new().map_err(|e| format!("failed to create cuVS resources: {e}"))?;
        let cuda_alloc = CudaAllocator(res.0);

        let index = CagraIndex::new()?;
        index.deserialize(&res, handle)?;
        // The sidecar, not `config.connectivity`, is the authority here — it
        // records what the index was actually built with.
        let (host_keys, connectivity) = read_host_keys(&keys_sidecar_path(handle))?;

        let description = format!(
            "cuvs-cagra \u{b7} {} \u{b7} {metric_name} \u{b7} M={connectivity} \u{b7} ef={expansion_search} \u{b7} sw={search_width} \u{b7} loaded[{handle}]",
            data_type.as_str(),
        );

        let mut metadata = std::collections::HashMap::new();
        metadata.insert("backend".into(), json!("cuvs-cagra"));
        metadata.insert("data_type".into(), json!(data_type.as_str()));
        metadata.insert("metric".into(), json!(metric_label(metric_name)));
        metadata.insert("connectivity".into(), json!(connectivity));
        metadata.insert("expansion_search".into(), json!(expansion_search));
        metadata.insert("search_width".into(), json!(search_width));
        metadata.insert("loaded_from".into(), json!(handle));
        // Build-time knobs the serialized index does not carry. `connectivity`
        // escapes this because the key sidecar records it.
        metadata.insert("expansion_add".into(), serde_json::Value::Null);
        metadata.insert("build_algo".into(), serde_json::Value::Null);
        metadata.insert("min_iterations".into(), serde_json::Value::Null);
        metadata.insert("max_iterations".into(), serde_json::Value::Null);
        metadata.insert("num_random_samplings".into(), serde_json::Value::Null);

        Ok(Self {
            search_buffers: UnsafeCell::new(None),
            index: UnsafeCell::new(Some(index)),
            res,
            cuda_alloc,
            dimensions,
            metric,
            data_type,
            connectivity,
            expansion_add: 0,
            build_algo: cuvs_sys::cuvsCagraGraphBuildAlgo::AUTO_SELECT,
            expansion_search,
            search_width,
            min_iterations,
            max_iterations,
            num_random_samplings,
            host_vectors: Vec::new(),
            host_keys,
            dirty: Cell::new(false),
            description,
            metadata,
        })
    }

    /// Create search params once, reused across all search calls.
    fn build_search_params(&self, neighbor_count: usize) -> Result<cuvs::cagra::SearchParams, String> {
        let effective_itopk = self.expansion_search.max(neighbor_count);
        let params = cuvs::cagra::SearchParams::new()
            .map_err(|e| format!("search params: {e}"))?
            .set_itopk_size(effective_itopk);
        unsafe {
            let raw = params.0;
            if self.search_width > 0 {
                (*raw).search_width = self.search_width;
            }
            if self.min_iterations > 0 {
                (*raw).min_iterations = self.min_iterations;
            }
            if self.max_iterations > 0 {
                (*raw).max_iterations = self.max_iterations;
            }
            if self.num_random_samplings > 0 {
                (*raw).num_random_samplings = self.num_random_samplings;
            }
        }
        Ok(params)
    }

    /// Build (or rebuild) the CAGRA index from accumulated host buffers.
    fn build_index(&self) -> Result<(), String> {
        let num_vectors = self.host_keys.len();
        let dimensions = self.dimensions;
        let mut shape = [num_vectors as i64, dimensions as i64];

        let mut host_dl = unsafe {
            dl_tensor(
                self.host_vectors.as_ptr() as *mut _,
                shape.as_mut_ptr(),
                2,
                self.data_type.dl_type(),
                false,
            )
        };

        let build_params = cuvs::cagra::IndexParams::new()
            .map_err(|e| format!("failed to create CAGRA index params: {e}"))?
            .set_graph_degree(self.connectivity)
            .set_intermediate_graph_degree(self.expansion_add)
            .set_build_algo(self.build_algo);
        unsafe { (*build_params.0).metric = self.metric };

        let index = CagraIndex::new()?;
        index.build(&self.res, build_params.0, &mut host_dl)?;

        unsafe { *self.index.get() = Some(index) };
        self.dirty.set(false);
        Ok(())
    }

    /// Copy device tensor to a pre-allocated host slice, synchronising the stream.
    unsafe fn device_to_host<T>(&self, device_ptr: *const T, host: &mut [T]) -> Result<(), String> {
        let bytes = std::mem::size_of_val(host);
        let stream = self.res.get_cuda_stream().map_err(|e| format!("{e}"))?;
        let err = cuvs_sys::cudaMemcpyAsync(
            host.as_mut_ptr() as *mut _,
            device_ptr as *const _,
            bytes,
            cuvs_sys::cudaMemcpyKind_cudaMemcpyDefault,
            stream,
        );
        if err != cuvs_sys::cudaError::cudaSuccess {
            return Err(format!("cudaMemcpyAsync D2H failed: {err:?}"));
        }
        self.res.sync_stream().map_err(|e| format!("{e}"))
    }

    /// Copy host bytes to a pre-allocated device pointer.
    unsafe fn host_to_device(&self, host: &[u8], device_ptr: *mut std::ffi::c_void) -> Result<(), String> {
        let stream = self.res.get_cuda_stream().map_err(|e| format!("{e}"))?;
        let err = cuvs_sys::cudaMemcpyAsync(
            device_ptr,
            host.as_ptr() as *const _,
            host.len(),
            cuvs_sys::cudaMemcpyKind_cudaMemcpyDefault,
            stream,
        );
        if err != cuvs_sys::cudaError::cudaSuccess {
            return Err(format!("cudaMemcpyAsync H2D failed: {err:?}"));
        }
        Ok(())
    }
}

impl Backend for CuvsBackend {
    fn description(&self) -> String {
        self.description.clone()
    }

    fn metadata(&self) -> std::collections::HashMap<String, serde_json::Value> {
        self.metadata.clone()
    }

    fn add(&mut self, keys: &[Key], vectors: Vectors) -> Result<(), String> {
        let f32_data = vectors.data.to_f32();
        self.data_type.convert_from_f32(&f32_data, &mut self.host_vectors);
        self.host_keys.extend_from_slice(keys);
        self.dirty.set(true);
        Ok(())
    }

    fn search(
        &self,
        queries: Vectors,
        count: usize,
        out_keys: &mut [Key],
        out_distances: &mut [Distance],
        out_counts: &mut [usize],
    ) -> Result<(), String> {
        if self.dirty.get() || unsafe { (*self.index.get()).is_none() } {
            self.build_index()?;
        }

        let num_queries = queries.len();

        // Allocate GPU buffers and search params on first use, and again whenever
        // a batch outgrows them or asks for a different neighbor count — the
        // `--self-search` pass replays the whole base at its own count, and the
        // capacity never shrinks so alternating batch sizes don't thrash.
        let buffers = unsafe { &mut *self.search_buffers.get() };
        if !buffers.as_ref().is_some_and(|b| b.fits(num_queries, count)) {
            let capacity = num_queries.max(buffers.as_ref().map_or(0, |b| b.capacity_queries));
            let search_params = self.build_search_params(count)?;
            // Free the old buffers before allocating the new ones: reallocation
            // is rare and device memory is the scarce resource here.
            *buffers = None;
            *buffers = Some(SearchBuffers::allocate(
                self.cuda_alloc.clone(),
                capacity,
                queries.dimensions,
                count,
                self.data_type,
                search_params,
            )?);
        }
        let buffers = buffers.as_mut().unwrap();
        buffers.set_batch_rows(num_queries);

        // Upload queries to GPU. For f32 data_type, copy directly from the source
        // data without an intermediate staging buffer.
        let query_f32 = queries.data.to_f32();
        if matches!(self.data_type, CuvsDataType::F32) {
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    query_f32.as_ptr() as *const u8,
                    query_f32.len() * std::mem::size_of::<f32>(),
                )
            };
            unsafe { self.host_to_device(bytes, buffers.queries.as_mut_ptr())? };
        } else {
            buffers.query_staging.clear();
            self.data_type.convert_from_f32(&query_f32, &mut buffers.query_staging);
            unsafe { self.host_to_device(&buffers.query_staging, buffers.queries.as_mut_ptr())? };
        }

        // Non-owning DLPack views over the pre-allocated GpuTensor memory.
        // We pass raw `*mut DLManagedTensor` pointers into cuvs-sys directly,
        // so there's no `cuvs::ManagedTensor` wrapper to mem::forget afterward.
        let mut queries_dl = unsafe {
            dl_tensor(
                buffers.queries.as_mut_ptr(),
                buffers.queries_shape.as_mut_ptr(),
                2,
                self.data_type.dl_type(),
                true,
            )
        };
        let mut neighbors_dl = unsafe {
            dl_tensor(
                buffers.neighbors.as_mut_ptr() as _,
                buffers.neighbors_shape.as_mut_ptr(),
                2,
                dl_key(),
                true,
            )
        };
        let mut distances_dl = unsafe {
            dl_tensor(
                buffers.distances.as_mut_ptr() as _,
                buffers.distances_shape.as_mut_ptr(),
                2,
                dl_distance(),
                true,
            )
        };

        let index = unsafe { &*self.index.get() }.as_ref().ok_or("index not built")?;

        index.search(
            &self.res,
            buffers.search_params.0,
            &mut queries_dl,
            &mut neighbors_dl,
            &mut distances_dl,
        )?;

        // D2H copy results — only the rows this batch actually filled, which is
        // a prefix of the (possibly larger) capacity.
        let filled = num_queries * count;
        unsafe {
            self.device_to_host(buffers.neighbors.as_ptr(), &mut buffers.neighbors_host[..filled])?;
            self.device_to_host(buffers.distances.as_ptr(), &mut buffers.distances_host[..filled])?;
        }

        // Map CAGRA's 0-based row indices back to the caller's keys. Out-of-range
        // indices are CAGRA's tail padding.
        for (query_index, found_count) in out_counts[..num_queries].iter_mut().enumerate() {
            let offset = query_index * count;
            let hits = (0..count).filter_map(|rank| {
                let neighbor_index = buffers.neighbors_host[offset + rank] as usize;
                let key = *self.host_keys.get(neighbor_index)?;
                Some((key, buffers.distances_host[offset + rank]))
            });
            *found_count = retrieval::write_row(
                hits,
                &mut out_keys[offset..offset + count],
                &mut out_distances[offset..offset + count],
            );
        }

        Ok(())
    }

    /// The host-key sidecar has one entry per indexed vector, on both the
    /// freshly-built and the deserialized path.
    fn indexed_count(&self) -> Option<usize> {
        Some(self.host_keys.len())
    }

    fn memory_bytes(&self) -> usize {
        let num_vectors = self.host_keys.len();
        let host_bytes = self.host_vectors.len() + num_vectors * std::mem::size_of::<Key>();
        let gpu_bytes = num_vectors * self.dimensions * self.data_type.bytes_per_element()
            + num_vectors * self.connectivity * std::mem::size_of::<u32>();
        host_bytes + gpu_bytes
    }

    fn save(&self, handle: &str) -> Result<(), String> {
        let index_ref = unsafe { &*self.index.get() };
        let index = index_ref.as_ref().ok_or("CAGRA index not built — nothing to save")?;
        index.serialize(&self.res, handle, /*include_dataset=*/ true)?;
        write_host_keys(&keys_sidecar_path(handle), &self.host_keys, self.connectivity)
    }
}

// #region main

fn main() {
    let cli = Cli::parse();

    let mut state = BenchState::load(&cli.common).unwrap_or_bail("benchmark state");
    let dimensions_sweep = cli.common.dimensions_sweep(state.dimensions());

    cli.common.ensure_single_config(&[
        dimensions_sweep.len(),
        cli.data_type.len(),
        cli.metric.len(),
        cli.connectivity.len(),
        cli.expansion_add.len(),
        cli.expansion_search.len(),
        cli.search_width.len(),
    ]);

    let mut summary = SweepSummary::default();
    for (&dimensions, data_type, metric, connectivity, expansion_add, expansion_search, search_width) in iproduct!(
        &dimensions_sweep,
        &cli.data_type,
        &cli.metric,
        &cli.connectivity,
        &cli.expansion_add,
        &cli.expansion_search,
        &cli.search_width
    ) {
        state
            .check_dimensions(dimensions)
            .unwrap_or_bail("invalid --dimensions");

        let description = format!(
            "cuvs-cagra · {data_type} · {metric} · d={dimensions} · M={connectivity} · ef={expansion_add}/{expansion_search} · sw={search_width}"
        );

        let cagra_tuning = CagraTuning {
            build_algo: &cli.build_algo,
            search_width: *search_width,
            min_iterations: cli.min_iterations,
            max_iterations: cli.max_iterations,
            num_random_samplings: cli.num_random_samplings,
        };

        summary.record(run_config(
            &description,
            cli.common.index.as_deref(),
            || {
                CuvsBackend::new(
                    IndexConfig {
                        dimensions,
                        data_type,
                        metric,
                        connectivity: *connectivity,
                        expansion_add: *expansion_add,
                        expansion_search: *expansion_search,
                    },
                    cagra_tuning,
                )
            },
            |h| {
                CuvsBackend::load(
                    h,
                    IndexConfig {
                        dimensions,
                        data_type,
                        metric,
                        connectivity: *connectivity,
                        expansion_add: *expansion_add,
                        expansion_search: *expansion_search,
                    },
                    cagra_tuning,
                )
            },
            &mut state,
            dimensions,
        ));
    }

    summary.print();
}
