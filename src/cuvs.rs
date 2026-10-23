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
//! # See: https://docs.nvidia.com/datascience/install/
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

use std::{fmt, ptr::NonNull};

use clap::Parser;
use cuvs::distance_type::DistanceType;
use itertools::iproduct;
use numkong as nk;
use serde_json::json;

use retrieval::{
    run_config, spell_list, Backend, BatchKeys, BenchState, CastScratch, CommonArgs, Distance, Hits, IndexConfig,
    IndexFile, SweepSummary, UnwrapOrBail, Vectors,
};

// #region CudaAllocator

/// RMM-backed CUDA device memory allocator for NumKong tensors.
///
/// GPU-backed tensors must only use `as_ptr()` / `as_mut_ptr()` — host-side
/// accessors like `as_slice()` would dereference GPU pointers from the CPU.
#[derive(Clone)]
struct CudaAllocator(cuvs_sys::cuvsResources_t);

unsafe impl nk::Allocator for CudaAllocator {
    fn allocate(&self, layout: std::alloc::Layout) -> Result<NonNull<[u8]>, nk::AllocError> {
        if layout.size() == 0 {
            let dangling = NonNull::new(std::ptr::without_provenance_mut(layout.align())).ok_or(nk::AllocError)?;
            return Ok(NonNull::slice_from_raw_parts(dangling, 0));
        }
        let mut pointer: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: RMM either writes a device pointer to `layout.size()` bytes or reports the failure.
        let status = unsafe { cuvs_sys::cuvsRMMAlloc(self.0, &mut pointer, layout.size()) };
        if status != cuvs_sys::cuvsError_t::CUVS_SUCCESS {
            return Err(nk::AllocError);
        }
        let pointer = NonNull::new(pointer.cast::<u8>()).ok_or(nk::AllocError)?;
        Ok(NonNull::slice_from_raw_parts(pointer, layout.size()))
    }

    unsafe fn deallocate(&self, pointer: NonNull<u8>, layout: std::alloc::Layout) {
        if layout.size() > 0 {
            // SAFETY: `pointer` came from `cuvsRMMAlloc` on these resources with this size.
            let _ = unsafe { cuvs_sys::cuvsRMMFree(self.0, pointer.as_ptr().cast(), layout.size()) };
        }
    }
}

type GpuTensor<T> = nk::Tensor<T, CudaAllocator>;

// #region DLPack

/// Build a non-owning `DLManagedTensor` descriptor.
///
/// The caller must keep the underlying data and `shape` alive for the
/// lifetime of the returned struct.  `deleter` is always `None` — ownership
/// stays with the caller (either a host buffer or a `GpuTensor`).
fn dl_tensor(
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

/// Supported CAGRA scalar quantization types (matches the C API), as `--data-type` spells them.
#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum CuvsDataType {
    F32,
    F16,
    I8,
    U8,
}

impl CuvsDataType {
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
// Built on cuvs-sys, since `cuvs::cagra::Index` hides its pointer and exposes no serialize or deserialize.

struct CagraIndex(cuvs_sys::cuvsCagraIndex_t);

fn cuvs_check(status: cuvs_sys::cuvsError_t, context: &str) -> Result<(), String> {
    if status == cuvs_sys::cuvsError_t::CUVS_SUCCESS {
        Ok(())
    } else {
        Err(format!("{context}: cuvsError_t = {status:?}"))
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
        resources: &cuvs::Resources,
        params: cuvs_sys::cuvsCagraIndexParams_t,
        dataset_dl: *mut cuvs_sys::DLManagedTensor,
    ) -> Result<(), String> {
        unsafe {
            cuvs_check(
                cuvs_sys::cuvsCagraBuild(resources.0, params, dataset_dl, self.0),
                "cuvsCagraBuild",
            )
        }
    }

    /// Search the index. All three tensors must reference live memory
    /// (typically GPU buffers wrapped in `DLManagedTensor` views).
    fn search(
        &self,
        resources: &cuvs::Resources,
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
                cuvs_sys::cuvsCagraSearch(resources.0, params, self.0, queries, neighbors, distances, prefilter),
                "cuvsCagraSearch",
            )
        }
    }

    /// Saves the graph with its dataset, so a load needs nothing else.
    fn serialize(&self, resources: &cuvs::Resources, path: &str) -> Result<(), String> {
        let path_cstring = std::ffi::CString::new(path).map_err(|error| format!("path contains NUL: {error}"))?;
        unsafe {
            cuvs_check(
                cuvs_sys::cuvsCagraSerialize(resources.0, path_cstring.as_ptr(), self.0, true),
                "cuvsCagraSerialize",
            )
        }
    }

    fn deserialize(&self, resources: &cuvs::Resources, path: &str) -> Result<(), String> {
        let path_cstring = std::ffi::CString::new(path).map_err(|error| format!("path contains NUL: {error}"))?;
        unsafe {
            cuvs_check(
                cuvs_sys::cuvsCagraDeserialize(resources.0, path_cstring.as_ptr(), self.0),
                "cuvsCagraDeserialize",
            )
        }
    }
}

impl Drop for CagraIndex {
    fn drop(&mut self) {
        unsafe {
            // Drop cannot report a failed destroy.
            let _ = cuvs_sys::cuvsCagraIndexDestroy(self.0);
        }
    }
}

/// Path of the host-rows sidecar. CAGRA's serializer persists the device-side dataset and graph
/// but knows nothing about which base row each of its sequential ids came from, so we write the rows
/// ourselves next to the index file; keys follow from rows at search time.
fn rows_sidecar_path(handle: &str) -> String {
    format!("{handle}.rows")
}

/// Sidecar layout: `magic | graph_degree: u64 | count: u64 | rows: u32...`, all little-endian.
/// Widths are fixed rather than `usize` because this is a file read back on whatever machine opens
/// the index, and rows fit `u32` because CAGRA's own ids do. The magic makes any older sidecar a
/// loud error instead of one that parses into nonsense.
const ROWS_SIDECAR_MAGIC: &[u8; 8] = b"RETRIRW1";
const ROWS_SIDECAR_HEADER_BYTES: usize = 24;

/// `graph_degree` rides along because CAGRA's deserializer cannot report the
/// build parameters, and `memory_bytes()` needs the degree to size the graph.
fn write_host_rows(path: &str, rows: &[u32], graph_degree: usize) -> Result<(), String> {
    let mut bytes = Vec::with_capacity_in(
        ROWS_SIDECAR_HEADER_BYTES + std::mem::size_of_val(rows),
        std::alloc::System,
    );
    bytes.extend_from_slice(ROWS_SIDECAR_MAGIC);
    bytes.extend_from_slice(&(graph_degree as u64).to_le_bytes());
    bytes.extend_from_slice(&(rows.len() as u64).to_le_bytes());
    for &row in rows {
        bytes.extend_from_slice(&row.to_le_bytes());
    }
    std::fs::write(path, &bytes).map_err(|error| format!("write {path}: {error}"))
}

fn read_host_rows(path: &str) -> Result<(Vec<u32, std::alloc::System>, usize), String> {
    let bytes = std::fs::read(path).map_err(|error| format!("read {path}: {error}"))?;
    if bytes.len() < ROWS_SIDECAR_HEADER_BYTES || &bytes[0..8] != ROWS_SIDECAR_MAGIC {
        return Err(format!("{path}: not a row sidecar of this version, rebuild the index"));
    }
    let graph_degree = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    let count = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
    let body = &bytes[ROWS_SIDECAR_HEADER_BYTES..];
    if body.len() < count * 4 {
        return Err(format!(
            "{path}: expected {} bytes, got {}",
            ROWS_SIDECAR_HEADER_BYTES + count * 4,
            bytes.len()
        ));
    }
    let mut rows = Vec::with_capacity_in(count, std::alloc::System);
    rows.extend(
        body.chunks_exact(4)
            .take(count)
            .map(|row| u32::from_le_bytes(row.try_into().unwrap())),
    );
    Ok((rows, graph_degree))
}

// #region GpuQueries

/// Typed GPU query buffer, matching the `VectorSlice` enum pattern.
enum GpuQueries {
    F32(GpuTensor<f32>),
    F16(GpuTensor<nk::f16>),
    I8(GpuTensor<i8>),
    U8(GpuTensor<u8>),
}

impl GpuQueries {
    fn allocate(data_type: CuvsDataType, shape: &[usize], allocator: CudaAllocator) -> Result<Self, String> {
        let error = |error| format!("GPU query alloc failed: {error}");
        // SAFETY: device memory is only written by the H2D copies and cuVS, never read from the host.
        unsafe {
            Ok(match data_type {
                CuvsDataType::F32 => Self::F32(GpuTensor::uninitialized_in(shape, allocator).map_err(error)?),
                CuvsDataType::F16 => Self::F16(GpuTensor::uninitialized_in(shape, allocator).map_err(error)?),
                CuvsDataType::I8 => Self::I8(GpuTensor::uninitialized_in(shape, allocator).map_err(error)?),
                CuvsDataType::U8 => Self::U8(GpuTensor::uninitialized_in(shape, allocator).map_err(error)?),
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
    neighbors: GpuTensor<u32>,
    distances: GpuTensor<Distance>,

    /// Rows the buffers were sized for. Batches at or below this reuse them.
    capacity_queries: usize,
    /// Neighbors per query the buffers — and `search_params` — were sized for.
    neighbor_count: usize,

    queries_shape: [i64; 2],
    /// The neighbors' and distances' shared `[queries, neighbor_count]` shape.
    results_shape: [i64; 2],

    neighbors_host: Vec<u32, std::alloc::System>,
    distances_host: Vec<Distance, std::alloc::System>,

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
        let error = |error| format!("GPU alloc failed: {error}");
        let queries = GpuQueries::allocate(data_type, &[capacity_queries, dimensions], allocator.clone())?;
        let mut neighbors_host = Vec::with_capacity_in(capacity_queries * neighbor_count, std::alloc::System);
        neighbors_host.resize(capacity_queries * neighbor_count, 0);
        let mut distances_host = Vec::with_capacity_in(capacity_queries * neighbor_count, std::alloc::System);
        distances_host.resize(capacity_queries * neighbor_count, Distance::default());
        let neighbors = unsafe {
            GpuTensor::<u32>::uninitialized_in(&[capacity_queries, neighbor_count], allocator.clone()).map_err(error)?
        };
        let distances = unsafe {
            GpuTensor::<Distance>::uninitialized_in(&[capacity_queries, neighbor_count], allocator).map_err(error)?
        };
        Ok(Self {
            queries,
            neighbors,
            distances,
            capacity_queries,
            neighbor_count,
            queries_shape: [capacity_queries as i64, dimensions as i64],
            results_shape: [capacity_queries as i64, neighbor_count as i64],
            neighbors_host,
            distances_host,
            search_params,
        })
    }

    /// Whether this allocation can serve a `query_count` × `neighbor_count` batch.
    fn fits(&self, query_count: usize, neighbor_count: usize) -> bool {
        query_count <= self.capacity_queries && neighbor_count == self.neighbor_count
    }

    /// Point the DLPack shapes at the current batch, leaving the allocation
    /// alone. Without this a trailing partial batch would keep the previous
    /// batch's row count and make CAGRA search stale rows.
    fn set_batch_rows(&mut self, query_count: usize) {
        self.queries_shape[0] = query_count as i64;
        self.results_shape[0] = query_count as i64;
    }
}

// #region CLI

#[derive(Parser, Debug)]
#[command(name = "retri-eval-cuvs", about = "Benchmark cuVS CAGRA (GPU)")]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,

    /// Quantization types (comma-separated for sweep)
    #[arg(
        long,
        env = "RETRIEVAL_DATA_TYPE",
        value_delimiter = ',',
        value_enum,
        default_value = "f32"
    )]
    data_type: Vec<CuvsDataType>,

    /// Distance metrics (comma-separated for sweep)
    #[arg(
        long,
        env = "RETRIEVAL_METRIC",
        value_delimiter = ',',
        value_enum,
        default_value = "l2"
    )]
    metric: Vec<Metric>,

    /// CAGRA output graph degree — the shared-vocabulary name for HNSW's M.
    #[arg(long, env = "RETRIEVAL_CONNECTIVITY", value_delimiter = ',', default_value = "32", value_parser = retrieval::parse_count_flag)]
    connectivity: Vec<usize>,

    /// CAGRA intermediate graph degree before pruning — analogous to
    /// `ef_construction`.
    #[arg(long, env = "RETRIEVAL_EXPANSION_ADD", value_delimiter = ',', default_value = "64", value_parser = retrieval::parse_count_flag)]
    expansion_add: Vec<usize>,

    /// CAGRA internal top-i list retained during search — analogous to
    /// `ef_search`. Higher values improve recall at the cost of speed.
    #[arg(long, env = "RETRIEVAL_EXPANSION_SEARCH", value_delimiter = ',', default_value = "64", value_parser = retrieval::parse_count_flag)]
    expansion_search: Vec<usize>,

    /// Graph nodes used as starting points per search iteration, or `auto`
    /// for cuVS's choice (comma-separated for sweep). Higher values improve recall.
    #[arg(long, env = "RETRIEVAL_SEARCH_WIDTH", value_delimiter = ',', default_value = "auto", value_parser = parse_count_or_auto)]
    search_width: Vec<Option<usize>>,

    /// Minimum search iterations, preventing early termination; cuVS chooses when unset.
    #[arg(long, env = "RETRIEVAL_MIN_ITERATIONS", value_parser = retrieval::parse_count_flag)]
    min_iterations: Option<usize>,

    /// Maximum search iterations; cuVS chooses when unset.
    #[arg(long, env = "RETRIEVAL_MAX_ITERATIONS", value_parser = retrieval::parse_count_flag)]
    max_iterations: Option<usize>,

    /// Random sampling rounds for the initial search points; cuVS chooses when unset.
    #[arg(long, env = "RETRIEVAL_NUM_RANDOM_SAMPLINGS", value_parser = |text: &str| retrieval::parse_count_flag(text).and_then(|count| u32::try_from(count).map_err(|_| "expected a positive count".into())))]
    num_random_samplings: Option<u32>,

    /// Graph build algorithm.
    #[arg(long, env = "RETRIEVAL_BUILD_ALGO", value_enum, default_value = "auto")]
    build_algo: BuildAlgo,

    /// Persisted index: loaded and searched if the path exists, else built and saved; needs a single config
    #[arg(long, env = "RETRIEVAL_INDEX", value_parser = clap::builder::NonEmptyStringValueParser::new())]
    index: Option<String>,
}

/// Parses a positive count, or `auto` as `None` for cuVS's own choice.
fn parse_count_or_auto(text: &str) -> Result<Option<usize>, String> {
    match text {
        "auto" => Ok(None),
        _ => retrieval::parse_count(text)
            .map(Some)
            .ok_or_else(|| "expected a positive count or auto".into()),
    }
}

/// Spells a count that cuVS chooses when unset, as `auto`.
fn spell_count_or_auto<T: fmt::Display>(value: Option<T>) -> String {
    value.map_or_else(|| "auto".into(), |count| count.to_string())
}

impl From<CuvsDataType> for CastScratch {
    fn from(data_type: CuvsDataType) -> Self {
        match data_type {
            CuvsDataType::F32 => Self::F32(Vec::new_in(std::alloc::System)),
            CuvsDataType::F16 => Self::F16(Vec::new_in(std::alloc::System)),
            CuvsDataType::I8 => Self::I8(Vec::new_in(std::alloc::System)),
            CuvsDataType::U8 => Self::U8(Vec::new_in(std::alloc::System)),
        }
    }
}

// #region Metric

/// CAGRA metrics, as `--metric` spells them.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    #[value(alias = "l2sq")]
    L2,
    Ip,
    Cos,
}

impl From<Metric> for DistanceType {
    fn from(metric: Metric) -> Self {
        match metric {
            Metric::L2 => Self::L2Expanded,
            Metric::Ip => Self::InnerProduct,
            Metric::Cos => Self::CosineExpanded,
        }
    }
}

impl fmt::Display for Metric {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        retrieval::spell_value(self, formatter)
    }
}

impl fmt::Display for CuvsDataType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        retrieval::spell_value(self, formatter)
    }
}

/// CAGRA graph build algorithms, as `--build-algo` spells them.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuildAlgo {
    Auto,
    #[value(name = "nn_descent")]
    NnDescent,
}

impl From<BuildAlgo> for cuvs_sys::cuvsCagraGraphBuildAlgo {
    fn from(build_algo: BuildAlgo) -> Self {
        match build_algo {
            BuildAlgo::Auto => Self::AUTO_SELECT,
            BuildAlgo::NnDescent => Self::NN_DESCENT,
        }
    }
}

impl fmt::Display for BuildAlgo {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        retrieval::spell_value(self, formatter)
    }
}

// #region Backend

pub struct CuvsBackend {
    /// Input cast to `data_type` on the host, before it is appended or copied to the device.
    cast: CastScratch,
    // GPU resources that depend on `resources` are declared BEFORE it, so they drop first.
    search_buffers: Option<SearchBuffers>,
    /// The built graph, or `None` until the next search builds it from the host buffers.
    index: Option<CagraIndex>,

    resources: cuvs::Resources,
    cuda_allocator: CudaAllocator,
    dimensions: usize,
    metric: DistanceType,
    data_type: CuvsDataType,
    connectivity: usize,
    expansion_add: usize,
    build_algo: cuvs_sys::cuvsCagraGraphBuildAlgo,
    expansion_search: usize,
    search_width: Option<usize>,
    min_iterations: Option<usize>,
    max_iterations: Option<usize>,
    num_random_samplings: Option<u32>,

    host_vectors: Vec<u8, std::alloc::System>,
    host_rows: Vec<u32, std::alloc::System>,

    metadata: std::collections::HashMap<String, serde_json::Value>,
}

unsafe impl Send for CuvsBackend {}

/// CAGRA knobs with no counterpart in the shared vocabulary; `None` leaves the choice to cuVS.
#[derive(Clone, Copy)]
pub struct CagraTuning {
    pub build_algo: BuildAlgo,
    pub search_width: Option<usize>,
    pub min_iterations: Option<usize>,
    pub max_iterations: Option<usize>,
    pub num_random_samplings: Option<u32>,
}

impl CuvsBackend {
    pub fn new(config: IndexConfig<CuvsDataType, Metric>, tuning: CagraTuning) -> Result<Self, String> {
        let IndexConfig {
            dimensions,
            data_type,
            metric,
            connectivity,
            expansion_add,
            expansion_search,
        } = config;
        let CagraTuning {
            build_algo,
            search_width,
            min_iterations,
            max_iterations,
            num_random_samplings,
        } = tuning;
        let resources = cuvs::Resources::new().map_err(|error| format!("failed to create cuVS resources: {error}"))?;
        let cuda_allocator = CudaAllocator(resources.0);
        let metadata = std::collections::HashMap::from([
            ("backend".into(), json!("cuvs-cagra")),
            ("data_type".into(), json!(data_type.to_string())),
            ("metric".into(), json!(metric.to_string())),
            ("connectivity".into(), json!(connectivity)),
            ("expansion_add".into(), json!(expansion_add)),
            ("expansion_search".into(), json!(expansion_search)),
            // cuVS's own `0` for "choose for me", which keeps result hashes from before `auto`.
            ("search_width".into(), json!(search_width.unwrap_or(0))),
            ("build_algo".into(), json!(build_algo.to_string())),
            ("min_iterations".into(), json!(min_iterations.unwrap_or(0))),
            ("max_iterations".into(), json!(max_iterations.unwrap_or(0))),
            ("num_random_samplings".into(), json!(num_random_samplings.unwrap_or(0))),
        ]);
        Ok(Self {
            cast: data_type.into(),
            search_buffers: None,
            index: None,
            resources,
            cuda_allocator,
            dimensions,
            metric: metric.into(),
            data_type,
            connectivity,
            expansion_add,
            build_algo: build_algo.into(),
            expansion_search,
            search_width,
            min_iterations,
            max_iterations,
            num_random_samplings,
            host_vectors: Vec::new_in(std::alloc::System),
            host_rows: Vec::new_in(std::alloc::System),
            metadata,
        })
    }

    /// Open a saved CAGRA index. CAGRA's deserializer reports no build params, so the graph degree
    /// comes from the row sidecar written alongside the index, and the other build-time settings
    /// are reported as null.
    pub fn load(path: &str, config: IndexConfig<CuvsDataType, Metric>, tuning: CagraTuning) -> Result<Self, String> {
        let mut backend = Self::new(config, tuning)?;
        let index = CagraIndex::new()?;
        index.deserialize(&backend.resources, path)?;
        let (host_rows, connectivity) = read_host_rows(&rows_sidecar_path(path))?;
        backend.index = Some(index);
        backend.host_rows = host_rows;
        backend.connectivity = connectivity;
        backend.metadata.insert("connectivity".into(), json!(connectivity));
        backend.metadata.insert("loaded_from".into(), json!(path));
        for unknown in [
            "expansion_add",
            "build_algo",
            "min_iterations",
            "max_iterations",
            "num_random_samplings",
        ] {
            backend.metadata.insert(unknown.into(), serde_json::Value::Null);
        }
        Ok(backend)
    }

    /// Create search params once, reused across all search calls.
    fn build_search_params(&self, neighbor_count: usize) -> Result<cuvs::cagra::SearchParams, String> {
        let effective_itopk = self.expansion_search.max(neighbor_count);
        let params = cuvs::cagra::SearchParams::new()
            .map_err(|error| format!("search params: {error}"))?
            .set_itopk_size(effective_itopk);
        unsafe {
            let raw = params.0;
            if let Some(search_width) = self.search_width {
                (*raw).search_width = search_width;
            }
            if let Some(min_iterations) = self.min_iterations {
                (*raw).min_iterations = min_iterations;
            }
            if let Some(max_iterations) = self.max_iterations {
                (*raw).max_iterations = max_iterations;
            }
            if let Some(num_random_samplings) = self.num_random_samplings {
                (*raw).num_random_samplings = num_random_samplings;
            }
        }
        Ok(params)
    }

    /// Build the CAGRA index from the accumulated host buffers.
    fn build_index(&mut self) -> Result<(), String> {
        let vector_count = self.host_rows.len();
        let dimensions = self.dimensions;
        let mut shape = [vector_count as i64, dimensions as i64];

        let mut host_dl = dl_tensor(
            self.host_vectors.as_ptr() as *mut _,
            shape.as_mut_ptr(),
            2,
            self.data_type.dl_type(),
            false,
        );

        let build_params = cuvs::cagra::IndexParams::new()
            .map_err(|error| format!("failed to create CAGRA index params: {error}"))?
            .set_graph_degree(self.connectivity)
            .set_intermediate_graph_degree(self.expansion_add)
            .set_build_algo(self.build_algo);
        unsafe { (*build_params.0).metric = self.metric };

        let index = CagraIndex::new()?;
        index.build(&self.resources, build_params.0, &mut host_dl)?;
        self.index = Some(index);
        Ok(())
    }

    /// Copy device tensor to a pre-allocated host slice, synchronising the stream.
    unsafe fn device_to_host<T>(
        resources: &cuvs::Resources,
        device_ptr: *const T,
        host: &mut [T],
    ) -> Result<(), String> {
        let bytes = std::mem::size_of_val(host);
        let stream = resources.get_cuda_stream().map_err(|error| format!("{error}"))?;
        let status = cuvs_sys::cudaMemcpyAsync(
            host.as_mut_ptr() as *mut _,
            device_ptr as *const _,
            bytes,
            cuvs_sys::cudaMemcpyKind_cudaMemcpyDefault,
            stream,
        );
        if status != cuvs_sys::cudaError::cudaSuccess {
            return Err(format!("cudaMemcpyAsync D2H failed: {status:?}"));
        }
        resources.sync_stream().map_err(|error| format!("{error}"))
    }

    /// Copy host bytes to a pre-allocated device pointer.
    unsafe fn host_to_device(
        resources: &cuvs::Resources,
        host: &[u8],
        device_ptr: *mut std::ffi::c_void,
    ) -> Result<(), String> {
        let stream = resources.get_cuda_stream().map_err(|error| format!("{error}"))?;
        let status = cuvs_sys::cudaMemcpyAsync(
            device_ptr,
            host.as_ptr() as *const _,
            host.len(),
            cuvs_sys::cudaMemcpyKind_cudaMemcpyDefault,
            stream,
        );
        if status != cuvs_sys::cudaError::cudaSuccess {
            return Err(format!("cudaMemcpyAsync H2D failed: {status:?}"));
        }
        Ok(())
    }
}

impl Backend for CuvsBackend {
    fn metadata(&self) -> std::collections::HashMap<String, serde_json::Value> {
        self.metadata.clone()
    }

    fn add(&mut self, keys: BatchKeys<'_>, vectors: Vectors) -> Result<(), String> {
        for row in keys.rows() {
            self.host_rows
                .push(u32::try_from(row).map_err(|_| "CAGRA rows end at 2^32".to_string())?);
        }
        let bytes = self.cast.encode(&vectors)?;
        self.host_vectors.extend_from_slice(bytes);
        self.index = None;
        Ok(())
    }

    fn search(&mut self, queries: Vectors, mut hits: Hits<'_>) -> Result<(), String> {
        if self.index.is_none() {
            self.build_index()?;
        }
        let query_count = queries.len();
        let count = hits.count();

        // Buffers only grow and the old ones free first, so alternating batches never thrash scarce device memory.
        if !self
            .search_buffers
            .as_ref()
            .is_some_and(|buffers| buffers.fits(query_count, count))
        {
            let search_params = self.build_search_params(count)?;
            let capacity = query_count.max(
                self.search_buffers
                    .as_ref()
                    .map_or(0, |buffers| buffers.capacity_queries),
            );
            self.search_buffers = None;
            self.search_buffers = Some(SearchBuffers::allocate(
                self.cuda_allocator.clone(),
                capacity,
                queries.dimensions(),
                count,
                self.data_type,
                search_params,
            )?);
        }
        let Some(buffers) = self.search_buffers.as_mut() else {
            unreachable!("the buffers were just allocated")
        };
        buffers.set_batch_rows(query_count);

        // Queries already in `data_type` upload straight from the source rows.
        let bytes = self.cast.encode(&queries)?;
        unsafe { Self::host_to_device(&self.resources, bytes, buffers.queries.as_mut_ptr())? };

        let mut queries_tensor = dl_tensor(
            buffers.queries.as_mut_ptr(),
            buffers.queries_shape.as_mut_ptr(),
            2,
            self.data_type.dl_type(),
            true,
        );
        let mut neighbors_tensor = dl_tensor(
            buffers.neighbors.as_mut_ptr() as _,
            buffers.results_shape.as_mut_ptr(),
            2,
            dl_key(),
            true,
        );
        let mut distances_tensor = dl_tensor(
            buffers.distances.as_mut_ptr() as _,
            buffers.results_shape.as_mut_ptr(),
            2,
            dl_distance(),
            true,
        );
        let index = self.index.as_ref().ok_or("index not built")?;
        index.search(
            &self.resources,
            buffers.search_params.0,
            &mut queries_tensor,
            &mut neighbors_tensor,
            &mut distances_tensor,
        )?;

        // Only the rows this batch filled, a prefix of the possibly larger capacity.
        let filled = query_count * count;
        unsafe {
            Self::device_to_host(
                &self.resources,
                buffers.neighbors.as_ptr(),
                &mut buffers.neighbors_host[..filled],
            )?;
            Self::device_to_host(
                &self.resources,
                buffers.distances.as_ptr(),
                &mut buffers.distances_host[..filled],
            )?;
        }

        // CAGRA answers with 0-based row indices; out-of-range ones are its tail padding.
        let answers = buffers.neighbors_host[..filled]
            .chunks_exact(count)
            .zip(buffers.distances_host[..filled].chunks_exact(count));
        for ((neighbors, distances), hit_row) in answers.zip(hits.rows()) {
            hit_row.write_rows(neighbors.iter().zip(distances).filter_map(|(&neighbor, &distance)| {
                let row = *self.host_rows.get(neighbor as usize)?;
                Some(Ok((row as usize, distance)))
            }))?;
        }
        Ok(())
    }

    /// The host-row sidecar has one entry per indexed vector, on both the
    /// freshly-built and the deserialized path.
    fn indexed_count(&self) -> Option<usize> {
        Some(self.host_rows.len())
    }

    fn memory_bytes(&self) -> usize {
        let vector_count = self.host_rows.len();
        let host_bytes = self.host_vectors.len() + std::mem::size_of_val(self.host_rows.as_slice());
        let gpu_bytes = vector_count * self.dimensions * self.data_type.bytes_per_element()
            + vector_count * self.connectivity * std::mem::size_of::<u32>();
        host_bytes + gpu_bytes
    }

    fn save(&self, path: &str) -> Result<(), String> {
        let index = self.index.as_ref().ok_or("CAGRA index not built — nothing to save")?;
        index.serialize(&self.resources, path)?;
        write_host_rows(&rows_sidecar_path(path), &self.host_rows, self.connectivity)
    }
}

// #region main

fn main() {
    let cli: Cli = retrieval::parse_cli();

    let mut state = BenchState::load(&cli.common).unwrap_or_bail("benchmark state");
    eprintln!("- Data types: {}", spell_list(&cli.data_type));
    eprintln!("- Metrics: {}", spell_list(&cli.metric));
    eprintln!("- Connectivity: {}", spell_list(&cli.connectivity));
    eprintln!("- Expansion add: {}", spell_list(&cli.expansion_add));
    eprintln!("- Expansion search: {}", spell_list(&cli.expansion_search));
    eprintln!(
        "- Search width: {}",
        spell_list(cli.search_width.iter().copied().map(spell_count_or_auto))
    );
    eprintln!("- Min iterations: {}", spell_count_or_auto(cli.min_iterations));
    eprintln!("- Max iterations: {}", spell_count_or_auto(cli.max_iterations));
    eprintln!("- Random samplings: {}", spell_count_or_auto(cli.num_random_samplings));
    eprintln!("- Build algorithm: {}", cli.build_algo);
    let dimensions_sweep = cli.common.dimensions_sweep(state.dimensions());
    let configs = iproduct!(
        &dimensions_sweep,
        &cli.data_type,
        &cli.metric,
        &cli.connectivity,
        &cli.expansion_add,
        &cli.expansion_search,
        &cli.search_width
    );
    let index_file = IndexFile::new(cli.index.as_deref(), configs.clone().count());
    eprintln!("- Index: {index_file}");

    let mut summary = SweepSummary::default();
    for (&dimensions, &data_type, &metric, &connectivity, &expansion_add, &expansion_search, &search_width) in configs {
        let description = format!(
            "cuvs-cagra · {data_type} · {metric} · d={dimensions} · M={connectivity} · ef={expansion_add}/{expansion_search} · sw={}",
            spell_count_or_auto(search_width),
        );
        let config = IndexConfig {
            dimensions,
            data_type,
            metric,
            connectivity,
            expansion_add,
            expansion_search,
        };
        let tuning = CagraTuning {
            build_algo: cli.build_algo,
            search_width,
            min_iterations: cli.min_iterations,
            max_iterations: cli.max_iterations,
            num_random_samplings: cli.num_random_samplings,
        };
        let backend = match index_file {
            IndexFile::Load(path) => CuvsBackend::load(path, config, tuning),
            IndexFile::None | IndexFile::Save(_) => CuvsBackend::new(config, tuning),
        };
        summary.record(run_config(&description, index_file, backend, &mut state, dimensions));
    }

    summary.finish();
}
