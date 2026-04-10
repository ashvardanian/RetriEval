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
//! cargo build --release --features cuvs-backend
//!
//! ./target/release/retri-eval-cuvs \
//!     --vectors datasets/turing_10M/base.10M.fbin \
//!     --queries datasets/turing_10M/query.public.100K.fbin \
//!     --neighbors datasets/turing_10M/groundtruth.public.100K.ibin \
//!     --dtype f32,f16 --metric l2 \
//!     --output results/turing_10M
//! ```

use std::cell::Cell;
use std::cell::UnsafeCell;
use std::ptr::NonNull;

use clap::Parser;
use cuvs::distance_type::DistanceType;
use itertools::iproduct;
use retrieval::{run, Backend, BenchState, CommonArgs, Distance, Key, Vectors};

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
            let _ =
                cuvs_sys::cuvsRMMFree(self.0, ptr.as_ptr() as *mut std::ffi::c_void, layout.size());
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
    dtype: cuvs_sys::DLDataType,
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
            dtype,
            shape,
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        },
        manager_ctx: std::ptr::null_mut(),
        deleter: None,
    }
}

/// Transmute a stack-local `DLManagedTensor` into a `cuvs::ManagedTensor`.
///
/// `ManagedTensor` is a transparent newtype around `DLManagedTensor`.
/// Our descriptors have `deleter = None`, so their `Drop` is a no-op.
unsafe fn as_managed(raw: cuvs_sys::DLManagedTensor) -> cuvs::ManagedTensor {
    std::mem::transmute(raw)
}

// #region Dtype

/// Supported CAGRA scalar quantization types (matches the C API).
#[derive(Debug, Clone, Copy)]
enum CuvsDtype {
    F32,
    F16,
    I8,
    U8,
}

impl CuvsDtype {
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "f32" => Ok(Self::F32),
            "f16" => Ok(Self::F16),
            "i8" => Ok(Self::I8),
            "u8" => Ok(Self::U8),
            _ => Err(format!(
                "unknown CAGRA dtype: {s}. supported: f32, f16, i8, u8"
            )),
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

    /// Convert f32 values to the target dtype, appending raw bytes to `output`.
    fn convert_from_f32(self, source: &[f32], output: &mut Vec<u8>) {
        match self {
            Self::F32 => {
                let bytes = unsafe {
                    std::slice::from_raw_parts(source.as_ptr() as *const u8, source.len() * 4)
                };
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

// #region GpuQueries

/// Typed GPU query buffer, matching the `VectorSlice` enum pattern.
enum GpuQueries {
    F32(GpuTensor<f32>),
    F16(GpuTensor<numkong::f16>),
    I8(GpuTensor<i8>),
    U8(GpuTensor<u8>),
}

impl GpuQueries {
    fn allocate(
        dtype: CuvsDtype,
        shape: &[usize],
        allocator: CudaAllocator,
    ) -> Result<Self, String> {
        let error = |e| format!("GPU query alloc failed: {e}");
        unsafe {
            Ok(match dtype {
                CuvsDtype::F32 => Self::F32(GpuTensor::try_empty_in(shape, allocator).map_err(error)?),
                CuvsDtype::F16 => Self::F16(GpuTensor::try_empty_in(shape, allocator).map_err(error)?),
                CuvsDtype::I8 => Self::I8(GpuTensor::try_empty_in(shape, allocator).map_err(error)?),
                CuvsDtype::U8 => Self::U8(GpuTensor::try_empty_in(shape, allocator).map_err(error)?),
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

    fn byte_size(&self) -> usize {
        match self {
            Self::F32(tensor) => tensor.numel() * 4,
            Self::F16(tensor) => tensor.numel() * 2,
            Self::I8(tensor) => tensor.numel(),
            Self::U8(tensor) => tensor.numel(),
        }
    }
}

// #region SearchBuffers

/// Pre-allocated GPU + host buffers for search, reused across calls.
struct SearchBuffers {
    queries: GpuQueries,
    neighbors: GpuTensor<Key>,
    distances: GpuTensor<Distance>,

    queries_shape: [i64; 2],
    neighbors_shape: [i64; 2],
    distances_shape: [i64; 2],

    neighbors_host: Vec<Key>,
    distances_host: Vec<Distance>,

    /// Reusable host-side buffer for dtype conversion before H2D copy.
    query_staging: Vec<u8>,
}

impl SearchBuffers {
    fn allocate(
        allocator: CudaAllocator,
        num_queries: usize,
        dimensions: usize,
        neighbor_count: usize,
        dtype: CuvsDtype,
    ) -> Result<Self, String> {
        let error = |e| format!("GPU alloc failed: {e}");
        let queries =
            GpuQueries::allocate(dtype, &[num_queries, dimensions], allocator.clone())?;
        let neighbors = unsafe {
            GpuTensor::<Key>::try_empty_in(&[num_queries, neighbor_count], allocator.clone())
                .map_err(error)?
        };
        let distances = unsafe {
            GpuTensor::<Distance>::try_empty_in(&[num_queries, neighbor_count], allocator)
                .map_err(error)?
        };
        Ok(Self {
            queries,
            neighbors,
            distances,
            queries_shape: [num_queries as i64, dimensions as i64],
            neighbors_shape: [num_queries as i64, neighbor_count as i64],
            distances_shape: [num_queries as i64, neighbor_count as i64],
            neighbors_host: vec![Key::default(); num_queries * neighbor_count],
            distances_host: vec![Distance::default(); num_queries * neighbor_count],
            query_staging: Vec::new(),
        })
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
    dtype: Vec<String>,

    /// Distance metric: l2, ip, cos
    #[arg(long, value_delimiter = ',', default_value = "l2")]
    metric: Vec<String>,

    /// CAGRA output graph degree (analogous to HNSW M)
    #[arg(long, value_delimiter = ',', default_value = "32")]
    graph_degree: Vec<usize>,

    /// CAGRA intermediate graph degree before pruning (analogous to expansion_add)
    #[arg(long, value_delimiter = ',', default_value = "64")]
    intermediate_graph_degree: Vec<usize>,

    /// Intermediate search results retained during search (analogous to expansion_search).
    /// Higher values improve recall at the cost of speed.
    #[arg(long, value_delimiter = ',', default_value = "64")]
    itopk_size: Vec<usize>,
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

// #region Backend

pub struct CuvsBackend {
    // GPU resources that depend on `res` are declared BEFORE `res`
    // so they drop first (Rust drops fields in declaration order).
    search_buffers: UnsafeCell<Option<SearchBuffers>>,
    index: UnsafeCell<Option<cuvs::cagra::Index>>,

    res: cuvs::Resources,
    cuda_alloc: CudaAllocator,
    dimensions: usize,
    metric: DistanceType,
    dtype: CuvsDtype,
    graph_degree: usize,
    intermediate_graph_degree: usize,
    itopk_size: usize,

    host_vectors: Vec<u8>,
    host_keys: Vec<Key>,
    dirty: Cell<bool>,

    description: String,
    metadata: std::collections::HashMap<String, serde_json::Value>,
}

unsafe impl Send for CuvsBackend {}
unsafe impl Sync for CuvsBackend {}

impl CuvsBackend {
    pub fn new(
        dimensions: usize,
        dtype_name: &str,
        metric_name: &str,
        graph_degree: usize,
        intermediate_graph_degree: usize,
        itopk_size: usize,
    ) -> Result<Self, String> {
        let metric = parse_metric(metric_name)?;
        let dtype = CuvsDtype::from_str(dtype_name)?;
        let res =
            cuvs::Resources::new().map_err(|e| format!("failed to create cuVS resources: {e}"))?;
        let cuda_alloc = CudaAllocator(res.0);

        let description = format!(
            "cuvs-cagra \u{b7} {} \u{b7} {metric_name} \u{b7} gd={graph_degree} \u{b7} igd={intermediate_graph_degree} \u{b7} itopk={itopk_size}",
            dtype.as_str(),
        );

        use serde_json::json;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("backend".into(), json!("cuvs-cagra"));
        metadata.insert("dtype".into(), json!(dtype.as_str()));
        metadata.insert("metric".into(), json!(metric_label(metric_name)));
        metadata.insert("graph_degree".into(), json!(graph_degree));
        metadata.insert(
            "intermediate_graph_degree".into(),
            json!(intermediate_graph_degree),
        );
        metadata.insert("itopk_size".into(), json!(itopk_size));

        Ok(Self {
            search_buffers: UnsafeCell::new(None),
            index: UnsafeCell::new(None),
            res,
            cuda_alloc,
            dimensions,
            metric,
            dtype,
            graph_degree,
            intermediate_graph_degree,
            itopk_size,
            host_vectors: Vec::new(),
            host_keys: Vec::new(),
            dirty: Cell::new(false),
            description,
            metadata,
        })
    }

    /// Build (or rebuild) the CAGRA index from accumulated host buffers.
    fn build_index(&self) -> Result<(), String> {
        let num_vectors = self.host_keys.len();
        let dimensions = self.dimensions;
        let mut shape = [num_vectors as i64, dimensions as i64];

        let host_dl = unsafe {
            dl_tensor(
                self.host_vectors.as_ptr() as *mut _,
                shape.as_mut_ptr(),
                2,
                self.dtype.dl_type(),
                false,
            )
        };

        let build_params = cuvs::cagra::IndexParams::new()
            .map_err(|e| format!("failed to create CAGRA index params: {e}"))?
            .set_graph_degree(self.graph_degree)
            .set_intermediate_graph_degree(self.intermediate_graph_degree);
        unsafe { (*build_params.0).metric = self.metric };

        let managed = unsafe { as_managed(host_dl) };
        let index = cuvs::cagra::Index::build(&self.res, &build_params, managed)
            .map_err(|e| format!("cuVS CAGRA build failed: {e}"))?;

        unsafe { *self.index.get() = Some(index) };
        self.dirty.set(false);
        Ok(())
    }

    /// Copy device tensor to a pre-allocated host slice, synchronising the stream.
    unsafe fn device_to_host<T>(&self, device_ptr: *const T, host: &mut [T]) -> Result<(), String> {
        let bytes = host.len() * std::mem::size_of::<T>();
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
    unsafe fn host_to_device(
        &self,
        host: &[u8],
        device_ptr: *mut std::ffi::c_void,
    ) -> Result<(), String> {
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
        self.dtype
            .convert_from_f32(&f32_data, &mut self.host_vectors);
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
        let dimensions = queries.dimensions;
        let neighbor_count = count;

        // Lazily allocate GPU search buffers; reused on subsequent calls.
        let buffers = unsafe { &mut *self.search_buffers.get() };
        if buffers.is_none() {
            *buffers = Some(SearchBuffers::allocate(
                self.cuda_alloc.clone(),
                num_queries,
                dimensions,
                neighbor_count,
                self.dtype,
            )?);
        }
        let buffers = buffers.as_mut().unwrap();

        // Convert queries to target dtype on the reusable staging buffer.
        buffers.query_staging.clear();
        let query_f32 = queries.data.to_f32();
        self.dtype
            .convert_from_f32(&query_f32, &mut buffers.query_staging);

        // H2D copy into pre-allocated GPU tensor.
        unsafe { self.host_to_device(&buffers.query_staging, buffers.queries.as_mut_ptr())? };

        // Build non-owning DLPack descriptors over the GpuTensor memory.
        let queries_tensor = unsafe {
            dl_tensor(
                buffers.queries.as_mut_ptr(),
                buffers.queries_shape.as_mut_ptr(),
                2,
                self.dtype.dl_type(),
                true,
            )
        };
        let neighbors_tensor = unsafe {
            dl_tensor(
                buffers.neighbors.as_mut_ptr() as _,
                buffers.neighbors_shape.as_mut_ptr(),
                2,
                dl_key(),
                true,
            )
        };
        let distances_tensor = unsafe {
            dl_tensor(
                buffers.distances.as_mut_ptr() as _,
                buffers.distances_shape.as_mut_ptr(),
                2,
                dl_distance(),
                true,
            )
        };

        // CAGRA requires itopk_size >= the number of neighbors requested.
        let effective_itopk = self.itopk_size.max(neighbor_count);
        let search_params = cuvs::cagra::SearchParams::new()
            .map_err(|e| format!("search params: {e}"))?
            .set_itopk_size(effective_itopk);

        let index = unsafe { &*self.index.get() }
            .as_ref()
            .ok_or("index not built")?;

        // Wrap as ManagedTensor for the safe search API (deleter=None, Drop is no-op).
        let queries_managed = unsafe { as_managed(queries_tensor) };
        let neighbors_managed = unsafe { as_managed(neighbors_tensor) };
        let distances_managed = unsafe { as_managed(distances_tensor) };

        index
            .search(
                &self.res,
                &search_params,
                &queries_managed,
                &neighbors_managed,
                &distances_managed,
            )
            .map_err(|e| format!("cuVS CAGRA search failed: {e}"))?;

        // Prevent Drop from touching our GpuTensor-owned memory.
        std::mem::forget(queries_managed);
        std::mem::forget(neighbors_managed);
        std::mem::forget(distances_managed);

        // D2H copy results.
        unsafe {
            self.device_to_host(buffers.neighbors.as_ptr(), &mut buffers.neighbors_host)?;
            self.device_to_host(buffers.distances.as_ptr(), &mut buffers.distances_host)?;
        }

        // Map CAGRA 0-based indices back to original keys.
        for query_idx in 0..num_queries {
            let offset = query_idx * neighbor_count;
            let mut found = 0;
            for rank in 0..neighbor_count {
                let neighbor_idx = buffers.neighbors_host[offset + rank] as usize;
                if neighbor_idx < self.host_keys.len() {
                    out_keys[offset + rank] = self.host_keys[neighbor_idx];
                    out_distances[offset + rank] = buffers.distances_host[offset + rank];
                    found += 1;
                } else {
                    out_keys[offset + rank] = Key::MAX;
                    out_distances[offset + rank] = Distance::INFINITY;
                }
            }
            out_counts[query_idx] = found;
        }

        Ok(())
    }

    fn memory_bytes(&self) -> usize {
        let num_vectors = self.host_keys.len();
        let host_bytes = self.host_vectors.len() + num_vectors * std::mem::size_of::<Key>();
        let gpu_bytes = num_vectors * self.dimensions * self.dtype.bytes_per_element()
            + num_vectors * self.graph_degree * std::mem::size_of::<u32>();
        host_bytes + gpu_bytes
    }
}

// #region main

fn main() {
    let cli = Cli::parse();

    let mut state = BenchState::load(&cli.common).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    let dimensions = state.dimensions();

    for (dtype, metric, graph_degree, intermediate_graph_degree, itopk_size) in iproduct!(
        &cli.dtype,
        &cli.metric,
        &cli.graph_degree,
        &cli.intermediate_graph_degree,
        &cli.itopk_size
    ) {
        let mut backend = CuvsBackend::new(
            dimensions,
            dtype,
            metric,
            *graph_degree,
            *intermediate_graph_degree,
            *itopk_size,
        )
        .unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(1);
        });

        run(&mut backend, &mut state).unwrap_or_else(|e| {
            eprintln!("Benchmark failed: {e}");
            std::process::exit(1);
        });
    }

    eprintln!("Benchmark complete.");
}
