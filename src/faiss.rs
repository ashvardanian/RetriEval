//! FAISS HNSW benchmark binary.
//!
//! ## Prerequisites
//!
//! Requires a C++ compiler, CMake, and BLAS (FAISS is built from bundled source):
//!
//! ```sh
//! # Ubuntu / Debian
//! sudo apt install cmake g++ libopenblas-dev
//! ```
//!
//! ## Build & Install
//!
//! ```sh
//! cargo install --path . --features faiss-static
//! ```
//!
//! This statically links a bundled copy of FAISS — no system `libfaiss` needed.
//! For dynamic linking against a pre-installed `libfaiss_c`, use `--features faiss-backend` instead.
//!
//! ## Examples
//!
//! ```sh
//! retri-eval-faiss \
//!     --vectors datasets/turing_10M/base.10M.fbin \
//!     --queries datasets/turing_10M/query.public.100K.fbin \
//!     --neighbors datasets/turing_10M/groundtruth.public.100K.ibin \
//!     --dtype f32,bf16,f16,i8 \
//!     --metric l2 \
//!     --output results/
//! ```
//!
//! Binary hamming-distance search via BinaryHNSW (1024-bit vectors in `.b1bin`):
//! ```sh
//! retri-eval-faiss \
//!     --vectors datasets/binary_1M/base.1M.b1bin \
//!     --queries datasets/binary_1M/query.10K.b1bin \
//!     --neighbors datasets/binary_1M/groundtruth.10K.ibin \
//!     --dtype b1 \
//!     --metric hamming \
//!     --output results/binary_1M
//! ```

use std::cell::UnsafeCell;
use std::collections::HashMap;

use clap::Parser;
use faiss::Index as _;
use itertools::iproduct;
use retrieval::{try_run_config, Backend, BenchState, CommonArgs, Distance, Key, SweepSummary, Vectors};
use serde_json::{json, Value};

extern "C" {
    fn omp_set_num_threads(num_threads: i32);
    fn faiss_ParameterSpace_new(space: *mut *mut std::ffi::c_void) -> i32;
    fn faiss_ParameterSpace_set_index_parameters(
        space: *const std::ffi::c_void,
        index: *mut std::ffi::c_void,
        params: *const std::ffi::c_char,
    ) -> i32;
    fn faiss_ParameterSpace_free(space: *mut std::ffi::c_void);
    fn faiss_get_last_error() -> *const std::ffi::c_char;
}

/// FAISS C-API convention: `0` means success, anything else means the thread-local error is populated.
#[inline]
fn faiss_call_succeeded(return_code: i32) -> bool {
    return_code == 0
}

/// Read the thread-local last-error message populated by `FAISS_TRY` in `faiss/c_api/macros_impl.h`.
fn faiss_last_error() -> Option<String> {
    // SAFETY: pointer is either null or into FAISS's thread-local buffer — we copy into an owned String.
    unsafe {
        let error_ptr = faiss_get_last_error();
        if error_ptr.is_null() {
            return None;
        }
        let message = std::ffi::CStr::from_ptr(error_ptr).to_string_lossy().into_owned();
        (!message.is_empty()).then_some(message)
    }
}

/// RAII wrapper over `faiss::ParameterSpace` — free on drop, no manual cleanup at every exit path.
struct FaissParameterSpace {
    handle: *mut std::ffi::c_void,
}

impl FaissParameterSpace {
    fn new() -> Result<Self, String> {
        let mut handle: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: `&mut handle` is a valid out-pointer for FAISS to write into.
        let return_code = unsafe { faiss_ParameterSpace_new(&mut handle) };
        if !faiss_call_succeeded(return_code) || handle.is_null() {
            return Err(
                faiss_last_error().unwrap_or_else(|| "faiss_ParameterSpace_new: null handle, no FAISS error".into())
            );
        }
        Ok(Self { handle })
    }

    /// Apply e.g. `"efConstruction=128 efSearch=64"` to `index`. Non-zero rc means unknown parameter or no
    /// dispatcher for the index type; the captured last-error distinguishes.
    fn set_index_parameters(&mut self, index: *mut std::ffi::c_void, parameters: &str) -> Result<(), String> {
        let parameters_cstring =
            std::ffi::CString::new(parameters).map_err(|e| format!("parameter string contained interior NUL: {e}"))?;
        // SAFETY: `self.handle` non-null (enforced by `new`); `index` and `parameters_cstring` live through the call.
        let return_code =
            unsafe { faiss_ParameterSpace_set_index_parameters(self.handle, index, parameters_cstring.as_ptr()) };
        if faiss_call_succeeded(return_code) {
            Ok(())
        } else {
            Err(faiss_last_error().unwrap_or_else(|| {
                format!("faiss_ParameterSpace_set_index_parameters: rc={return_code}, no FAISS error")
            }))
        }
    }
}

impl Drop for FaissParameterSpace {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: `handle` came from `faiss_ParameterSpace_new` and was never freed while we held it.
            unsafe { faiss_ParameterSpace_free(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "retri-eval-faiss", about = "Benchmark FAISS HNSW")]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,

    /// Comma-separated quantization types: f32, f16, bf16, u8, i8, b1
    #[arg(long, value_delimiter = ',', default_value = "bf16")]
    dtype: Vec<String>,

    /// Comma-separated distance metrics: ip, l2
    #[arg(long, value_delimiter = ',', default_value = "l2")]
    metric: Vec<String>,

    /// HNSW connectivity parameter (M), comma-separated for sweep
    #[arg(long, value_delimiter = ',', default_value = "32")]
    connectivity: Vec<usize>,

    /// HNSW expansion factor during indexing, comma-separated for sweep
    #[arg(long, value_delimiter = ',', default_value = "128")]
    expansion_add: Vec<usize>,

    /// HNSW expansion factor during search, comma-separated for sweep
    #[arg(long, value_delimiter = ',', default_value = "64")]
    expansion_search: Vec<usize>,

    /// Number of threads (sets OMP_NUM_THREADS)
    #[arg(long, default_value_t = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1))]
    threads: usize,
}

fn parse_metric(s: &str) -> Result<(&'static str, faiss::MetricType), String> {
    match s {
        "ip" => Ok(("ip", faiss::MetricType::InnerProduct)),
        "l2" | "l2sq" => Ok(("l2", faiss::MetricType::L2)),
        _ => Err(format!("unknown FAISS metric: {s}. FAISS HNSW supports: ip, l2")),
    }
}

fn index_factory_string(dtype: &str, connectivity: usize) -> Result<String, String> {
    match dtype {
        "f32" => Ok(format!("HNSW{connectivity},Flat")),
        "f16" => Ok(format!("HNSW{connectivity},SQfp16")),
        "bf16" => Ok(format!("HNSW{connectivity},SQbf16")),
        "u8" => Ok(format!("HNSW{connectivity},SQ8_direct")),
        "i8" => Ok(format!("HNSW{connectivity},SQ8_direct_signed")),
        "b1" => Ok(format!("BHNSW{connectivity}")),
        _ => Err(format!("unknown FAISS dtype: {dtype}")),
    }
}

fn is_binary(dtype: &str) -> bool {
    dtype == "b1"
}

/// Unpack FAISS search results into output buffers, translating FAISS internal
/// sequential IDs back to our keys via `key_map`.
fn unpack_search_results<D: Copy>(
    labels: &[faiss::Idx],
    distances: &[D],
    key_map: &[Key],
    count: usize,
    to_distance: impl Fn(D) -> Distance,
    out_keys: &mut [Key],
    out_distances: &mut [Distance],
    out_counts: &mut [usize],
) {
    for (query_idx, found_count) in out_counts.iter_mut().enumerate() {
        let offset = query_idx * count;
        let mut found = 0;
        for rank in 0..count {
            let faiss_id = labels[offset + rank];
            if let Some(internal_id) = faiss_id.get() {
                out_keys[offset + rank] = key_map[internal_id as usize];
                out_distances[offset + rank] = to_distance(distances[offset + rank]);
                found += 1;
            } else {
                out_keys[offset + rank] = Key::MAX;
                out_distances[offset + rank] = Distance::INFINITY;
            }
        }
        *found_count = found;
    }
}

enum FaissIndex {
    Float(UnsafeCell<faiss::index::IndexImpl>),
    Binary(UnsafeCell<faiss::index::BinaryIndexImpl>),
}

struct FaissBackend {
    index: FaissIndex,
    /// Maps FAISS internal sequential ID → our Key. FAISS assigns IDs 0, 1, 2, ...
    /// in insertion order, but our keys come from the (shuffled) dataset.
    key_map: Vec<Key>,
    description: String,
    metadata: HashMap<String, Value>,
}

// SAFETY: FAISS manages its own thread safety via OpenMP.
unsafe impl Send for FaissBackend {}
unsafe impl Sync for FaissBackend {}

impl FaissBackend {
    fn new(
        dimensions: usize,
        dtype_name: &str,
        metric_name: &str,
        connectivity: usize,
        expansion_add: usize,
        expansion_search: usize,
        threads: usize,
    ) -> Result<Self, String> {
        unsafe {
            omp_set_num_threads(threads as i32);
        }

        let factory = index_factory_string(dtype_name, connectivity)?;
        let binary = is_binary(dtype_name);

        // For binary indices, dimensions is already in bits (from .b1bin header).
        // For float indices, dimensions is the scalar count.
        let dim = dimensions as u32;

        let (metric_label, index) = if binary {
            let index = faiss::index::index_binary_factory(dim, &factory)
                .map_err(|e| format!("failed to create FAISS binary index: {e}"))?;
            ("hamming", FaissIndex::Binary(UnsafeCell::new(index)))
        } else {
            let (label, faiss_metric) = parse_metric(metric_name)?;
            let index = faiss::index::index_factory(dim, &factory, faiss_metric)
                .map_err(|e| format!("failed to create FAISS index: {e}"))?;
            (label, FaissIndex::Float(UnsafeCell::new(index)))
        };

        // Apply efConstruction / efSearch via FAISS ParameterSpace. Binary HNSW has no dispatcher in
        // `faiss/AutoTune.cpp` so the call is a no-op there: `IndexBinaryHNSW` keeps its compile-time
        // defaults (40 / 16). The CLI still accepts any `--expansion-add` / `--expansion-search` values
        // for `--dtype b1`, but only the float HNSW path actually tunes them.
        if !binary {
            let inner_index_ptr = match &index {
                FaissIndex::Float(cell) => unsafe { (*cell.get()).inner_ptr() as *mut std::ffi::c_void },
                FaissIndex::Binary(cell) => unsafe { (*cell.get()).inner_ptr() as *mut std::ffi::c_void },
            };
            let parameter_string = format!("efConstruction={expansion_add} efSearch={expansion_search}");
            let mut parameter_space = FaissParameterSpace::new()?;
            parameter_space
                .set_index_parameters(inner_index_ptr, &parameter_string)
                .map_err(|e| format!("FAISS ParameterSpace rejected `{parameter_string}`: {e}"))?;
        }

        let description = format!(
            "faiss · {dtype_name} · {metric_label} · M={connectivity} · \
             ef={expansion_add}/{expansion_search} · {threads} threads",
        );

        let mut metadata = HashMap::new();
        metadata.insert("backend".into(), json!("faiss"));
        metadata.insert("library_version".into(), json!(faiss_version()));
        metadata.insert("dtype".into(), json!(dtype_name));
        metadata.insert("metric".into(), json!(metric_label));
        metadata.insert("connectivity".into(), json!(connectivity));
        metadata.insert("expansion_add".into(), json!(expansion_add));
        metadata.insert("expansion_search".into(), json!(expansion_search));
        metadata.insert("threads".into(), json!(threads));

        Ok(Self {
            index,
            key_map: Vec::new(),
            description,
            metadata,
        })
    }
}

impl Backend for FaissBackend {
    fn description(&self) -> String {
        self.description.clone()
    }
    fn metadata(&self) -> HashMap<String, Value> {
        self.metadata.clone()
    }

    fn add(&mut self, keys: &[Key], vectors: Vectors) -> Result<(), String> {
        // Record keys in insertion order — FAISS assigns internal IDs 0, 1, 2, ...
        self.key_map.extend_from_slice(keys);

        // SAFETY: `add` has exclusive `&mut self` access.
        match &self.index {
            FaissIndex::Float(index) => {
                let data = vectors.data.to_f32();
                unsafe { &mut *index.get() }
                    .add(&data)
                    .map_err(|e| format!("FAISS add failed: {e}"))
            }
            FaissIndex::Binary(index) => {
                let data = match &vectors.data {
                    retrieval::VectorSlice::B1x8(bytes) => *bytes,
                    _ => return Err("FAISS binary index requires B1x8 data".into()),
                };
                unsafe { &mut *index.get() }
                    .add(data)
                    .map_err(|e| format!("FAISS binary add failed: {e}"))
            }
        }
    }

    fn search(
        &self,
        queries: Vectors,
        count: usize,
        out_keys: &mut [Key],
        out_distances: &mut [Distance],
        out_counts: &mut [usize],
    ) -> Result<(), String> {
        // SAFETY: `run` never calls `search` and `add` concurrently; search is
        // the only reader and FAISS is internally thread-safe via OpenMP.
        match &self.index {
            FaissIndex::Float(index) => {
                let data = queries.data.to_f32();

                let result = unsafe { &mut *index.get() }
                    .search(&data, count)
                    .map_err(|e| format!("FAISS search failed: {e}"))?;

                unpack_search_results(
                    &result.labels,
                    &result.distances,
                    &self.key_map,
                    count,
                    |d| d,
                    out_keys,
                    out_distances,
                    out_counts,
                );
                Ok(())
            }
            FaissIndex::Binary(index) => {
                let data = match &queries.data {
                    retrieval::VectorSlice::B1x8(bytes) => *bytes,
                    _ => return Err("FAISS binary index requires B1x8 data".into()),
                };

                let result = unsafe { &mut *index.get() }
                    .search(data, count)
                    .map_err(|e| format!("FAISS binary search failed: {e}"))?;

                unpack_search_results(
                    &result.labels,
                    &result.distances,
                    &self.key_map,
                    count,
                    |d| d as Distance,
                    out_keys,
                    out_distances,
                    out_counts,
                );
                Ok(())
            }
        }
    }

    fn memory_bytes(&self) -> usize {
        retrieval::process_rss_bytes() as usize
    }
}

/// FAISS version string. FAISS's C API only exposes `faiss_get_version()` — no
/// ISA introspection (the Python `has_AVX512*` flags come from numpy's CPU
/// probe, not from FAISS). We don't duplicate that guess here.
fn faiss_version() -> String {
    use std::ffi::CStr;
    extern "C" {
        fn faiss_get_version() -> *const std::os::raw::c_char;
    }
    unsafe {
        let p = faiss_get_version();
        if p.is_null() {
            "unknown".to_string()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

fn main() {
    let cli = Cli::parse();

    eprintln!("faiss v{}", faiss_version());

    let mut state = BenchState::load(&cli.common).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    let dimensions = state.dimensions();

    let mut summary = SweepSummary::default();
    for (dtype, metric, &connectivity, &expansion_add, &expansion_search) in iproduct!(
        &cli.dtype,
        &cli.metric,
        &cli.connectivity,
        &cli.expansion_add,
        &cli.expansion_search
    ) {
        let description =
            format!("faiss · {dtype} · {metric} · M={connectivity} · ef={expansion_add}/{expansion_search}");
        let backend = FaissBackend::new(
            dimensions,
            dtype,
            metric,
            connectivity,
            expansion_add,
            expansion_search,
            cli.threads,
        );
        summary.record(try_run_config(&description, backend, &mut state));
    }

    summary.print();
}
