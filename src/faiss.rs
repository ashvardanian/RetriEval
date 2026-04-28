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
//!     --data_type f32,bf16,f16,i8 \
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
//!     --data_type b1 \
//!     --metric hamming \
//!     --output results/binary_1M
//! ```

use std::cell::UnsafeCell;
use std::collections::HashMap;

use clap::Parser;
use faiss::index::io::{read_index, write_index};
use faiss::Index as _;
use itertools::iproduct;
use retrieval::{bail, run_config, Backend, BenchState, CommonArgs, Distance, Key, SweepSummary, Vectors};
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
    data_type: Vec<String>,

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

fn index_factory_string(data_type: &str, connectivity: usize) -> Result<String, String> {
    // `IDMap,…` wraps the inner index so FAISS persists our keys natively
    // alongside the vectors — `add_with_ids` / `read_index` round-trip the
    // (key, vector) pairs without us shadowing them in a sidecar.
    // Binary HNSW has no `IDMapBinary` analogue exposed via faiss-sys 0.7,
    // so the binary path keeps the bare factory and a translation table.
    match data_type {
        "f32" => Ok(format!("IDMap,HNSW{connectivity},Flat")),
        "f16" => Ok(format!("IDMap,HNSW{connectivity},SQfp16")),
        "bf16" => Ok(format!("IDMap,HNSW{connectivity},SQbf16")),
        "u8" => Ok(format!("IDMap,HNSW{connectivity},SQ8_direct")),
        "i8" => Ok(format!("IDMap,HNSW{connectivity},SQ8_direct_signed")),
        "b1" => Ok(format!("BHNSW{connectivity}")),
        _ => Err(format!("unknown FAISS data_type: {data_type}")),
    }
}

fn metric_label_for(metric: faiss::MetricType) -> &'static str {
    match metric {
        faiss::MetricType::InnerProduct => "ip",
        faiss::MetricType::L2 => "l2",
    }
}

fn is_binary(data_type: &str) -> bool {
    data_type == "b1"
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
    /// Translation table from FAISS internal sequential ID → our Key. Only
    /// populated for binary indexes — float indexes use the `IDMap` factory
    /// prefix, so FAISS persists keys natively and search returns them in
    /// `result.labels` directly.
    binary_key_map: Option<Vec<Key>>,
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
        let dimensions = dimensions as u32;

        let (metric_label, index) = if binary {
            let index = faiss::index::index_binary_factory(dimensions, &factory)
                .map_err(|e| format!("failed to create FAISS binary index: {e}"))?;
            ("hamming", FaissIndex::Binary(UnsafeCell::new(index)))
        } else {
            let (label, faiss_metric) = parse_metric(metric_name)?;
            let index = faiss::index::index_factory(dimensions, &factory, faiss_metric)
                .map_err(|e| format!("failed to create FAISS index: {e}"))?;
            (label, FaissIndex::Float(UnsafeCell::new(index)))
        };

        // Apply efConstruction / efSearch via FAISS ParameterSpace. Binary HNSW has no dispatcher in
        // `faiss/AutoTune.cpp` so the call is a no-op there: `IndexBinaryHNSW` keeps its compile-time
        // defaults (40 / 16). The CLI still accepts any `--expansion-add` / `--expansion-search` values
        // for `--data_type b1`, but only the float HNSW path actually tunes them.
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
        metadata.insert("data_type".into(), json!(dtype_name));
        metadata.insert("metric".into(), json!(metric_label));
        metadata.insert("dimensions".into(), json!(dimensions));
        metadata.insert("connectivity".into(), json!(connectivity));
        metadata.insert("expansion_add".into(), json!(expansion_add));
        metadata.insert("expansion_search".into(), json!(expansion_search));
        metadata.insert("threads".into(), json!(threads));

        Ok(Self {
            index,
            binary_key_map: if binary { Some(Vec::new()) } else { None },
            description,
            metadata,
        })
    }

    /// Sibling of `new` for opening a previously-saved index. The FAISS file
    /// alone preserves dimensions, metric, and the index structure; build-time
    /// params (M, efConstruction, the data_type factory label) aren't reported on
    /// load because faiss-sys 0.7 doesn't bind any HNSW introspection symbols.
    /// Only the float (`IDMap,…`) path is supported — binary HNSW lacks
    /// `IDMapBinary` in faiss-sys, so its keys can't be persisted natively.
    pub fn load(handle: &str, expansion_search: usize, threads: usize) -> Result<Self, String> {
        unsafe {
            omp_set_num_threads(threads as i32);
        }

        let idx = read_index(handle).map_err(|e| {
            format!(
                "FAISS read_index({handle}): {e}\n  \
                 (binary `--data_type b1` indexes can't be loaded yet — \
                 faiss-sys 0.7 doesn't bind IndexBinaryIDMap)"
            )
        })?;
        let metric_type = idx.metric_type();
        let dimensions = idx.d();
        let index = FaissIndex::Float(UnsafeCell::new(idx));

        // efSearch is the only post-load tunable; binary HNSW would have no
        // dispatcher anyway and we already errored out for it.
        let inner_ptr = match &index {
            FaissIndex::Float(c) => unsafe { (*c.get()).inner_ptr() as *mut std::ffi::c_void },
            FaissIndex::Binary(_) => unreachable!(),
        };
        let parameter_string = format!("efSearch={expansion_search}");
        let mut parameter_space = FaissParameterSpace::new()?;
        parameter_space
            .set_index_parameters(inner_ptr, &parameter_string)
            .map_err(|e| format!("FAISS ParameterSpace rejected `{parameter_string}`: {e}"))?;

        let metric_label = metric_label_for(metric_type);
        let description = format!(
            "faiss · {metric_label} · d={dimensions} · ef=?/{expansion_search} · {threads} threads · loaded[{handle}]",
        );

        let mut metadata = HashMap::new();
        metadata.insert("backend".into(), json!("faiss"));
        metadata.insert("library_version".into(), json!(faiss_version()));
        metadata.insert("metric".into(), json!(metric_label));
        metadata.insert("dimensions".into(), json!(dimensions));
        metadata.insert("expansion_search".into(), json!(expansion_search));
        metadata.insert("threads".into(), json!(threads));
        metadata.insert("loaded_from".into(), json!(handle));

        Ok(Self {
            index,
            binary_key_map: None,
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
        // SAFETY: `add` has exclusive `&mut self` access.
        match &self.index {
            FaissIndex::Float(index) => {
                let data = vectors.data.to_f32();
                let xids: Vec<faiss::Idx> = keys.iter().map(|&k| faiss::Idx::new(k as u64)).collect();
                unsafe { &mut *index.get() }
                    .add_with_ids(&data, &xids)
                    .map_err(|e| format!("FAISS add_with_ids failed: {e}"))
            }
            FaissIndex::Binary(index) => {
                let data = match &vectors.data {
                    retrieval::VectorSlice::B1x8(bytes) => *bytes,
                    _ => return Err("FAISS binary index requires B1x8 data".into()),
                };
                // Binary path has no IDMap; remember keys in insertion order so
                // `search` can translate FAISS's 0..N internal IDs back.
                self.binary_key_map
                    .as_mut()
                    .expect("binary index must carry a key_map")
                    .extend_from_slice(keys);
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

                // IDMap stored our keys natively; labels already are user keys.
                for (query_idx, found_count) in out_counts.iter_mut().enumerate() {
                    let offset = query_idx * count;
                    let mut found = 0;
                    for rank in 0..count {
                        match result.labels[offset + rank].get() {
                            Some(key_u64) => {
                                out_keys[offset + rank] = key_u64 as Key;
                                out_distances[offset + rank] = result.distances[offset + rank];
                                found += 1;
                            }
                            None => {
                                out_keys[offset + rank] = Key::MAX;
                                out_distances[offset + rank] = Distance::INFINITY;
                            }
                        }
                    }
                    *found_count = found;
                }
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

                let key_map = self
                    .binary_key_map
                    .as_deref()
                    .ok_or("binary search requires a key_map populated by add()")?;
                unpack_search_results(
                    &result.labels,
                    &result.distances,
                    key_map,
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

    fn save(&self, handle: &str) -> Result<(), String> {
        match &self.index {
            FaissIndex::Float(c) => {
                let idx = unsafe { &*c.get() };
                write_index(idx, handle).map_err(|e| format!("FAISS write_index({handle}): {e}"))
            }
            FaissIndex::Binary(_) => Err("FAISS binary HNSW save is not supported — faiss-sys 0.7 doesn't bind \
                 IndexBinaryIDMap, so keys can't be persisted natively"
                .into()),
        }
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

    let mut state = BenchState::load(&cli.common).unwrap_or_else(|e| bail(&format!("{e}")));
    let dimensions_sweep = cli.common.dimensions_sweep(state.dimensions());

    cli.common.ensure_single_config(&[
        dimensions_sweep.len(),
        cli.data_type.len(),
        cli.metric.len(),
        cli.connectivity.len(),
        cli.expansion_add.len(),
        cli.expansion_search.len(),
    ]);

    let mut summary = SweepSummary::default();
    for (&dimensions, data_type, metric, &connectivity, &expansion_add, &expansion_search) in iproduct!(
        &dimensions_sweep,
        &cli.data_type,
        &cli.metric,
        &cli.connectivity,
        &cli.expansion_add,
        &cli.expansion_search
    ) {
        state
            .check_dimensions(dimensions)
            .unwrap_or_else(|e| bail(&format!("invalid --dimensions: {e}")));

        let description =
            format!("faiss · {data_type} · {metric} · d={dimensions} · M={connectivity} · ef={expansion_add}/{expansion_search}");

        summary.record(run_config(
            &description,
            cli.common.index.as_deref(),
            || {
                FaissBackend::new(
                    dimensions,
                    data_type,
                    metric,
                    connectivity,
                    expansion_add,
                    expansion_search,
                    cli.threads,
                )
            },
            |h| FaissBackend::load(h, expansion_search, cli.threads),
            &mut state,
            dimensions,
        ));
    }

    summary.print();
}
