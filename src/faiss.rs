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
//!     --base-vectors datasets/turing_10M/base.10M.fbin \
//!     --query-vectors datasets/turing_10M/query.public.100K.fbin \
//!     --query-neighbors datasets/turing_10M/groundtruth.public.100K.ibin \
//!     --data-type f32,bf16,f16,i8 \
//!     --metric l2 \
//!     --output results/
//! ```
//!
//! Binary hamming-distance search via BinaryHNSW (1024-bit vectors in `.b1bin`):
//! ```sh
//! retri-eval-faiss \
//!     --base-vectors datasets/binary_1M/base.1M.b1bin \
//!     --query-vectors datasets/binary_1M/query.10K.b1bin \
//!     --query-neighbors datasets/binary_1M/groundtruth.10K.ibin \
//!     --data-type b1 \
//!     --output results/binary_1M
//! ```

use std::{collections::HashMap, fmt};

use clap::Parser;
use faiss::{
    index::io::{read_index, write_index},
    Index as _,
};
use itertools::iproduct;
use serde_json::{json, Value};

use retrieval::{
    run_config, spell_list, Backend, BenchState, CommonArgs, Distance, IndexConfig, Key, SweepSummary, Threads,
    UnwrapOrBail, Vectors,
};

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

    /// Quantization types (comma-separated for sweep)
    #[arg(long, value_delimiter = ',', value_enum, default_value = "bf16")]
    data_type: Vec<DataType>,

    /// Distance metrics (comma-separated for sweep); binary data always uses Hamming
    #[arg(long, value_delimiter = ',', value_enum, default_value = "l2")]
    metric: Vec<Metric>,

    /// HNSW connectivity parameter (M), comma-separated for sweep
    #[arg(long, value_delimiter = ',', default_value = "32", value_parser = retrieval::parse_count_flag)]
    connectivity: Vec<usize>,

    /// HNSW expansion factor during indexing, comma-separated for sweep
    #[arg(long, value_delimiter = ',', default_value = "128", value_parser = retrieval::parse_count_flag)]
    expansion_add: Vec<usize>,

    /// HNSW expansion factor during search, comma-separated for sweep
    #[arg(long, value_delimiter = ',', default_value = "64", value_parser = retrieval::parse_count_flag)]
    expansion_search: Vec<usize>,

    /// Number of threads (sets OMP_NUM_THREADS)
    #[arg(long, default_value = "0", value_parser = retrieval::parse_threads_flag)]
    threads: Threads,
}

/// FAISS storage types, as `--data-type` spells them.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum DataType {
    F32,
    F16,
    Bf16,
    U8,
    I8,
    B1,
}

impl fmt::Display for DataType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        retrieval::spell_value(self, formatter)
    }
}

/// FAISS HNSW metrics, as `--metric` spells them.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum Metric {
    Ip,
    #[value(alias = "l2sq")]
    L2,
}

impl From<Metric> for faiss::MetricType {
    fn from(metric: Metric) -> Self {
        match metric {
            Metric::Ip => Self::InnerProduct,
            Metric::L2 => Self::L2,
        }
    }
}

impl fmt::Display for Metric {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        retrieval::spell_value(self, formatter)
    }
}

fn index_factory_string(data_type: DataType, connectivity: usize) -> String {
    // `IDMap,…` wraps the inner index so FAISS persists our keys natively
    // alongside the vectors — `add_with_ids` / `read_index` round-trip the
    // (key, vector) pairs without us shadowing them in a sidecar.
    // Binary HNSW has no `IDMapBinary` analogue exposed via faiss-sys 0.7,
    // so the binary path keeps the bare factory and a translation table.
    match data_type {
        DataType::F32 => format!("IDMap,HNSW{connectivity},Flat"),
        DataType::F16 => format!("IDMap,HNSW{connectivity},SQfp16"),
        DataType::Bf16 => format!("IDMap,HNSW{connectivity},SQbf16"),
        DataType::U8 => format!("IDMap,HNSW{connectivity},SQ8_direct"),
        DataType::I8 => format!("IDMap,HNSW{connectivity},SQ8_direct_signed"),
        DataType::B1 => format!("BHNSW{connectivity}"),
    }
}

fn metric_label_for(metric: faiss::MetricType) -> &'static str {
    match metric {
        faiss::MetricType::InnerProduct => "ip",
        faiss::MetricType::L2 => "l2",
    }
}

/// Buffers one query batch's results, translating FAISS's internal sequential
/// IDs back to our keys. `key_map` is empty on the `IDMap` paths, where FAISS
/// already returns the caller's keys.
struct SearchResults<'a, D> {
    labels: &'a [faiss::Idx],
    distances: &'a [D],
    key_map: &'a [Key],
    count: usize,
}

fn unpack_search_results<D: Copy>(
    results: SearchResults<'_, D>,
    to_distance: impl Fn(D) -> Distance,
    out_keys: &mut [Key],
    out_distances: &mut [Distance],
    out_counts: &mut [usize],
) {
    let SearchResults {
        labels,
        distances,
        key_map,
        count,
    } = results;
    for (query_index, found_count) in out_counts.iter_mut().enumerate() {
        let offset = query_index * count;
        let hits = (0..count).filter_map(|rank| {
            let internal_id = labels[offset + rank].get()?;
            let key = match key_map.is_empty() {
                true => internal_id as Key,
                false => key_map[internal_id as usize],
            };
            Some((key, to_distance(distances[offset + rank])))
        });
        *found_count = retrieval::write_row(
            hits,
            &mut out_keys[offset..offset + count],
            &mut out_distances[offset..offset + count],
        );
    }
}

enum FaissIndex {
    Float(faiss::index::IndexImpl),
    Binary(faiss::index::BinaryIndexImpl),
}

struct FaissBackend {
    scratch: Vec<f32, std::alloc::System>,
    ids: Vec<faiss::Idx, std::alloc::System>,
    index: FaissIndex,
    /// Translation table from FAISS internal sequential ID → our Key. Only
    /// populated for binary indexes — float indexes use the `IDMap` factory
    /// prefix, so FAISS persists keys natively and search returns them in
    /// `result.labels` directly.
    binary_key_map: Option<Vec<Key, std::alloc::System>>,
    description: String,
    metadata: HashMap<String, Value>,
}

impl FaissBackend {
    fn new(config: IndexConfig<DataType, Metric>, threads: Threads) -> Result<Self, String> {
        let IndexConfig {
            dimensions,
            data_type,
            metric,
            connectivity,
            expansion_add,
            expansion_search,
        } = config;

        unsafe {
            omp_set_num_threads(threads.0.get() as i32);
        }

        let factory = index_factory_string(data_type, connectivity);
        let binary = data_type == DataType::B1;

        // For binary indices, dimensions is already in bits (from .b1bin header).
        // For float indices, dimensions is the scalar count.
        let dimensions = dimensions as u32;

        let (metric_label, index) = if binary {
            let index = faiss::index::index_binary_factory(dimensions, &factory)
                .map_err(|e| format!("failed to create FAISS binary index: {e}"))?;
            ("hamming", FaissIndex::Binary(index))
        } else {
            let index = faiss::index::index_factory(dimensions, &factory, metric.into())
                .map_err(|e| format!("failed to create FAISS index: {e}"))?;
            (metric_label_for(metric.into()), FaissIndex::Float(index))
        };

        // Apply efConstruction / efSearch via FAISS ParameterSpace. Binary HNSW has no dispatcher in
        // `faiss/AutoTune.cpp` so the call is a no-op there: `IndexBinaryHNSW` keeps its compile-time
        // defaults (40 / 16). The CLI still accepts any `--expansion-add` / `--expansion-search` values
        // for `--data-type b1`, but only the float HNSW path actually tunes them.
        if !binary {
            let inner_index_ptr = match &index {
                FaissIndex::Float(cell) => cell.inner_ptr() as *mut std::ffi::c_void,
                FaissIndex::Binary(cell) => cell.inner_ptr() as *mut std::ffi::c_void,
            };
            let parameter_string = format!("efConstruction={expansion_add} efSearch={expansion_search}");
            let mut parameter_space = FaissParameterSpace::new()?;
            parameter_space
                .set_index_parameters(inner_index_ptr, &parameter_string)
                .map_err(|e| format!("FAISS ParameterSpace rejected `{parameter_string}`: {e}"))?;
        }

        let description = format!(
            "faiss · {data_type} · {metric_label} · M={connectivity} · \
             ef={expansion_add}/{expansion_search} · {threads} threads",
        );

        let mut metadata = HashMap::new();
        metadata.insert("backend".into(), json!("faiss"));
        metadata.insert("library_version".into(), json!(faiss_version()));
        metadata.insert("data_type".into(), json!(data_type.to_string()));
        metadata.insert("metric".into(), json!(metric_label));
        metadata.insert("connectivity".into(), json!(connectivity));
        metadata.insert("expansion_add".into(), json!(expansion_add));
        metadata.insert("expansion_search".into(), json!(expansion_search));
        metadata.insert("threads".into(), json!(threads.0.get()));

        Ok(Self {
            scratch: Vec::new_in(std::alloc::System),
            ids: Vec::new_in(std::alloc::System),
            index,
            binary_key_map: if binary {
                Some(Vec::new_in(std::alloc::System))
            } else {
                None
            },
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
    pub fn load(handle: &str, expansion_search: usize, threads: Threads) -> Result<Self, String> {
        unsafe {
            omp_set_num_threads(threads.0.get() as i32);
        }

        let idx = read_index(handle).map_err(|e| {
            format!(
                "FAISS read_index({handle}): {e}\n  \
                 (binary `--data-type b1` indexes can't be loaded yet — \
                 faiss-sys 0.7 doesn't bind IndexBinaryIDMap)"
            )
        })?;
        let metric_type = idx.metric_type();
        let dimensions = idx.d();
        let index = FaissIndex::Float(idx);

        // efSearch is the only post-load tunable; binary HNSW would have no
        // dispatcher anyway and we already errored out for it.
        let inner_ptr = match &index {
            FaissIndex::Float(c) => c.inner_ptr() as *mut std::ffi::c_void,
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
        metadata.insert("expansion_search".into(), json!(expansion_search));
        metadata.insert("threads".into(), json!(threads.0.get()));
        metadata.insert("loaded_from".into(), json!(handle));
        // faiss-sys binds no HNSW introspection, so these are genuinely unknown
        // on the load path. Emitted as null rather than omitted: absent would
        // read as "this backend has no such knob".
        metadata.insert("data_type".into(), Value::Null);
        metadata.insert("connectivity".into(), Value::Null);
        metadata.insert("expansion_add".into(), Value::Null);

        Ok(Self {
            scratch: Vec::new_in(std::alloc::System),
            ids: Vec::new_in(std::alloc::System),
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
        match &mut self.index {
            FaissIndex::Float(index) => {
                let data = vectors.data.to_f32_in(&mut self.scratch)?;
                self.ids.clear();
                self.ids.extend(keys.iter().map(|&k| faiss::Idx::new(k as u64)));
                index
                    .add_with_ids(data, &self.ids)
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
                index.add(data).map_err(|e| format!("FAISS binary add failed: {e}"))
            }
        }
    }

    fn search(
        &mut self,
        queries: Vectors,
        count: usize,
        out_keys: &mut [Key],
        out_distances: &mut [Distance],
        out_counts: &mut [usize],
    ) -> Result<(), String> {
        match &mut self.index {
            FaissIndex::Float(index) => {
                let data = queries.data.to_f32_in(&mut self.scratch)?;

                let result = index
                    .search(data, count)
                    .map_err(|e| format!("FAISS search failed: {e}"))?;

                // IDMap stored our keys natively, so an empty `key_map` means
                // "labels already are user keys".
                unpack_search_results(
                    SearchResults {
                        labels: &result.labels,
                        distances: &result.distances,
                        key_map: &[],
                        count,
                    },
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

                let result = index
                    .search(data, count)
                    .map_err(|e| format!("FAISS binary search failed: {e}"))?;

                let key_map = self
                    .binary_key_map
                    .as_deref()
                    .ok_or("binary search requires a key_map populated by add()")?;
                unpack_search_results(
                    SearchResults {
                        labels: &result.labels,
                        distances: &result.distances,
                        key_map,
                        count,
                    },
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
            FaissIndex::Float(index) => {
                write_index(index, handle).map_err(|e| format!("FAISS write_index({handle}): {e}"))
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
    let cli: Cli = retrieval::parse_cli();

    eprintln!("FAISS {}", faiss_version());

    let mut state = BenchState::load(&cli.common).unwrap_or_bail("benchmark state");
    eprintln!("- Data types: {}", spell_list(&cli.data_type));
    eprintln!("- Metrics: {}", spell_list(&cli.metric));
    eprintln!("- Connectivity: {}", spell_list(&cli.connectivity));
    eprintln!("- Expansion add: {}", spell_list(&cli.expansion_add));
    eprintln!("- Expansion search: {}", spell_list(&cli.expansion_search));
    eprintln!("- Threads: {}", cli.threads);
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
    for (&dimensions, &data_type, &metric, &connectivity, &expansion_add, &expansion_search) in iproduct!(
        &dimensions_sweep,
        &cli.data_type,
        &cli.metric,
        &cli.connectivity,
        &cli.expansion_add,
        &cli.expansion_search
    ) {
        state.check_dimensions(dimensions).unwrap_or_bail("invalid --dims");

        let description =
            format!("faiss · {data_type} · {metric} · d={dimensions} · M={connectivity} · ef={expansion_add}/{expansion_search}");

        summary.record(run_config(
            &description,
            cli.common.index.as_deref(),
            || {
                FaissBackend::new(
                    IndexConfig {
                        dimensions,
                        data_type,
                        metric,
                        connectivity,
                        expansion_add,
                        expansion_search,
                    },
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

#[cfg(test)]
mod tests {
    use super::*;
    use retrieval::VectorSlice;

    #[test]
    fn repeated_search_reuses_conversion_storage_and_preserves_keys() {
        let config = IndexConfig {
            dimensions: 2,
            data_type: DataType::F32,
            metric: Metric::L2,
            connectivity: 8,
            expansion_add: 32,
            expansion_search: 16,
        };
        let mut backend = FaissBackend::new(config, Threads::ONE).unwrap();
        backend
            .add(
                &[42, 7],
                Vectors {
                    data: VectorSlice::I8(&[3, 4, 0, 0]),
                    dimensions: 2,
                },
            )
            .unwrap();
        let scratch = backend.scratch.as_ptr();
        let mut keys = [0; 3];
        let mut distances = [0.0; 3];
        let mut counts = [0];
        for _ in 0..2 {
            backend
                .search(
                    Vectors {
                        data: VectorSlice::I8(&[0, 0]),
                        dimensions: 2,
                    },
                    3,
                    &mut keys,
                    &mut distances,
                    &mut counts,
                )
                .unwrap();
            assert_eq!(keys, [7, 42, Key::MAX]);
            assert_eq!(distances, [0.0, 25.0, Distance::INFINITY]);
            assert_eq!(counts, [2]);
            assert_eq!(scratch, backend.scratch.as_ptr());
        }
    }
}
