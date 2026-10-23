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
    run_config, spell_list, Backend, BenchState, CommonArgs, Distance, Hits, IndexConfig, IndexFile, Key, SweepSummary,
    Threads, UnwrapOrBail, VectorSlice, Vectors,
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
        if return_code != 0 || handle.is_null() {
            return Err(
                faiss_last_error().unwrap_or_else(|| "faiss_ParameterSpace_new: null handle, no FAISS error".into())
            );
        }
        Ok(Self { handle })
    }

    /// Apply e.g. `"efConstruction=128 efSearch=64"` to `index`. Non-zero rc means unknown parameter or no
    /// dispatcher for the index type; the captured last-error distinguishes.
    fn set_index_parameters(&mut self, index: *mut std::ffi::c_void, parameters: &str) -> Result<(), String> {
        let parameters_cstring = std::ffi::CString::new(parameters)
            .map_err(|error| format!("parameter string contained interior NUL: {error}"))?;
        // SAFETY: `self.handle` non-null (enforced by `new`); `index` and `parameters_cstring` live through the call.
        let return_code =
            unsafe { faiss_ParameterSpace_set_index_parameters(self.handle, index, parameters_cstring.as_ptr()) };
        match return_code {
            0 => Ok(()),
            _ => Err(faiss_last_error().unwrap_or_else(|| {
                format!("faiss_ParameterSpace_set_index_parameters: rc={return_code}, no FAISS error")
            })),
        }
    }
}

impl Drop for FaissParameterSpace {
    fn drop(&mut self) {
        // SAFETY: `handle` came non-null from `faiss_ParameterSpace_new` and is freed only here.
        unsafe { faiss_ParameterSpace_free(self.handle) };
    }
}

#[derive(Parser, Debug)]
#[command(name = "retri-eval-faiss", about = "Benchmark FAISS HNSW")]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,

    /// Quantization types (comma-separated for sweep)
    #[arg(
        long,
        env = "RETRIEVAL_DATA_TYPE",
        value_delimiter = ',',
        value_enum,
        default_value = "bf16"
    )]
    data_type: Vec<DataType>,

    /// Distance metrics (comma-separated for sweep); binary data always uses Hamming
    #[arg(
        long,
        env = "RETRIEVAL_METRIC",
        value_delimiter = ',',
        value_enum,
        default_value = "l2"
    )]
    metric: Vec<Metric>,

    /// HNSW connectivity parameter (M), comma-separated for sweep
    #[arg(long, env = "RETRIEVAL_CONNECTIVITY", value_delimiter = ',', default_value = "32", value_parser = retrieval::parse_count_flag)]
    connectivity: Vec<usize>,

    /// HNSW expansion factor during indexing, comma-separated for sweep
    #[arg(long, env = "RETRIEVAL_EXPANSION_ADD", value_delimiter = ',', default_value = "128", value_parser = retrieval::parse_count_flag)]
    expansion_add: Vec<usize>,

    /// HNSW expansion factor during search, comma-separated for sweep
    #[arg(long, env = "RETRIEVAL_EXPANSION_SEARCH", value_delimiter = ',', default_value = "64", value_parser = retrieval::parse_count_flag)]
    expansion_search: Vec<usize>,

    /// Number of threads (sets OMP_NUM_THREADS)
    #[arg(long, env = "RETRIEVAL_THREADS", default_value = "0", value_parser = retrieval::parse_threads_flag)]
    threads: Threads,

    /// Persisted index: loaded and searched if the path exists, else built and saved; needs a single config
    #[arg(long, env = "RETRIEVAL_INDEX", value_parser = clap::builder::NonEmptyStringValueParser::new())]
    index: Option<String>,
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
    // `IDMap` persists keys beside the vectors; faiss-sys 0.7 has no `IDMapBinary`, so binary indexes keep a key table.
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

/// Writes one batch of FAISS answers, `count` per query: labels are our keys on the `IDMap` paths,
/// and indices into `key_map` on the binary path, which has no `IDMap`. Negative labels are misses.
fn write_hits<D: Copy>(
    labels: &[faiss::Idx],
    distances: &[D],
    key_map: Option<&[Key]>,
    to_distance: impl Fn(D) -> Distance,
    mut hits: Hits<'_>,
) -> Result<(), String> {
    let count = hits.count();
    let answers = labels.chunks_exact(count).zip(distances.chunks_exact(count));
    for ((labels, distances), row) in answers.zip(hits.rows()) {
        row.write(labels.iter().zip(distances).filter_map(|(label, &distance)| {
            let label = label.get()?;
            let key = key_map.map_or(label as Key, |key_map| key_map[label as usize]);
            Some(Ok((key, to_distance(distance))))
        }))?;
    }
    Ok(())
}

enum FaissIndex {
    Float(faiss::index::IndexImpl),
    /// Binary HNSW has no `IDMap` analogue in faiss-sys 0.7, so the keys ride along in insertion order.
    Binary {
        index: faiss::index::BinaryIndexImpl,
        keys: Vec<Key, std::alloc::System>,
    },
}

struct FaissBackend {
    scratch: Vec<f32, std::alloc::System>,
    ids: Vec<faiss::Idx, std::alloc::System>,
    index: FaissIndex,
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
        let factory = index_factory_string(data_type, connectivity);
        // Binary dimensions are bits, from the `.b1bin` header; float dimensions are scalars.
        let dimensions = dimensions as u32;
        let (metric_label, index) = match data_type {
            DataType::B1 => {
                let index = faiss::index::index_binary_factory(dimensions, &factory)
                    .map_err(|error| format!("failed to create FAISS binary index: {error}"))?;
                let keys = Vec::new_in(std::alloc::System);
                ("hamming", FaissIndex::Binary { index, keys })
            }
            _ => {
                let index = faiss::index::index_factory(dimensions, &factory, metric.into())
                    .map_err(|error| format!("failed to create FAISS index: {error}"))?;
                // Binary HNSW has no dispatcher in `faiss/AutoTune.cpp`, so only the float path tunes
                // efConstruction and efSearch; `IndexBinaryHNSW` keeps its compile-time 40 and 16.
                tune(
                    &index,
                    &format!("efConstruction={expansion_add} efSearch={expansion_search}"),
                )?;
                (metric_label_for(metric.into()), FaissIndex::Float(index))
            }
        };
        let metadata = HashMap::from([
            ("data_type".into(), json!(data_type.to_string())),
            ("metric".into(), json!(metric_label)),
            ("connectivity".into(), json!(connectivity)),
            ("expansion_add".into(), json!(expansion_add)),
            ("expansion_search".into(), json!(expansion_search)),
        ]);
        Ok(Self::wrap(index, threads, metadata))
    }

    /// Opens a saved float (`IDMap,…`) index; faiss-sys 0.7 binds no HNSW introspection, so the
    /// build-time settings are reported as null, and binary HNSW cannot be saved at all.
    pub fn load(path: &str, expansion_search: usize, threads: Threads) -> Result<Self, String> {
        let index = read_index(path).map_err(|error| {
            format!(
                "FAISS read_index({path}): {error}\n  \
                 (binary `--data-type b1` indexes can't be loaded yet — \
                 faiss-sys 0.7 doesn't bind IndexBinaryIDMap)"
            )
        })?;
        tune(&index, &format!("efSearch={expansion_search}"))?;
        let metadata = HashMap::from([
            ("metric".into(), json!(metric_label_for(index.metric_type()))),
            ("expansion_search".into(), json!(expansion_search)),
            ("loaded_from".into(), json!(path)),
            ("data_type".into(), Value::Null),
            ("connectivity".into(), Value::Null),
            ("expansion_add".into(), Value::Null),
        ]);
        Ok(Self::wrap(FaissIndex::Float(index), threads, metadata))
    }

    fn wrap(index: FaissIndex, threads: Threads, mut metadata: HashMap<String, Value>) -> Self {
        // SAFETY: OpenMP reads the thread count on its next parallel region.
        unsafe { omp_set_num_threads(threads.0.get() as i32) };
        metadata.insert("backend".into(), json!("faiss"));
        metadata.insert("library_version".into(), json!(faiss_version()));
        metadata.insert("threads".into(), json!(threads.0.get()));
        Self {
            scratch: Vec::new_in(std::alloc::System),
            ids: Vec::new_in(std::alloc::System),
            index,
            metadata,
        }
    }
}

/// Applies `parameters`, like `"efConstruction=128 efSearch=64"`, through FAISS's ParameterSpace.
fn tune(index: &faiss::index::IndexImpl, parameters: &str) -> Result<(), String> {
    FaissParameterSpace::new()?
        .set_index_parameters(index.inner_ptr() as *mut std::ffi::c_void, parameters)
        .map_err(|error| format!("FAISS ParameterSpace rejected `{parameters}`: {error}"))
}

impl Backend for FaissBackend {
    fn metadata(&self) -> HashMap<String, Value> {
        self.metadata.clone()
    }

    fn add(&mut self, keys: &[Key], vectors: Vectors) -> Result<(), String> {
        match &mut self.index {
            FaissIndex::Float(index) => {
                let data = vectors.to_f32_in(&mut self.scratch)?;
                self.ids.clear();
                self.ids.extend(keys.iter().map(|&key| faiss::Idx::new(u64::from(key))));
                index
                    .add_with_ids(data, &self.ids)
                    .map_err(|error| format!("FAISS add_with_ids failed: {error}"))
            }
            FaissIndex::Binary {
                index,
                keys: indexed_keys,
            } => {
                let VectorSlice::B1x8(bytes) = vectors.values() else {
                    return Err("FAISS binary index requires B1x8 data".into());
                };
                indexed_keys.extend_from_slice(keys);
                index
                    .add(bytes)
                    .map_err(|error| format!("FAISS binary add failed: {error}"))
            }
        }
    }

    fn search(&mut self, queries: Vectors, hits: Hits<'_>) -> Result<(), String> {
        let count = hits.count();
        match &mut self.index {
            FaissIndex::Float(index) => {
                let data = queries.to_f32_in(&mut self.scratch)?;
                let result = index
                    .search(data, count)
                    .map_err(|error| format!("FAISS search failed: {error}"))?;
                write_hits(&result.labels, &result.distances, None, |distance| distance, hits)
            }
            FaissIndex::Binary { index, keys } => {
                let VectorSlice::B1x8(bytes) = queries.values() else {
                    return Err("FAISS binary index requires B1x8 data".into());
                };
                let result = index
                    .search(bytes, count)
                    .map_err(|error| format!("FAISS binary search failed: {error}"))?;
                write_hits(
                    &result.labels,
                    &result.distances,
                    Some(keys),
                    |distance| distance as Distance,
                    hits,
                )
            }
        }
    }

    fn memory_bytes(&self) -> usize {
        retrieval::process_rss_bytes() as usize
    }

    fn save(&self, path: &str) -> Result<(), String> {
        match &self.index {
            FaissIndex::Float(index) => {
                write_index(index, path).map_err(|error| format!("FAISS write_index({path}): {error}"))
            }
            FaissIndex::Binary { .. } => Err("FAISS binary HNSW save is not supported — faiss-sys 0.7 doesn't bind \
                 IndexBinaryIDMap, so keys can't be persisted natively"
                .into()),
        }
    }
}

/// FAISS version string; its C API exposes no ISA introspection to report beside it.
fn faiss_version() -> String {
    use std::ffi::CStr;

    extern "C" {
        fn faiss_get_version() -> *const std::os::raw::c_char;
    }
    // SAFETY: the version is null or a static C string.
    unsafe {
        let version = faiss_get_version();
        match version.is_null() {
            true => "unknown".into(),
            false => CStr::from_ptr(version).to_string_lossy().into_owned(),
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
    let configs = iproduct!(
        &dimensions_sweep,
        &cli.data_type,
        &cli.metric,
        &cli.connectivity,
        &cli.expansion_add,
        &cli.expansion_search
    );
    let index_file = IndexFile::new(cli.index.as_deref(), configs.clone().count());
    eprintln!("- Index: {index_file}");

    let mut summary = SweepSummary::default();
    for (&dimensions, &data_type, &metric, &connectivity, &expansion_add, &expansion_search) in configs {
        let description = format!(
            "faiss · {data_type} · {metric} · d={dimensions} · M={connectivity} · ef={expansion_add}/{expansion_search} · {} threads",
            cli.threads
        );
        let backend = match index_file {
            IndexFile::Load(path) => FaissBackend::load(path, expansion_search, cli.threads),
            IndexFile::None | IndexFile::Save(_) => FaissBackend::new(
                IndexConfig {
                    dimensions,
                    data_type,
                    metric,
                    connectivity,
                    expansion_add,
                    expansion_search,
                },
                cli.threads,
            ),
        };
        summary.record(run_config(&description, index_file, backend, &mut state, dimensions));
    }

    summary.finish();
}

#[cfg(test)]
mod tests {
    use super::*;
    use retrieval::SearchBuffers;

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
            .add(&[42, 7], Vectors::new(VectorSlice::I8(&[3, 4, 0, 0]), 2))
            .unwrap();
        let scratch = backend.scratch.as_ptr();
        let mut buffers = SearchBuffers::new(1, 3).unwrap();
        for _ in 0..2 {
            backend
                .search(Vectors::new(VectorSlice::I8(&[0, 0]), 2), buffers.hits(0..1))
                .unwrap();
            let (keys, distances, found) = buffers.rows().next().unwrap();
            assert_eq!(keys, [7, 42, Key::MAX]);
            assert_eq!(distances, [0.0, 25.0, Distance::INFINITY]);
            assert_eq!(found, 2);
            assert_eq!(scratch, backend.scratch.as_ptr());
        }
    }
}
