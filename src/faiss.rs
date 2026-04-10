//! FAISS HNSW benchmark binary.
//!
//! ## Prerequisites
//!
//! Requires the FAISS C API library (`libfaiss_c`). Install via conda:
//!
//! ```sh
//! conda install -c conda-forge libfaiss
//! ```
//!
//! The conda package includes `libfaiss.so` but not the C API wrapper.
//! Build `libfaiss_c.so` from the FAISS source `c_api/` directory against
//! the installed `libfaiss`:
//!
//! ```sh
//! git clone --depth 1 --branch v1.10.0 https://github.com/facebookresearch/faiss.git /tmp/faiss
//! cd /tmp/faiss/c_api && mkdir build && cd build
//! cmake .. -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
//!     -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
//!     -DCMAKE_CXX_FLAGS="-I$CONDA_PREFIX/include" \
//!     -DCMAKE_SHARED_LINKER_FLAGS="-L$CONDA_PREFIX/lib" \
//!     -DBUILD_SHARED_LIBS=ON -DFAISS_ENABLE_GPU=OFF
//! make -j$(nproc) && cp libfaiss_c.so "$CONDA_PREFIX/lib/"
//! ```
//!
//! ## Build & Install
//!
//! ```sh
//! RUSTFLAGS="-L $CONDA_PREFIX/lib" cargo install --path . --features faiss-backend
//! ```
//!
//! ## Examples
//!
//! ```sh
//! retri-eval-faiss \
//!     --vectors datasets/turing_10M/base.10M.fbin \
//!     --queries datasets/turing_10M/query.public.100K.fbin \
//!     --neighbors datasets/turing_10M/groundtruth.public.100K.ibin \
//!     --dtype f32,f16,i8 \
//!     --metric l2 \
//!     --threads 16 \
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
//!     --threads 16 \
//!     --output results/binary_1M
//! ```

use std::cell::UnsafeCell;
use std::collections::HashMap;

use clap::Parser;
use faiss::Index as _;
use itertools::iproduct;
use retrieval::{run, Backend, BenchState, CommonArgs, Distance, Key, Vectors};
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

    /// HNSW connectivity parameter (M)
    #[arg(long, default_value_t = 32)]
    connectivity: usize,

    /// HNSW expansion factor during indexing
    #[arg(long, default_value_t = 128)]
    expansion_add: usize,

    /// HNSW expansion factor during search
    #[arg(long, default_value_t = 64)]
    expansion_search: usize,

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
        "u8" => Ok(format!("HNSW{connectivity},SQ8bit_direct")),
        "i8" => Ok(format!("HNSW{connectivity},SQ8bit_direct_signed")),
        "b1" => Ok(format!("BinaryHNSW{connectivity}")),
        _ => Err(format!("unknown FAISS dtype: {dtype}")),
    }
}

fn is_binary(dtype: &str) -> bool {
    dtype == "b1"
}

/// Unpack FAISS search results into output buffers.
/// Generic over the distance type — FAISS returns `f32` for float indices and `i32` for binary.
fn unpack_search_results<D: Copy>(
    labels: &[faiss::Idx],
    distances: &[D],
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
            let neighbor_idx = labels[offset + rank];
            if let Some(id) = neighbor_idx.get() {
                out_keys[offset + rank] = id as Key;
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

        // Set HNSW expansion factors via FAISS ParameterSpace API
        let inner_ptr = match &index {
            FaissIndex::Float(i) => unsafe { (*i.get()).inner_ptr() as *mut std::ffi::c_void },
            FaissIndex::Binary(i) => unsafe { (*i.get()).inner_ptr() as *mut std::ffi::c_void },
        };
        let params_str = format!("efConstruction={expansion_add} efSearch={expansion_search}");
        unsafe {
            let mut space: *mut std::ffi::c_void = std::ptr::null_mut();
            if faiss_ParameterSpace_new(&mut space) == 0 && !space.is_null() {
                let c_params = std::ffi::CString::new(params_str.as_str()).unwrap();
                faiss_ParameterSpace_set_index_parameters(space, inner_ptr, c_params.as_ptr());
                faiss_ParameterSpace_free(space);
            }
        }

        let description = format!(
            "faiss · {dtype_name} · {metric_label} · M={connectivity} · ef={expansion_add}/{expansion_search} · {threads} threads",
        );

        let mut metadata = HashMap::new();
        metadata.insert("backend".into(), json!("faiss"));
        metadata.insert("dtype".into(), json!(dtype_name));
        metadata.insert("metric".into(), json!(metric_label));
        metadata.insert("connectivity".into(), json!(connectivity));
        metadata.insert("expansion_add".into(), json!(expansion_add));
        metadata.insert("expansion_search".into(), json!(expansion_search));
        metadata.insert("threads".into(), json!(threads));

        Ok(Self {
            index,
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

    fn add(&mut self, _keys: &[Key], vectors: Vectors) -> Result<(), String> {
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
        0
    }
}

fn main() {
    let cli = Cli::parse();

    let mut state = BenchState::load(&cli.common).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    let dimensions = state.dimensions();

    for (dtype, metric) in iproduct!(&cli.dtype, &cli.metric) {
        let mut index = FaissBackend::new(
            dimensions,
            dtype,
            metric,
            cli.connectivity,
            cli.expansion_add,
            cli.expansion_search,
            cli.threads,
        )
        .unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(1);
        });

        run(&mut index, &mut state).unwrap_or_else(|e| {
            eprintln!("Benchmark failed: {e}");
            std::process::exit(1);
        });
    }

    eprintln!("Benchmark complete.");
}
