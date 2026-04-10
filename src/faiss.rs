//! FAISS HNSW benchmark binary.
//!
//! ```sh
//! cargo run --release --bin retri-eval-faiss --features faiss-backend -- \
//!     --vectors datasets/turing_10M/base.10M.fbin \
//!     --queries datasets/turing_10M/query.public.100K.fbin \
//!     --neighbors datasets/turing_10M/groundtruth.public.100K.ibin \
//!     --dtype f32,f16,i8 \
//!     --metric l2 \
//!     --threads 16 \
//!     --output results/
//! ```

use std::collections::HashMap;

use clap::Parser;
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
        _ => Err(format!(
            "unknown FAISS metric: {s}. FAISS HNSW supports: ip, l2"
        )),
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

struct FaissBackend {
    index: faiss::Index,
    description: String,
    metadata: HashMap<String, Value>,
}

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
        let dim = if is_binary(dtype_name) {
            dimensions * 8
        } else {
            dimensions
        } as u32;
        let (metric_label, faiss_metric) = parse_metric(metric_name)?;

        let index = faiss::index::index_factory(dim, &factory, faiss_metric)
            .map_err(|e| format!("failed to create FAISS index: {e}"))?;

        // Set HNSW expansion factors via FAISS ParameterSpace API
        let params_str = format!("efConstruction={expansion_add} efSearch={expansion_search}");
        unsafe {
            let mut space: *mut std::ffi::c_void = std::ptr::null_mut();
            if faiss_ParameterSpace_new(&mut space) == 0 && !space.is_null() {
                let c_params = std::ffi::CString::new(params_str.as_str()).unwrap();
                faiss_ParameterSpace_set_index_parameters(
                    space,
                    index.inner_ptr(),
                    c_params.as_ptr(),
                );
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
        let data = vectors.data.to_f32();
        self.index
            .add(&data)
            .map_err(|e| format!("FAISS add failed: {e}"))
    }

    fn search(
        &self,
        queries: Vectors,
        count: usize,
        out_keys: &mut [Key],
        out_distances: &mut [Distance],
        out_counts: &mut [usize],
    ) -> Result<(), String> {
        let dimensions = queries.dimensions;
        let data = queries.data.to_f32();
        let num_queries = data.len() / dimensions;

        let result = self
            .index
            .search(&data, count)
            .map_err(|e| format!("FAISS search failed: {e}"))?;

        for query_idx in 0..num_queries {
            let offset = query_idx * count;
            let mut found = 0;
            for rank in 0..count {
                let neighbor_idx = result.labels[offset + rank];
                if neighbor_idx >= 0 {
                    out_keys[offset + rank] = neighbor_idx as Key;
                    out_distances[offset + rank] = result.distances[offset + rank];
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
