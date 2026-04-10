//! FAISS HNSW benchmark binary.
//!
//! ```sh
//! cargo run --release --bin bench-faiss --features faiss-backend -- \
//!     --vectors datasets/turing_10M/base.10M.fbin \
//!     --queries datasets/turing_10M/query.public.100K.fbin \
//!     --neighbors datasets/turing_10M/groundtruth.public.100K.ibin \
//!     --dtype f32,f16,i8 \
//!     --metric l2 \
//!     --threads 16 \
//!     --output turing-10M-faiss.jsonl
//! ```

use clap::Parser;
use itertools::iproduct;
use usearch_bench::{
    dataset, div_ceil, fmt_thousands, run, Backend, BenchState, CommonArgs, Distance, Key,
    VectorSlice, Vectors,
};

extern "C" {
    fn omp_set_num_threads(num_threads: i32);
}

// #region CLI

#[derive(Parser, Debug)]
#[command(name = "bench-faiss", about = "Benchmark FAISS HNSW")]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,

    /// Comma-separated quantization types: f32, f16, bf16, u8, i8, b1
    #[arg(long, value_delimiter = ',', default_value = "f32")]
    dtype: Vec<String>,

    /// Comma-separated distance metrics: ip, l2, cos
    #[arg(long, value_delimiter = ',', default_value = "l2")]
    metric: Vec<String>,

    /// HNSW connectivity parameter (M)
    #[arg(long, default_value_t = 16)]
    connectivity: usize,

    /// Number of threads (sets OMP_NUM_THREADS)
    #[arg(long, default_value_t = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1))]
    threads: usize,
}

// #region Local metric mapping

fn parse_faiss_metric(s: &str) -> Result<faiss::MetricType, String> {
    match s {
        "ip" => Ok(faiss::MetricType::InnerProduct),
        _ => Ok(faiss::MetricType::L2),
    }
}

/// Return a short human-readable label for the metric string.
fn metric_label(s: &str) -> &str {
    match s {
        "ip" => "ip",
        "cos" => "cos",
        _ => "l2",
    }
}

// #region FaissDtype

enum FaissDtype {
    F32,
    F16,
    BF16,
    U8,
    I8,
    B1,
}

impl FaissDtype {
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "f32" => Ok(Self::F32),
            "f16" => Ok(Self::F16),
            "bf16" => Ok(Self::BF16),
            "u8" => Ok(Self::U8),
            "i8" => Ok(Self::I8),
            "b1" => Ok(Self::B1),
            _ => Err(format!("unknown FAISS dtype: {s}")),
        }
    }

    fn index_factory_string(&self, connectivity: usize) -> String {
        match self {
            Self::F32 => format!("HNSW{connectivity},Flat"),
            Self::F16 => format!("HNSW{connectivity},SQfp16"),
            Self::BF16 => format!("HNSW{connectivity},SQbf16"),
            Self::U8 => format!("HNSW{connectivity},SQ8bit_direct"),
            Self::I8 => format!("HNSW{connectivity},SQ8bit_direct_signed"),
            Self::B1 => format!("BinaryHNSW{connectivity}"),
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::U8 => "u8",
            Self::I8 => "i8",
            Self::B1 => "b1",
        }
    }

    fn is_binary(&self) -> bool {
        matches!(self, Self::B1)
    }
}

// #region Backend

struct FaissBackend {
    index: faiss::Index,
    description: String,
}

impl Backend for FaissBackend {
    fn description(&self) -> String {
        self.description.clone()
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
        let num_vectors = data.len() / dimensions;

        let result = self
            .index
            .search(&data, count)
            .map_err(|e| format!("FAISS search failed: {e}"))?;

        for q in 0..num_vectors {
            let offset = q * count;
            let mut found = 0;
            for j in 0..count {
                let idx = result.labels[offset + j];
                if idx >= 0 {
                    out_keys[offset + j] = idx as Key;
                    out_distances[offset + j] = result.distances[offset + j];
                    found += 1;
                } else {
                    out_keys[offset + j] = Key::MAX;
                    out_distances[offset + j] = Distance::INFINITY;
                }
            }
            out_counts[q] = found;
        }
        Ok(())
    }

    fn memory_bytes(&self) -> usize {
        0
    }
}

// #region main

fn main() {
    let cli = Cli::parse();

    unsafe {
        omp_set_num_threads(cli.threads as i32);
    }

    let mut state = BenchState::load(&cli.common).unwrap_or_else(|e| {
        eprintln!("Failed to load benchmark state: {e}");
        std::process::exit(1);
    });
    let dimensions = state.dimensions();

    for (dtype_str, metric_str) in iproduct!(&cli.dtype, &cli.metric) {
        let faiss_metric = parse_faiss_metric(metric_str).unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(1);
        });

        let dtype = FaissDtype::from_str(dtype_str).unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(1);
        });

        let factory = dtype.index_factory_string(cli.connectivity);
        let dim = if dtype.is_binary() {
            dimensions * 8
        } else {
            dimensions
        } as u32;

        let index = faiss::index::index_factory(dim, &factory, faiss_metric).unwrap_or_else(|e| {
            eprintln!("Failed to create FAISS index: {e}");
            std::process::exit(1);
        });

        let description = format!(
            "faiss · {} · {} · M={} · {} threads",
            dtype.as_str(),
            metric_label(metric_str),
            cli.connectivity,
            cli.threads,
        );
        let mut backend = FaissBackend { index, description };

        run(&mut backend, &mut state).unwrap_or_else(|e| {
            eprintln!("Benchmark failed: {e}");
            std::process::exit(1);
        });
    }

    eprintln!("Benchmark complete.");
}
