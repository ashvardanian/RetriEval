//! USearch HNSW benchmark binary.
//!
//! ## Build & Install
//!
//! USearch is the default backend — no extra system dependencies required:
//!
//! ```sh
//! cargo install --path . --features usearch-backend
//! ```
//!
//! ## Examples
//!
//! Quick sweep over quantization types & metrics (Wiki 1M):
//! ```sh
//! retri-eval-usearch \
//!     --vectors datasets/wiki_1M/base.1M.fbin \
//!     --queries datasets/wiki_1M/query.public.100K.fbin \
//!     --neighbors datasets/wiki_1M/groundtruth.public.100K.ibin \
//!     --dtype f32,bf16,e5m2 \
//!     --metric ip,cos,l2 \
//!     --output results/
//! ```
//!
//! Turing 10M at 99% recall (M=32, ef=256/1024):
//! ```sh
//! retri-eval-usearch \
//!     --vectors datasets/turing_10M/base.10M.fbin \
//!     --queries datasets/turing_10M/query.public.100K.fbin \
//!     --neighbors datasets/turing_10M/groundtruth.public.100K.ibin \
//!     --dtype f32,bf16,e5m2,e4m3,e3m2,e2m3,i8 \
//!     --shards 2 \
//!     --metric l2 \
//!     --connectivity 32 \
//!     --expansion-add 256 \
//!     --expansion-search 1024 \
//!     --output results/
//! ```
//!
//! Turing 100M with 20 measurement steps:
//! ```sh
//! retri-eval-usearch \
//!     --vectors datasets/turing_100M/base.100M.fbin \
//!     --queries datasets/turing_100M/query.public.100K.fbin \
//!     --neighbors datasets/turing_100M/groundtruth.public.100K.ibin \
//!     --dtype f32,bf16 \
//!     --shards 2 \
//!     --metric l2 \
//!     --connectivity 32 \
//!     --expansion-add 256 \
//!     --expansion-search 1024 \
//!     --epochs 20 \
//!     --output results/turing_100M
//! ```
//!
//! Binary 1M hamming-distance search (1024-bit vectors in `.b1bin` format):
//! ```sh
//! retri-eval-usearch \
//!     --vectors datasets/binary_1M/base.1M.b1bin \
//!     --queries datasets/binary_1M/query.10K.b1bin \
//!     --neighbors datasets/binary_1M/groundtruth.10K.ibin \
//!     --dtype b1 \
//!     --metric hamming \
//!     --output results/binary_1M
//! ```

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, Ordering};

use clap::Parser;
use fork_union::{IndexedSplit, SyncMutPtr, ThreadPool};
use retrieval::*;

#[derive(Parser, Debug)]
#[command(name = "retri-eval-usearch", about = "Benchmark USearch HNSW")]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,

    /// Quantization types (comma-separated)
    #[arg(long, value_delimiter = ',', default_value = "bf16")]
    dtype: Vec<String>,

    /// Distance metric: ip, l2, cos, hamming, jaccard, sorensen, pearson, haversine, divergence
    #[arg(long, value_delimiter = ',', default_value = "l2")]
    metric: Vec<String>,

    /// HNSW connectivity M (comma-separated for sweep, 0 = USearch default)
    #[arg(long, value_delimiter = ',', default_value = "0")]
    connectivity: Vec<usize>,

    /// HNSW construction expansion factor (comma-separated for sweep, 0 = USearch default)
    #[arg(long, value_delimiter = ',', default_value = "0")]
    expansion_add: Vec<usize>,

    /// HNSW search expansion factor (comma-separated for sweep, 0 = USearch default)
    #[arg(long, value_delimiter = ',', default_value = "0")]
    expansion_search: Vec<usize>,

    /// Number of index shards (comma-separated for sweep)
    #[arg(long, value_delimiter = ',', default_value = "1")]
    shards: Vec<usize>,

    /// Number of threads (comma-separated for sweep)
    #[arg(long, value_delimiter = ',', default_values_t = vec![std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)])]
    threads: Vec<usize>,
}

fn parse_metric(s: &str) -> Result<::usearch::MetricKind, String> {
    match s {
        "ip" => Ok(::usearch::MetricKind::IP),
        "l2" | "l2sq" => Ok(::usearch::MetricKind::L2sq),
        "cos" => Ok(::usearch::MetricKind::Cos),
        "hamming" => Ok(::usearch::MetricKind::Hamming),
        "jaccard" | "tanimoto" => Ok(::usearch::MetricKind::Tanimoto),
        "sorensen" => Ok(::usearch::MetricKind::Sorensen),
        "pearson" => Ok(::usearch::MetricKind::Pearson),
        "haversine" => Ok(::usearch::MetricKind::Haversine),
        "divergence" | "jensenshannon" => Ok(::usearch::MetricKind::Divergence),
        _ => Err(format!("unknown metric: {s}. supported: ip, l2sq, cos, hamming, jaccard, sorensen, pearson, haversine, divergence")),
    }
}

fn parse_dtype(s: &str) -> Result<::usearch::ScalarKind, String> {
    match s {
        "f64" => Ok(::usearch::ScalarKind::F64),
        "f32" => Ok(::usearch::ScalarKind::F32),
        "bf16" => Ok(::usearch::ScalarKind::BF16),
        "f16" => Ok(::usearch::ScalarKind::F16),
        "e5m2" => Ok(::usearch::ScalarKind::E5M2),
        "e4m3" => Ok(::usearch::ScalarKind::E4M3),
        "e3m2" => Ok(::usearch::ScalarKind::E3M2),
        "e2m3" => Ok(::usearch::ScalarKind::E2M3),
        "i8" => Ok(::usearch::ScalarKind::I8),
        "u8" => Ok(::usearch::ScalarKind::U8),
        "b1" => Ok(::usearch::ScalarKind::B1),
        _ => Err(format!("unknown dtype: {s}")),
    }
}

// #region Backend

pub struct USearchBackend {
    shards: Vec<::usearch::Index>,
    pool: UnsafeCell<ThreadPool>,
    description: String,
    metadata: std::collections::HashMap<String, serde_json::Value>,
}

unsafe impl Sync for USearchBackend {}
unsafe impl Send for USearchBackend {}

impl USearchBackend {
    pub fn new(
        dimensions: usize,
        metric_name: &str,
        dtype_name: &str,
        connectivity: usize,
        expansion_add: usize,
        expansion_search: usize,
        threads: usize,
        shards: usize,
    ) -> Result<Self, String> {
        let metric = parse_metric(metric_name)?;
        let dtype = parse_dtype(dtype_name)?;
        let opts = ::usearch::IndexOptions {
            dimensions,
            metric,
            quantization: dtype,
            connectivity,
            expansion_add,
            expansion_search,
            multi: false,
        };

        let shards = shards.max(1);
        let threads = threads.max(1);

        let mut shard_vec = Vec::with_capacity(shards);
        for _ in 0..shards {
            shard_vec.push(
                ::usearch::Index::new(&opts)
                    .map_err(|e| format!("failed to create USearch index: {e}"))?,
            );
        }

        let pool = ThreadPool::try_spawn(threads)
            .map_err(|e| format!("failed to create thread pool: {e}"))?;

        let fmt_param = |v: usize| {
            if v == 0 {
                "auto".to_string()
            } else {
                v.to_string()
            }
        };
        let mut description = format!(
            "usearch · {dtype_name} · {metric_name} · M={} · ef={}/{} · {threads} threads",
            fmt_param(connectivity),
            fmt_param(expansion_add),
            fmt_param(expansion_search),
        );
        if shards > 1 {
            description.push_str(&format!(" · {shards} shards"));
        }

        use serde_json::json;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("backend".into(), json!("usearch"));
        metadata.insert("dtype".into(), json!(dtype_name));
        metadata.insert("metric".into(), json!(metric_name));
        metadata.insert("connectivity".into(), json!(connectivity));
        metadata.insert("expansion_add".into(), json!(expansion_add));
        metadata.insert("expansion_search".into(), json!(expansion_search));
        metadata.insert("threads".into(), json!(threads));
        metadata.insert("shards".into(), json!(shards));

        Ok(Self {
            shards: shard_vec,
            pool: UnsafeCell::new(pool),
            description,
            metadata,
        })
    }

    fn pool_mut(&self) -> &mut ThreadPool {
        unsafe { &mut *self.pool.get() }
    }

    fn shard_count(&self) -> usize {
        self.shards.len()
    }
}

impl Backend for USearchBackend {
    fn description(&self) -> String {
        self.description.clone()
    }

    fn metadata(&self) -> std::collections::HashMap<String, serde_json::Value> {
        self.metadata.clone()
    }

    fn add(&mut self, keys: &[Key], vectors: Vectors) -> Result<(), String> {
        let num_vectors = keys.len();
        let dimensions = vectors.dimensions;
        let shard_count = self.shard_count();
        let pool = self.pool_mut();
        let threads = pool.threads();

        let per_shard = div_ceil(num_vectors, shard_count);
        for shard in &self.shards {
            shard
                .reserve_capacity_and_threads(shard.size() + per_shard, threads)
                .map_err(|e| format!("failed to reserve capacity: {e}"))?;
        }

        let failed = AtomicBool::new(false);
        let split = IndexedSplit::new(num_vectors, threads);
        let shards = &self.shards;

        pool.for_threads(|thread_index, _| {
            for i in split.get(thread_index) {
                if failed.load(Ordering::Relaxed) {
                    return;
                }
                let shard = &shards[i % shard_count];
                let ok = match vectors.data {
                    VectorSlice::F32(data) => shard
                        .add(keys[i] as u64, &data[i * dimensions..(i + 1) * dimensions])
                        .is_ok(),
                    VectorSlice::I8(data) => shard
                        .add(keys[i] as u64, &data[i * dimensions..(i + 1) * dimensions])
                        .is_ok(),
                    VectorSlice::U8(data) => shard
                        .add(keys[i] as u64, &data[i * dimensions..(i + 1) * dimensions])
                        .is_ok(),
                    VectorSlice::B1x8(data) => {
                        let stride = div_ceil(dimensions, 8);
                        shard
                            .add(
                                keys[i] as u64,
                                ::usearch::b1x8::from_u8s(&data[i * stride..(i + 1) * stride]),
                            )
                            .is_ok()
                    }
                };
                if !ok {
                    failed.store(true, Ordering::Relaxed);
                    return;
                }
            }
        });

        if failed.load(Ordering::Relaxed) {
            Err("one or more vectors failed to add".to_string())
        } else {
            Ok(())
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
        let dimensions = queries.dimensions;
        let num_queries = queries.len();

        debug_assert_eq!(out_keys.len(), num_queries * count);
        debug_assert_eq!(out_distances.len(), num_queries * count);
        debug_assert_eq!(out_counts.len(), num_queries);

        let keys_ptr = SyncMutPtr::new(out_keys.as_mut_ptr());
        let dists_ptr = SyncMutPtr::new(out_distances.as_mut_ptr());
        let counts_ptr = SyncMutPtr::new(out_counts.as_mut_ptr());
        let pool = self.pool_mut();
        let threads = pool.threads();
        let failed = AtomicBool::new(false);
        let split = IndexedSplit::new(num_queries, threads);
        let shards = &self.shards;

        pool.for_threads(|thread_index, _| {
            for i in split.get(thread_index) {
                if failed.load(Ordering::Relaxed) {
                    return;
                }

                let offset = i * count;
                unsafe {
                    let keys = keys_ptr.as_ptr().add(offset);
                    let distances = dists_ptr.as_ptr().add(offset);
                    for j in 0..count {
                        *keys.add(j) = Key::MAX;
                        *distances.add(j) = Distance::INFINITY;
                    }
                }

                for shard in shards {
                    let result = match queries.data {
                        VectorSlice::F32(data) => {
                            shard.search(&data[i * dimensions..(i + 1) * dimensions], count)
                        }
                        VectorSlice::I8(data) => {
                            shard.search(&data[i * dimensions..(i + 1) * dimensions], count)
                        }
                        VectorSlice::U8(data) => {
                            shard.search(&data[i * dimensions..(i + 1) * dimensions], count)
                        }
                        VectorSlice::B1x8(data) => {
                            let stride = div_ceil(dimensions, 8);
                            shard.search(
                                ::usearch::b1x8::from_u8s(&data[i * stride..(i + 1) * stride]),
                                count,
                            )
                        }
                    };

                    let matches = match result {
                        Ok(m) => m,
                        Err(_) => {
                            failed.store(true, Ordering::Relaxed);
                            return;
                        }
                    };

                    unsafe {
                        let keys = keys_ptr.as_ptr().add(offset);
                        let distances = dists_ptr.as_ptr().add(offset);

                        for j in 0..matches.keys.len().min(count) {
                            let dist = matches.distances[j];
                            let key = matches.keys[j] as Key;
                            if dist >= *distances.add(count - 1) {
                                break;
                            }
                            let mut pos = count - 1;
                            while pos > 0 && dist < *distances.add(pos - 1) {
                                *keys.add(pos) = *keys.add(pos - 1);
                                *distances.add(pos) = *distances.add(pos - 1);
                                pos -= 1;
                            }
                            *keys.add(pos) = key;
                            *distances.add(pos) = dist;
                        }
                    }
                }

                unsafe {
                    let keys = keys_ptr.as_ptr().add(offset);
                    let mut found = 0;
                    for j in 0..count {
                        if *keys.add(j) != Key::MAX {
                            found += 1;
                        } else {
                            break;
                        }
                    }
                    *counts_ptr.as_ptr().add(i) = found;
                }
            }
        });

        if failed.load(Ordering::Relaxed) {
            Err("one or more search queries failed".to_string())
        } else {
            Ok(())
        }
    }

    fn memory_bytes(&self) -> usize {
        self.shards.iter().map(|s| s.memory_usage()).sum()
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

    for (dtype, metric, connectivity, expansion_add, expansion_search, shards, threads) in itertools::iproduct!(
        &cli.dtype,
        &cli.metric,
        &cli.connectivity,
        &cli.expansion_add,
        &cli.expansion_search,
        &cli.shards,
        &cli.threads
    ) {
        let mut index = USearchBackend::new(
            dimensions,
            metric,
            dtype,
            *connectivity,
            *expansion_add,
            *expansion_search,
            *threads,
            *shards,
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
