//! USearch HNSW benchmark binary.
//!
//! ```sh
//! cargo run --release --bin retri-eval-usearch -- \
//!     --vectors datasets/turing_10M/base.10M.fbin \
//!     --queries datasets/turing_10M/query.public.100K.fbin \
//!     --neighbors datasets/turing_10M/groundtruth.public.100K.ibin \
//!     --dtype f32,f16,i8 \
//!     --metric l2 \
//!     --output turing-10M.jsonl
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
    #[arg(long, value_delimiter = ',', default_value = "f32")]
    dtype: Vec<String>,

    /// Distance metric: ip, l2, cos, hamming, jaccard, sorensen, pearson, haversine, divergence
    #[arg(long, value_delimiter = ',', default_value = "l2")]
    metric: Vec<String>,

    /// HNSW connectivity M (comma-separated for sweep)
    #[arg(long, value_delimiter = ',', default_value = "16")]
    connectivity: Vec<usize>,

    /// HNSW construction expansion factor (comma-separated for sweep)
    #[arg(long, value_delimiter = ',', default_value = "128")]
    expansion_add: Vec<usize>,

    /// HNSW search expansion factor (comma-separated for sweep)
    #[arg(long, value_delimiter = ',', default_value = "64")]
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

        let mut desc = format!(
            "usearch · {dtype_name} · {metric_name} · M={connectivity} · ef={expansion_add}/{expansion_search} · {threads} threads",
        );
        if shards > 1 {
            desc.push_str(&format!(" · {shards} shards"));
        }

        use serde_json::json;
        let mut meta = std::collections::HashMap::new();
        meta.insert("backend".into(), json!("usearch"));
        meta.insert("dtype".into(), json!(dtype_name));
        meta.insert("metric".into(), json!(metric_name));
        meta.insert("connectivity".into(), json!(connectivity));
        meta.insert("expansion_add".into(), json!(expansion_add));
        meta.insert("expansion_search".into(), json!(expansion_search));
        meta.insert("threads".into(), json!(threads));
        meta.insert("shards".into(), json!(shards));

        Ok(Self {
            shards: shard_vec,
            pool: UnsafeCell::new(pool),
            description: desc,
            metadata: meta,
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
        let n = keys.len();
        let d = vectors.dimensions;
        let s = self.shard_count();
        let pool = self.pool_mut();
        let threads = pool.threads();

        let per_shard = div_ceil(n, s);
        for shard in &self.shards {
            shard
                .reserve_capacity_and_threads(shard.size() + per_shard, threads)
                .map_err(|e| format!("failed to reserve capacity: {e}"))?;
        }

        let failed = AtomicBool::new(false);
        let split = IndexedSplit::new(n, threads);
        let shards = &self.shards;

        pool.for_threads(|thread_index, _| {
            for i in split.get(thread_index) {
                if failed.load(Ordering::Relaxed) {
                    return;
                }
                let shard = &shards[i % s];
                let ok = match vectors.data {
                    VectorSlice::F32(data) => {
                        shard.add(keys[i] as u64, &data[i * d..(i + 1) * d]).is_ok()
                    }
                    VectorSlice::I8(data) => {
                        shard.add(keys[i] as u64, &data[i * d..(i + 1) * d]).is_ok()
                    }
                    VectorSlice::U8(data) => {
                        shard.add(keys[i] as u64, &data[i * d..(i + 1) * d]).is_ok()
                    }
                    VectorSlice::B1x8(data) => {
                        let stride = div_ceil(d, 8);
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
        let d = queries.dimensions;
        let n = queries.len();

        debug_assert_eq!(out_keys.len(), n * count);
        debug_assert_eq!(out_distances.len(), n * count);
        debug_assert_eq!(out_counts.len(), n);

        let keys_ptr = SyncMutPtr::new(out_keys.as_mut_ptr());
        let dists_ptr = SyncMutPtr::new(out_distances.as_mut_ptr());
        let counts_ptr = SyncMutPtr::new(out_counts.as_mut_ptr());
        let pool = self.pool_mut();
        let threads = pool.threads();
        let failed = AtomicBool::new(false);
        let split = IndexedSplit::new(n, threads);
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
                        VectorSlice::F32(data) => shard.search(&data[i * d..(i + 1) * d], count),
                        VectorSlice::I8(data) => shard.search(&data[i * d..(i + 1) * d], count),
                        VectorSlice::U8(data) => shard.search(&data[i * d..(i + 1) * d], count),
                        VectorSlice::B1x8(data) => {
                            let stride = div_ceil(d, 8);
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
