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
//!     --base-vectors datasets/wiki_1M/base.1M.fbin \
//!     --query-vectors datasets/wiki_1M/query.public.100K.fbin \
//!     --query-neighbors datasets/wiki_1M/groundtruth.public.100K.ibin \
//!     --data-type f32,bf16,e5m2 \
//!     --metric ip,cos,l2 \
//!     --output results/
//! ```
//!
//! Turing 10M at 99% recall (M=32, ef=256/1024):
//! ```sh
//! retri-eval-usearch \
//!     --base-vectors datasets/turing_10M/base.10M.fbin \
//!     --query-vectors datasets/turing_10M/query.public.100K.fbin \
//!     --query-neighbors datasets/turing_10M/groundtruth.public.100K.ibin \
//!     --data-type f32,bf16,e5m2,e4m3,e3m2,e2m3,i8 \
//!     --metric l2 \
//!     --connectivity 48 \
//!     --expansion-add 768 \
//!     --expansion-search 384 \
//!     --output results/
//! ```
//!
//! Turing 100M with 20 measurement steps:
//! ```sh
//! retri-eval-usearch \
//!     --base-vectors datasets/turing_100M/base.100M.fbin \
//!     --query-vectors datasets/turing_100M/query.public.100K.fbin \
//!     --query-neighbors datasets/turing_100M/groundtruth.public.100K.ibin \
//!     --data-type f32,bf16 \
//!     --shards 2 \
//!     --metric l2 \
//!     --connectivity 32 \
//!     --expansion-add 256 \
//!     --expansion-search 1024 \
//!     --steps 20 \
//!     --output results/turing_100M
//! ```
//!
//! Binary 1M hamming-distance search (1024-bit vectors in `.b1bin` format):
//! ```sh
//! retri-eval-usearch \
//!     --base-vectors datasets/binary_1M/base.1M.b1bin \
//!     --query-vectors datasets/binary_1M/query.10K.b1bin \
//!     --query-neighbors datasets/binary_1M/groundtruth.10K.ibin \
//!     --data-type b1 \
//!     --metric hamming \
//!     --output results/binary_1M
//! ```

use std::{
    fmt::{self, Write},
    sync::atomic::{AtomicBool, Ordering},
};

use clap::Parser;
use fork_union::{IndexedSplit, SyncMutPtr, ThreadPool};
use serde_json::json;

use retrieval::{
    run_config, spell_list, Backend, BenchState, CommonArgs, Distance, IndexConfig, Key, SweepSummary, Threads,
    UnwrapOrBail, VectorSlice, Vectors,
};

#[derive(Parser, Debug)]
#[command(name = "retri-eval-usearch", about = "Benchmark USearch HNSW")]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,

    /// Quantization types (comma-separated for sweep)
    #[arg(long, value_delimiter = ',', value_enum, default_value = "bf16")]
    data_type: Vec<DataType>,

    /// Distance metrics (comma-separated for sweep)
    #[arg(long, value_delimiter = ',', value_enum, default_value = "l2")]
    metric: Vec<Metric>,

    /// HNSW connectivity M (comma-separated for sweep)
    #[arg(long, value_delimiter = ',', default_value = "16", value_parser = retrieval::parse_count_flag)]
    connectivity: Vec<usize>,

    /// HNSW construction expansion factor (comma-separated for sweep)
    #[arg(long, value_delimiter = ',', default_value = "128", value_parser = retrieval::parse_count_flag)]
    expansion_add: Vec<usize>,

    /// HNSW search expansion factor (comma-separated for sweep)
    #[arg(long, value_delimiter = ',', default_value = "64", value_parser = retrieval::parse_count_flag)]
    expansion_search: Vec<usize>,

    /// Number of index shards (comma-separated for sweep)
    #[arg(long, value_delimiter = ',', default_value = "1", value_parser = retrieval::parse_count_flag)]
    shards: Vec<usize>,

    /// Number of threads, 0 for all cores (comma-separated for sweep)
    #[arg(long, value_delimiter = ',', default_value = "0", value_parser = retrieval::parse_threads_flag)]
    threads: Vec<Threads>,
}

/// USearch quantization types, as `--data-type` spells them.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataType {
    F64,
    F32,
    Bf16,
    F16,
    E5m2,
    E4m3,
    E3m2,
    E2m3,
    I8,
    U8,
    B1,
}

impl From<DataType> for ::usearch::ScalarKind {
    fn from(data_type: DataType) -> Self {
        match data_type {
            DataType::F64 => Self::F64,
            DataType::F32 => Self::F32,
            DataType::Bf16 => Self::BF16,
            DataType::F16 => Self::F16,
            DataType::E5m2 => Self::E5M2,
            DataType::E4m3 => Self::E4M3,
            DataType::E3m2 => Self::E3M2,
            DataType::E2m3 => Self::E2M3,
            DataType::I8 => Self::I8,
            DataType::U8 => Self::U8,
            DataType::B1 => Self::B1,
        }
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        retrieval::spell_value(self, formatter)
    }
}

/// USearch metrics, as `--metric` spells them.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    Ip,
    #[value(alias = "l2sq")]
    L2,
    Cos,
    Hamming,
    #[value(alias = "tanimoto")]
    Jaccard,
    Sorensen,
    Pearson,
    Haversine,
    #[value(alias = "jensenshannon")]
    Divergence,
}

impl From<Metric> for ::usearch::MetricKind {
    fn from(metric: Metric) -> Self {
        match metric {
            Metric::Ip => Self::IP,
            Metric::L2 => Self::L2sq,
            Metric::Cos => Self::Cos,
            Metric::Hamming => Self::Hamming,
            Metric::Jaccard => Self::Tanimoto,
            Metric::Sorensen => Self::Sorensen,
            Metric::Pearson => Self::Pearson,
            Metric::Haversine => Self::Haversine,
            Metric::Divergence => Self::Divergence,
        }
    }
}

impl fmt::Display for Metric {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        retrieval::spell_value(self, formatter)
    }
}

fn metric_kind_name(kind: ::usearch::MetricKind) -> &'static str {
    match kind {
        ::usearch::MetricKind::IP => "ip",
        ::usearch::MetricKind::L2sq => "l2",
        ::usearch::MetricKind::Cos => "cos",
        ::usearch::MetricKind::Hamming => "hamming",
        ::usearch::MetricKind::Tanimoto => "jaccard",
        ::usearch::MetricKind::Sorensen => "sorensen",
        ::usearch::MetricKind::Pearson => "pearson",
        ::usearch::MetricKind::Haversine => "haversine",
        ::usearch::MetricKind::Divergence => "divergence",
        _ => "unknown",
    }
}

fn scalar_kind_name(kind: ::usearch::ScalarKind) -> &'static str {
    match kind {
        ::usearch::ScalarKind::F64 => "f64",
        ::usearch::ScalarKind::F32 => "f32",
        ::usearch::ScalarKind::BF16 => "bf16",
        ::usearch::ScalarKind::F16 => "f16",
        ::usearch::ScalarKind::E5M2 => "e5m2",
        ::usearch::ScalarKind::E4M3 => "e4m3",
        ::usearch::ScalarKind::E3M2 => "e3m2",
        ::usearch::ScalarKind::E2M3 => "e2m3",
        ::usearch::ScalarKind::I8 => "i8",
        ::usearch::ScalarKind::U8 => "u8",
        ::usearch::ScalarKind::B1 => "b1",
        _ => "unknown",
    }
}

// #region Backend

pub struct USearchBackend {
    shards: Vec<::usearch::Index, std::alloc::System>,
    pool: ThreadPool,
    description: String,
    metadata: std::collections::HashMap<String, serde_json::Value>,
}

impl USearchBackend {
    pub fn new(config: IndexConfig<DataType, Metric>, threads: Threads, shards: usize) -> Result<Self, String> {
        let IndexConfig {
            dimensions,
            data_type,
            metric,
            connectivity,
            expansion_add,
            expansion_search,
        } = config;

        let opts = ::usearch::IndexOptions {
            dimensions,
            metric: metric.into(),
            quantization: data_type.into(),
            connectivity,
            expansion_add,
            expansion_search,
            multi: false,
        };

        let mut shard_vec = Vec::with_capacity_in(shards, std::alloc::System);
        for _ in 0..shards {
            shard_vec.push(::usearch::Index::new(&opts).map_err(|e| format!("failed to create USearch index: {e}"))?);
        }

        if let Some(idx) = shard_vec.first() {
            eprintln!("  dispatch[{data_type}/{metric}]: {}", idx.hardware_acceleration());
        }

        let pool = ThreadPool::try_spawn(threads.0.get()).map_err(|e| format!("failed to create thread pool: {e}"))?;

        let mut description = format!(
            "usearch · {data_type} · {metric} · M={connectivity} · ef={expansion_add}/{expansion_search} · {threads} threads"
        );
        if shards > 1 {
            write!(description, " · {shards} shards").unwrap();
        }

        let mut metadata = std::collections::HashMap::new();
        metadata.insert("backend".into(), json!("usearch"));
        metadata.insert("library_version".into(), json!(usearch::version()));
        metadata.insert(
            "library_isa_compiled".into(),
            json!(usearch::hardware_acceleration_compiled()),
        );
        metadata.insert(
            "library_isa_available".into(),
            json!(usearch::hardware_acceleration_available()),
        );
        metadata.insert("data_type".into(), json!(data_type.to_string()));
        metadata.insert("metric".into(), json!(metric.to_string()));
        metadata.insert("connectivity".into(), json!(connectivity));
        metadata.insert("expansion_add".into(), json!(expansion_add));
        metadata.insert("expansion_search".into(), json!(expansion_search));
        metadata.insert("threads".into(), json!(threads.0.get()));
        metadata.insert("shards".into(), json!(shards));

        Ok(Self {
            shards: shard_vec,
            pool,
            description,
            metadata,
        })
    }

    fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// Path of the i-th shard's saved file. With `shards == 1` the user's path
    /// is used as-is; with multiple shards each shard appends `.shard_<i>` so
    /// the on-disk layout is unambiguous.
    fn shard_path(handle: &str, shard_index: usize, shard_count: usize) -> String {
        if shard_count <= 1 {
            handle.to_string()
        } else {
            format!("{handle}.shard_{shard_index}")
        }
    }

    /// Sibling of `new` for opening a previously-saved index. Build-time
    /// params (dimensions, metric, quantization, connectivity, expansion_add)
    /// live in the file header and are read back via accessors after load.
    /// Only true runtime knobs go on the call: where to load from, how many
    /// shards on disk, how many threads to drive search with, and the
    /// `expansion_search` that replaces the file's value.
    pub fn load(handle: &str, expansion_search: usize, threads: Threads, shards: usize) -> Result<Self, String> {
        // Throw-away opts: `Index::load` reinitializes the index from the
        // file header, so all of these get overwritten on load.
        let placeholder_opts = ::usearch::IndexOptions {
            dimensions: 1,
            metric: ::usearch::MetricKind::L2sq,
            quantization: ::usearch::ScalarKind::F32,
            connectivity: 0,
            expansion_add: 0,
            expansion_search: 0,
            multi: false,
        };

        let mut shard_vec = Vec::with_capacity_in(shards, std::alloc::System);
        for shard_index in 0..shards {
            let idx =
                ::usearch::Index::new(&placeholder_opts).map_err(|e| format!("failed to create USearch index: {e}"))?;
            let path = Self::shard_path(handle, shard_index, shards);
            idx.load(&path)
                .map_err(|e| format!("USearch load({path}) failed: {e}"))?;
            idx.change_expansion_search(expansion_search);
            shard_vec.push(idx);
        }

        let pool = ThreadPool::try_spawn(threads.0.get()).map_err(|e| format!("failed to create thread pool: {e}"))?;

        // Read the file's actual values back for description + metadata. The
        // vector width is not among them: `save_report` records the width the
        // harness sliced queries at, which is what produced the results.
        let head = &shard_vec[0];
        let connectivity = head.connectivity();
        let exp_add = head.expansion_add();
        let exp_search = head.expansion_search();
        let metric_name = metric_kind_name(head.metric_kind());
        let data_type_name = scalar_kind_name(head.scalar_kind());

        eprintln!(
            "  dispatch[{data_type_name}/{metric_name}]: {}",
            head.hardware_acceleration()
        );

        let mut description = format!(
            "usearch · {data_type_name} · {metric_name} · M={connectivity} · ef={exp_add}/{exp_search} · {threads} threads · loaded[{handle}]",
        );
        if shards > 1 {
            write!(description, " · {shards} shards").unwrap();
        }

        let mut metadata = std::collections::HashMap::new();
        metadata.insert("backend".into(), json!("usearch"));
        metadata.insert("library_version".into(), json!(usearch::version()));
        metadata.insert(
            "library_isa_compiled".into(),
            json!(usearch::hardware_acceleration_compiled()),
        );
        metadata.insert(
            "library_isa_available".into(),
            json!(usearch::hardware_acceleration_available()),
        );
        metadata.insert("data_type".into(), json!(data_type_name));
        metadata.insert("metric".into(), json!(metric_name));
        metadata.insert("connectivity".into(), json!(connectivity));
        metadata.insert("expansion_add".into(), json!(exp_add));
        metadata.insert("expansion_search".into(), json!(exp_search));
        metadata.insert("threads".into(), json!(threads.0.get()));
        metadata.insert("shards".into(), json!(shards));
        metadata.insert("loaded_from".into(), json!(handle));

        Ok(Self {
            shards: shard_vec,
            pool,
            description,
            metadata,
        })
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
        let pool = &mut self.pool;
        let threads = pool.threads();

        let per_shard = num_vectors.div_ceil(shard_count);
        for shard in &self.shards {
            shard
                .reserve_capacity_and_threads(shard.size() + per_shard, threads)
                .map_err(|e| format!("failed to reserve capacity: {e}"))?;
        }

        let failed = AtomicBool::new(false);
        let split = IndexedSplit::new(num_vectors, threads);
        let shards = &self.shards;

        pool.for_threads(|thread_index, _| {
            for vector_index in split.get(thread_index) {
                if failed.load(Ordering::Relaxed) {
                    return;
                }
                let shard = &shards[vector_index % shard_count];
                let ok = match vectors.data {
                    VectorSlice::F32(data) => shard
                        .add(
                            keys[vector_index] as u64,
                            &data[vector_index * dimensions..(vector_index + 1) * dimensions],
                        )
                        .is_ok(),
                    VectorSlice::I8(data) => shard
                        .add(
                            keys[vector_index] as u64,
                            &data[vector_index * dimensions..(vector_index + 1) * dimensions],
                        )
                        .is_ok(),
                    VectorSlice::U8(data) => shard
                        .add(
                            keys[vector_index] as u64,
                            &data[vector_index * dimensions..(vector_index + 1) * dimensions],
                        )
                        .is_ok(),
                    VectorSlice::B1x8(data) => {
                        let stride = dimensions.div_ceil(8);
                        shard
                            .add(
                                keys[vector_index] as u64,
                                ::usearch::b1x8::from_u8s(&data[vector_index * stride..(vector_index + 1) * stride]),
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
        &mut self,
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
        let pool = &mut self.pool;
        let threads = pool.threads();
        let failed = AtomicBool::new(false);
        let split = IndexedSplit::new(num_queries, threads);
        let shards = &self.shards;

        pool.for_threads(|thread_index, _| {
            for query_index in split.get(thread_index) {
                if failed.load(Ordering::Relaxed) {
                    return;
                }

                let offset = query_index * count;
                unsafe {
                    let keys = keys_ptr.as_ptr().add(offset);
                    let distances = dists_ptr.as_ptr().add(offset);
                    for rank in 0..count {
                        *keys.add(rank) = Key::MAX;
                        *distances.add(rank) = Distance::INFINITY;
                    }
                }

                for shard in shards {
                    let result = match queries.data {
                        VectorSlice::F32(data) => {
                            shard.search(&data[query_index * dimensions..(query_index + 1) * dimensions], count)
                        }
                        VectorSlice::I8(data) => {
                            shard.search(&data[query_index * dimensions..(query_index + 1) * dimensions], count)
                        }
                        VectorSlice::U8(data) => {
                            shard.search(&data[query_index * dimensions..(query_index + 1) * dimensions], count)
                        }
                        VectorSlice::B1x8(data) => {
                            let stride = dimensions.div_ceil(8);
                            shard.search(
                                ::usearch::b1x8::from_u8s(&data[query_index * stride..(query_index + 1) * stride]),
                                count,
                            )
                        }
                    };

                    let matches = match result {
                        Ok(matches) => matches,
                        Err(_) => {
                            failed.store(true, Ordering::Relaxed);
                            return;
                        }
                    };

                    unsafe {
                        let keys = keys_ptr.as_ptr().add(offset);
                        let distances = dists_ptr.as_ptr().add(offset);

                        for match_rank in 0..matches.keys.len().min(count) {
                            let distance = matches.distances[match_rank];
                            let key = matches.keys[match_rank] as Key;
                            if distance >= *distances.add(count - 1) {
                                break;
                            }
                            let mut position = count - 1;
                            while position > 0 && distance < *distances.add(position - 1) {
                                *keys.add(position) = *keys.add(position - 1);
                                *distances.add(position) = *distances.add(position - 1);
                                position -= 1;
                            }
                            *keys.add(position) = key;
                            *distances.add(position) = distance;
                        }
                    }
                }

                unsafe {
                    let keys = keys_ptr.as_ptr().add(offset);
                    let mut found = 0;
                    for rank in 0..count {
                        if *keys.add(rank) != Key::MAX {
                            found += 1;
                        } else {
                            break;
                        }
                    }
                    *counts_ptr.as_ptr().add(query_index) = found;
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

    fn save(&self, handle: &str) -> Result<(), String> {
        let shard_count = self.shard_count();
        for (shard_index, shard) in self.shards.iter().enumerate() {
            let path = Self::shard_path(handle, shard_index, shard_count);
            shard
                .save(&path)
                .map_err(|e| format!("USearch save({path}) failed: {e}"))?;
        }
        Ok(())
    }
}

// #region main

fn main() {
    let cli: Cli = retrieval::parse_cli();

    eprintln!("USearch {}", usearch::version());
    eprintln!("- Compiled for: {}", usearch::hardware_acceleration_compiled());
    eprintln!("- This machine: {}", usearch::hardware_acceleration_available());

    let mut state = BenchState::load(&cli.common).unwrap_or_bail("benchmark state");
    eprintln!("- Data types: {}", spell_list(&cli.data_type));
    eprintln!("- Metrics: {}", spell_list(&cli.metric));
    eprintln!("- Connectivity: {}", spell_list(&cli.connectivity));
    eprintln!("- Expansion add: {}", spell_list(&cli.expansion_add));
    eprintln!("- Expansion search: {}", spell_list(&cli.expansion_search));
    eprintln!("- Shards: {}", spell_list(&cli.shards));
    eprintln!("- Threads: {}", spell_list(&cli.threads));
    let dimensions_sweep = cli.common.dimensions_sweep(state.dimensions());

    cli.common.ensure_single_config(&[
        dimensions_sweep.len(),
        cli.data_type.len(),
        cli.metric.len(),
        cli.connectivity.len(),
        cli.expansion_add.len(),
        cli.expansion_search.len(),
        cli.shards.len(),
        cli.threads.len(),
    ]);

    let mut summary = SweepSummary::default();
    for (dimensions, data_type, metric, connectivity, expansion_add, expansion_search, shards, threads) in itertools::iproduct!(
        &dimensions_sweep,
        &cli.data_type,
        &cli.metric,
        &cli.connectivity,
        &cli.expansion_add,
        &cli.expansion_search,
        &cli.shards,
        &cli.threads
    ) {
        state.check_dimensions(*dimensions).unwrap_or_bail("invalid --dims");

        let description = format!(
            "usearch · {data_type} · {metric} · d={dimensions} · M={connectivity} · ef={expansion_add}/{expansion_search} · {threads} threads"
        );

        summary.record(run_config(
            &description,
            cli.common.index.as_deref(),
            || {
                USearchBackend::new(
                    IndexConfig {
                        dimensions: *dimensions,
                        data_type: *data_type,
                        metric: *metric,
                        connectivity: *connectivity,
                        expansion_add: *expansion_add,
                        expansion_search: *expansion_search,
                    },
                    *threads,
                    *shards,
                )
            },
            |h| USearchBackend::load(h, *expansion_search, *threads, *shards),
            &mut state,
            *dimensions,
        ));
    }

    summary.print();
}
