//! USearch NSW benchmark binary.
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
//!     --data-type bf16,e4m3,i8 \
//!     --metric ip,cos,l2 \
//!     --output results/
//! ```
//!
//! Turing 10M at 99% recall (M=48, ef=768/384):
//! ```sh
//! retri-eval-usearch \
//!     --base-vectors datasets/turing_10M/base.10M.fbin \
//!     --query-vectors datasets/turing_10M/query.public.100K.fbin \
//!     --query-neighbors datasets/turing_10M/groundtruth.public.100K.ibin \
//!     --data-type bf16,e4m3,i8 \
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
//!     --data-type bf16 \
//!     --memory-domains 0,1 \
//!     --metric l2 \
//!     --connectivity 32 \
//!     --expansion-add 256 \
//!     --expansion-search 1024 \
//!     --steps 20 \
//!     --output results/turing_100M
//! ```
//!
//! Binary 1M Hamming-distance search (1024-bit vectors in `.b1bin` format):
//! ```sh
//! retri-eval-usearch \
//!     --base-vectors datasets/binary_1M/base.1M.b1bin \
//!     --query-vectors datasets/binary_1M/query.10K.b1bin \
//!     --query-neighbors datasets/binary_1M/groundtruth.10K.ibin \
//!     --data-type b1 \
//!     --metric hamming \
//!     --output results/binary_1M
//! ```

use std::{collections::HashMap, fmt::Write};

use clap::Parser;
use numkong as nk;
use serde_json::{json, Value};
use usearch::{
    Arena, DenseIndex, DenseInsertConfig, DenseLoadRamConfig, DenseMetric, DenseOptions, DenseSearchConfig, DenseVerbs,
    Element, Executor, Scalar,
};

use retrieval::{
    run_config, spell_list, Backend, BatchKeys, BenchState, CommonArgs, Distance, Hits, IndexConfig, IndexFile, Key,
    SweepSummary, Threads, UnwrapOrBail, Vectors,
};

#[derive(Parser, Debug)]
#[command(name = "retri-eval-usearch", about = "Benchmark USearch NSW")]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,

    /// Stored quantization types (comma-separated for sweep)
    #[arg(
        long,
        env = "RETRIEVAL_DATA_TYPE",
        value_delimiter = ',',
        value_enum,
        default_value = "bf16"
    )]
    data_type: Vec<DataType>,

    /// Distance metrics (comma-separated for sweep)
    #[arg(
        long,
        env = "RETRIEVAL_METRIC",
        value_delimiter = ',',
        value_enum,
        default_value = "l2"
    )]
    metric: Vec<Metric>,

    /// Graph connectivity (comma-separated for sweep)
    #[arg(long, env = "RETRIEVAL_CONNECTIVITY", value_delimiter = ',', default_value = "16", value_parser = retrieval::parse_count_flag)]
    connectivity: Vec<usize>,

    /// Construction expansion factor (comma-separated for sweep)
    #[arg(long, env = "RETRIEVAL_EXPANSION_ADD", value_delimiter = ',', default_value = "128", value_parser = retrieval::parse_count_flag)]
    expansion_add: Vec<usize>,

    /// Search expansion factor (comma-separated for sweep)
    #[arg(long, env = "RETRIEVAL_EXPANSION_SEARCH", value_delimiter = ',', default_value = "64", value_parser = retrieval::parse_count_flag)]
    expansion_search: Vec<usize>,

    /// NUMA nodes to place one shard on apiece; empty takes every node the process may use
    #[arg(long, env = "RETRIEVAL_MEMORY_DOMAINS", value_delimiter = ',')]
    memory_domains: Vec<i32>,

    /// Number of threads, 0 for all cores (comma-separated for sweep)
    #[arg(long, env = "RETRIEVAL_THREADS", value_delimiter = ',', default_value = "0", value_parser = retrieval::parse_threads_flag)]
    threads: Vec<Threads>,

    /// Persisted index: loaded and searched if the path exists, else built and saved; needs a single config
    #[arg(long, env = "RETRIEVAL_INDEX", value_parser = clap::builder::NonEmptyStringValueParser::new())]
    index: Option<String>,
}

/// Stored quantization types, as `--data-type` spells them.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataType {
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

impl From<DataType> for Scalar {
    fn from(data_type: DataType) -> Self {
        match data_type {
            DataType::Bf16 => Self::BF16,
            DataType::F16 => Self::F16,
            DataType::E5m2 => Self::E5M2,
            DataType::E4m3 => Self::E4M3,
            DataType::E3m2 => Self::E3M2,
            DataType::E2m3 => Self::E2M3,
            DataType::I8 => Self::I8,
            DataType::U8 => Self::U8,
            DataType::B1 => Self::U1x8,
        }
    }
}

/// Metrics, as `--metric` spells them.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    Ip,
    L2,
    Cos,
    Hamming,
    #[value(alias = "tanimoto")]
    Jaccard,
}

impl From<Metric> for DenseMetric {
    fn from(metric: Metric) -> Self {
        match metric {
            Metric::Ip => Self::Inner,
            Metric::L2 => Self::Euclidean,
            Metric::Cos => Self::Angular,
            Metric::Hamming => Self::Hamming,
            Metric::Jaccard => Self::Jaccard,
        }
    }
}

impl std::fmt::Display for DataType {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        retrieval::spell_value(self, formatter)
    }
}

impl std::fmt::Display for Metric {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        retrieval::spell_value(self, formatter)
    }
}

// #region Backend

pub struct USearchBackend {
    index: DenseIndex,
    executor: Executor,
    arena: Arena,
    keys: Vec<usearch::Key>,
    expansion_add: usize,
    expansion_search: usize,
    metadata: HashMap<String, Value>,
}

fn failed(verb: &'static str) -> impl Fn(usearch::Error) -> String {
    move |error| format!("USearch {verb} failed with {}: {error}", error.status)
}

impl USearchBackend {
    pub fn new(
        config: IndexConfig<DataType, Metric>,
        threads: Threads,
        memory_domains: &[i32],
    ) -> Result<Self, String> {
        let options = DenseOptions {
            dimensions: config.dimensions,
            metric: config.metric.into(),
            quantization: config.data_type.into(),
            connectivity: config.connectivity,
            ..DenseOptions::default()
        };
        let index = DenseIndex::init_ram(&options, memory_domains).map_err(failed("init_ram"))?;
        let executor = Executor::for_ram(memory_domains, threads.0.get(), 0).map_err(failed("for_ram"))?;
        Self::wrap(
            index,
            executor,
            config.expansion_add,
            config.expansion_search,
            threads,
            memory_domains,
        )
    }

    /// Opens a saved `.safegraphs` monolith or `.safegraphs.json` manifest, whose header carries the
    /// geometry, metric, quantization and connectivity; only the search expansion is the caller's.
    pub fn load(path: &str, expansion_search: usize, threads: Threads, memory_domains: &[i32]) -> Result<Self, String> {
        let executor = Executor::for_ram(memory_domains, threads.0.get(), 0).map_err(failed("for_ram"))?;
        let index = DenseIndex::load_ram(DenseLoadRamConfig {
            executor: Some(&executor),
            memory_domains,
            ..DenseLoadRamConfig::from(path)
        })
        .map_err(failed("load_ram"))?;
        let mut backend = Self::wrap(index, executor, 0, expansion_search, threads, memory_domains)?;
        backend.metadata.insert("loaded_from".into(), json!(path));
        Ok(backend)
    }

    fn wrap(
        index: DenseIndex,
        executor: Executor,
        expansion_add: usize,
        expansion_search: usize,
        threads: Threads,
        memory_domains: &[i32],
    ) -> Result<Self, String> {
        let described = index.describe().map_err(failed("describe"))?;
        let metadata = described.metadata;
        let (metric, quantization, connectivity) = (metadata.metric, metadata.quantization, metadata.connectivity);
        eprintln!("  dispatch[{quantization}/{metric}]: {}", described.capabilities);
        let metadata = HashMap::from([
            ("backend".into(), json!("usearch")),
            ("library_version".into(), json!(usearch::version())),
            ("library_isa_compiled".into(), json!(usearch::capabilities_compiled())),
            ("library_isa_available".into(), json!(usearch::capabilities_available())),
            ("library_isa_dispatched".into(), json!(described.capabilities)),
            ("data_type".into(), json!(quantization.to_string())),
            ("metric".into(), json!(metric.to_string())),
            ("connectivity".into(), json!(connectivity)),
            ("expansion_add".into(), json!(expansion_add)),
            ("expansion_search".into(), json!(expansion_search)),
            ("threads".into(), json!(threads.0.get())),
            ("memory_domains".into(), json!(memory_domains)),
        ]);
        Ok(Self {
            arena: Arena::new_in(&executor).map_err(failed("arena"))?,
            index,
            executor,
            keys: Vec::new(),
            expansion_add,
            expansion_search,
            metadata,
        })
    }

    fn insert<Value: Element>(&self, vectors: nk::MatrixView<'_, Value>) -> usearch::Result<()> {
        self.index.on(&self.executor).insert(
            &self.keys,
            DenseInsertConfig {
                expansion_factor: self.expansion_add,
                ..DenseInsertConfig::from(vectors)
            },
        )
    }
}

impl Backend for USearchBackend {
    fn metadata(&self) -> HashMap<String, Value> {
        self.metadata.clone()
    }

    fn add(&mut self, keys: BatchKeys<'_>, vectors: Vectors) -> Result<(), String> {
        self.keys.clear();
        self.keys.extend(keys.keys().map(|key| usearch::Key::from(key.0)));
        let size = self.index.describe().map_err(failed("describe"))?.size;
        self.index.reserve(size + keys.len()).map_err(failed("reserve"))?;
        match vectors {
            Vectors::F32(view) => self.insert(view),
            Vectors::I8(view) => self.insert(view),
            Vectors::U8(view) => self.insert(view),
            Vectors::B1x8(view) => self.insert(view),
        }
        .map_err(failed("insert"))
    }

    fn search(&mut self, queries: Vectors, mut hits: Hits<'_>) -> Result<(), String> {
        let config = DenseSearchConfig {
            expansion_factor: self.expansion_search,
            ..DenseSearchConfig::from(hits.count())
        };
        let receiver = self.index.on(&self.executor);
        let arena = &mut self.arena;
        let matches = match queries {
            Vectors::F32(view) => receiver.search(view, config, arena),
            Vectors::I8(view) => receiver.search(view, config, arena),
            Vectors::U8(view) => receiver.search(view, config, arena),
            Vectors::B1x8(view) => receiver.search(view, config, arena),
        }
        .map_err(failed("search"))?;

        for (found, row) in matches.rows().map_err(failed("search"))?.zip(hits.rows()) {
            row.write(
                found
                    .keys
                    .iter()
                    .zip(found.distances)
                    .map(|(&key, &distance)| Ok((Key(u128::from(key)), distance as Distance))),
            )?;
        }
        Ok(())
    }

    fn memory_bytes(&self) -> usize {
        self.index.describe().map_or(0, |described| described.memory_usage)
    }

    fn indexed_count(&self) -> Option<usize> {
        self.index.describe().ok().map(|described| described.size)
    }

    fn save(&self, path: &str) -> Result<(), String> {
        self.index.on(&self.executor).save(path).map_err(failed("save"))
    }
}

// #region main

fn main() {
    let cli: Cli = retrieval::parse_cli();

    eprintln!("USearch {}", usearch::version());
    eprintln!("- Compiled for: {}", usearch::capabilities_compiled());
    eprintln!("- This machine: {}", usearch::capabilities_available());

    let mut state = BenchState::load(&cli.common).unwrap_or_bail("benchmark state");
    eprintln!("- Data types: {}", spell_list(&cli.data_type));
    eprintln!("- Metrics: {}", spell_list(&cli.metric));
    eprintln!("- Connectivity: {}", spell_list(&cli.connectivity));
    eprintln!("- Expansion add: {}", spell_list(&cli.expansion_add));
    eprintln!("- Expansion search: {}", spell_list(&cli.expansion_search));
    eprintln!("- Memory domains: {}", spell_list(&cli.memory_domains));
    eprintln!("- Threads: {}", spell_list(&cli.threads));
    let dimensions_sweep = cli.common.dimensions_sweep(state.dimensions());
    let configs = itertools::iproduct!(
        &dimensions_sweep,
        &cli.data_type,
        &cli.metric,
        &cli.connectivity,
        &cli.expansion_add,
        &cli.expansion_search,
        &cli.threads
    );
    let index_file = IndexFile::new(cli.index.as_deref(), configs.clone().count());
    eprintln!("- Index: {index_file}");

    let mut summary = SweepSummary::default();
    for (&dimensions, &data_type, &metric, &connectivity, &expansion_add, &expansion_search, &threads) in configs {
        let mut description = format!(
            "usearch · {data_type} · {metric} · d={dimensions} · M={connectivity} · ef={expansion_add}/{expansion_search} · {threads} threads"
        );
        if !cli.memory_domains.is_empty() {
            write!(description, " · domains {}", spell_list(&cli.memory_domains)).expect("a String takes any write");
        }
        let backend = match index_file {
            IndexFile::Load(path) => USearchBackend::load(path, expansion_search, threads, &cli.memory_domains),
            IndexFile::None | IndexFile::Save(_) => USearchBackend::new(
                IndexConfig {
                    dimensions,
                    data_type,
                    metric,
                    connectivity,
                    expansion_add,
                    expansion_search,
                },
                threads,
                &cli.memory_domains,
            ),
        };
        summary.record(run_config(&description, index_file, backend, &mut state, dimensions));
    }

    summary.finish();
}
