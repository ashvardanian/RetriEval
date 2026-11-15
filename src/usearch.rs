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
//!     --metric inner,angular,euclidean \
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
//!     --metric euclidean \
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
//!     --metric euclidean \
//!     --connectivity 32 \
//!     --expansion-add 256 \
//!     --expansion-search 1024 \
//!     --steps 20 \
//!     --output results/turing_100M
//! ```
//!
//! PubChem built in RAM, refined up to twice, saved and searched as an NVMe view:
//! ```sh
//! retri-eval-usearch \
//!     --base-vectors datasets/pubchem/base.b1bin \
//!     --query-vectors datasets/pubchem/query.10K.b1bin \
//!     --query-neighbors datasets/pubchem/groundtruth.10K.ibin \
//!     --data-type b1 \
//!     --metric jaccard \
//!     --steps 1 \
//!     --max-refines 2 \
//!     --residency nvme \
//!     --nvme-paths /mnt/nvme0 \
//!     --index /mnt/nvme0/pubchem.safegraphs.json \
//!     --output results/pubchem
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

use std::{
    collections::HashMap,
    fmt::Write,
    path::{Path, PathBuf},
    time::Instant,
};

use clap::Parser;
use numkong as nk;
use serde_json::{json, Value};
use usearch::{
    Arena, DenseIndex, DenseInsertConfig, DenseLoadRamConfig, DenseLoadVramConfig, DenseMetric, DenseOptions,
    DenseRefineConfig, DenseSearchConfig, DenseVerbs, DenseViewNvmeConfig, Element, Executor, RefineVerdict, Residency,
    Scalar,
};

use retrieval::{
    run_config, spell_list, Backend, BatchKeys, BenchState, CommonArgs, Distance, Hits, IndexConfig, IndexFile, Key,
    StepRefineEntry, SweepSummary, Threads, UnwrapOrBail, Vectors,
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
        default_value = "euclidean"
    )]
    metric: Vec<Metric>,

    /// Graph connectivity (comma-separated for sweep); 0 lets the library derive it
    #[arg(long, env = "RETRIEVAL_CONNECTIVITY", value_delimiter = ',', default_value = "0", value_parser = retrieval::parse_count_or_zero_flag)]
    connectivity: Vec<usize>,

    /// Construction expansion factor (comma-separated for sweep); 0 lets the library derive it
    #[arg(long, env = "RETRIEVAL_EXPANSION_ADD", value_delimiter = ',', default_value = "0", value_parser = retrieval::parse_count_or_zero_flag)]
    expansion_add: Vec<usize>,

    /// Search expansion factor (comma-separated for sweep); 0 lets the library derive it
    #[arg(long, env = "RETRIEVAL_EXPANSION_SEARCH", value_delimiter = ',', default_value = "0", value_parser = retrieval::parse_count_or_zero_flag)]
    expansion_search: Vec<usize>,

    /// NUMA nodes to place one shard on apiece; empty takes every node the process may use
    #[arg(long, env = "RETRIEVAL_MEMORY_DOMAINS", value_delimiter = ',')]
    memory_domains: Vec<i32>,

    /// Where the index serves searches from: built in RAM or VRAM, or built in RAM, saved to
    /// `--index` and searched as an NVMe view on the last step
    #[arg(long, env = "RETRIEVAL_RESIDENCY", default_value = "ram", value_parser = |text: &str| text.parse::<Residency>())]
    residency: Residency,

    /// Refinement rounds after the last insert step, until one is not improved
    #[arg(long, env = "RETRIEVAL_MAX_REFINES", default_value_t = 0)]
    max_refines: usize,

    /// Devices or directories an NVMe view serves its parts from; empty serves them where `--index` points
    #[arg(long, env = "RETRIEVAL_NVME_PATHS", value_delimiter = ',')]
    nvme_paths: Vec<PathBuf>,

    /// GPU ordinals to place one shard on apiece; empty derives one free device
    #[arg(long, env = "RETRIEVAL_DEVICES", value_delimiter = ',')]
    devices: Vec<i32>,

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
    #[value(alias = "ip")]
    Inner,
    #[value(alias = "l2")]
    Euclidean,
    #[value(alias = "cos")]
    Angular,
    Hamming,
    #[value(alias = "tanimoto")]
    Jaccard,
}

impl From<Metric> for DenseMetric {
    fn from(metric: Metric) -> Self {
        match metric {
            Metric::Inner => Self::Inner,
            Metric::Euclidean => Self::Euclidean,
            Metric::Angular => Self::Angular,
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

/// Where every config of a sweep places its index and how its build finishes, as the flags say.
#[derive(Clone, Copy)]
pub struct Placement<'cli> {
    residency: Residency,
    memory_domains: &'cli [i32],
    devices: &'cli [i32],
    nvme_paths: &'cli [PathBuf],
    max_refines: usize,
    /// The `--index` path an NVMe view is saved to and served from.
    checkpoint: Option<&'cli str>,
}

pub struct USearchBackend<'cli> {
    index: DenseIndex,
    executor: Executor,
    arena: Arena,
    keys: Vec<usearch::Key>,
    expansion_add: usize,
    expansion_search: usize,
    placement: Placement<'cli>,
    metadata: HashMap<String, Value>,
}

fn failed(verb: &'static str) -> impl Fn(usearch::Error) -> String {
    move |error| format!("USearch {verb} failed with {}: {error}", error.status)
}

/// Opens the checkpoint at `path` as a view served from `nvme_paths`, on an NVMe executor of its own.
fn view_nvme(path: &str, nvme_paths: &[PathBuf]) -> Result<(DenseIndex, Executor), String> {
    let served: Vec<&Path> = nvme_paths.iter().map(PathBuf::as_path).collect();
    let rings = if served.is_empty() {
        vec![Path::new(path)]
    } else {
        served.clone()
    };
    let executor = Executor::for_nvme(&rings, None, 0).map_err(failed("for_nvme"))?;
    let index = DenseIndex::view_nvme(DenseViewNvmeConfig {
        executor: Some(&executor),
        paths: &served,
        ..DenseViewNvmeConfig::from(path)
    })
    .map_err(failed("view_nvme"))?;
    Ok((index, executor))
}

impl<'cli> USearchBackend<'cli> {
    pub fn new(
        config: IndexConfig<DataType, Metric>,
        threads: Threads,
        placement: Placement<'cli>,
    ) -> Result<Self, String> {
        let options = DenseOptions {
            dimensions: config.dimensions,
            metric: config.metric.into(),
            quantization: config.data_type.into(),
            connectivity: config.connectivity,
            ..DenseOptions::default()
        };
        // An NVMe view is built in RAM, then saved and reopened once the inserts finish.
        let (index, executor) = match placement.residency {
            Residency::Vram => (
                DenseIndex::init_vram(&options, placement.devices).map_err(failed("init_vram"))?,
                Executor::for_vram(placement.devices, None, 0).map_err(failed("for_vram"))?,
            ),
            Residency::Ram | Residency::Nvme => (
                DenseIndex::init_ram(&options, placement.memory_domains).map_err(failed("init_ram"))?,
                Executor::for_ram(placement.memory_domains, None, threads.0.get()).map_err(failed("for_ram"))?,
            ),
            other => return Err(format!("USearch builds no {other} index here")),
        };
        Self::wrap(
            index,
            executor,
            config.expansion_add,
            config.expansion_search,
            threads,
            placement,
        )
    }

    /// Opens a saved `.safegraphs` monolith or `.safegraphs.json` manifest in the placement's
    /// residency, its header carrying the geometry, metric, quantization and connectivity; only the
    /// search expansion is the caller's.
    pub fn load(
        path: &str,
        expansion_search: usize,
        threads: Threads,
        placement: Placement<'cli>,
    ) -> Result<Self, String> {
        let (index, executor) = match placement.residency {
            Residency::Ram => {
                let executor =
                    Executor::for_ram(placement.memory_domains, None, threads.0.get()).map_err(failed("for_ram"))?;
                let index = DenseIndex::load_ram(DenseLoadRamConfig {
                    executor: Some(&executor),
                    memory_domains: placement.memory_domains,
                    ..DenseLoadRamConfig::from(path)
                })
                .map_err(failed("load_ram"))?;
                (index, executor)
            }
            Residency::Vram => {
                let executor = Executor::for_vram(placement.devices, None, 0).map_err(failed("for_vram"))?;
                let index = DenseIndex::load_vram(DenseLoadVramConfig {
                    executor: Some(&executor),
                    devices: placement.devices,
                    ..DenseLoadVramConfig::from(path)
                })
                .map_err(failed("load_vram"))?;
                (index, executor)
            }
            Residency::Nvme => view_nvme(path, placement.nvme_paths)?,
            other => return Err(format!("USearch opens no {other} index here")),
        };
        let mut backend = Self::wrap(index, executor, 0, expansion_search, threads, placement)?;
        backend.metadata.insert("loaded_from".into(), json!(path));
        Ok(backend)
    }

    fn wrap(
        index: DenseIndex,
        executor: Executor,
        expansion_add: usize,
        expansion_search: usize,
        threads: Threads,
        placement: Placement<'cli>,
    ) -> Result<Self, String> {
        let described = index.describe().map_err(failed("describe"))?;
        let metadata = described.metadata;
        let (metric, quantization, connectivity) = (metadata.metric, metadata.quantization, metadata.connectivity);
        eprintln!("  dispatch[{quantization}/{metric}]: {}", described.cpu_class);
        let metadata = HashMap::from([
            ("backend".into(), json!("usearch")),
            ("library_version".into(), json!(usearch::version())),
            ("library_isa_compiled".into(), json!(usearch::capabilities_compiled())),
            ("library_isa_available".into(), json!(usearch::capabilities_available())),
            ("library_isa_dispatched".into(), json!(described.cpu_class)),
            ("data_type".into(), json!(quantization.to_string())),
            ("metric".into(), json!(metric.to_string())),
            ("connectivity".into(), json!(connectivity)),
            ("expansion_add".into(), json!(expansion_add)),
            ("expansion_search".into(), json!(expansion_search)),
            ("threads".into(), json!(threads.0.get())),
            ("memory_domains".into(), json!(placement.memory_domains)),
            ("residency".into(), json!(placement.residency.to_string())),
            ("max_refines".into(), json!(placement.max_refines)),
        ]);
        Ok(Self {
            arena: Arena::new_in(&executor).map_err(failed("arena"))?,
            index,
            executor,
            keys: Vec::new(),
            expansion_add,
            expansion_search,
            placement,
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

impl Backend for USearchBackend<'_> {
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

        for (found, row) in matches.iter().map_err(failed("search"))?.zip(hits.rows()) {
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

    fn finish(&mut self) -> Result<Option<StepRefineEntry>, String> {
        let refine = match self.placement.max_refines {
            0 => None,
            max_refines => {
                let started = Instant::now();
                let mut changed = Vec::with_capacity(max_refines);
                for _ in 0..max_refines {
                    let report = self
                        .index
                        .on(&self.executor)
                        .refine(DenseRefineConfig::default())
                        .map_err(failed("refine"))?;
                    changed.push(report.changed as u64);
                    if report.verdict != RefineVerdict::Improved {
                        break;
                    }
                }
                let (passes, elapsed) = (changed.len(), started.elapsed().as_secs_f64());
                eprintln!("  refine: {passes} rounds in {elapsed:.2} s, changed {changed:?}");
                Some(StepRefineEntry {
                    passes,
                    elapsed,
                    changed,
                })
            }
        };
        if self.placement.residency == Residency::Nvme {
            let path = self
                .placement
                .checkpoint
                .ok_or("--residency nvme saves its view to --index")?;
            self.index.on(&self.executor).save(path).map_err(failed("save"))?;
            let (index, executor) = view_nvme(path, self.placement.nvme_paths)?;
            self.arena = Arena::new_in(&executor).map_err(failed("arena"))?;
            (self.index, self.executor) = (index, executor);
        }
        Ok(refine)
    }

    fn indexed_count(&self) -> Option<usize> {
        self.index.describe().ok().map(|described| described.size)
    }

    fn save(&self, path: &str) -> Result<(), String> {
        // A view already is the checkpoint `finish` saved.
        if self.placement.residency == Residency::Nvme {
            return Ok(());
        }
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
    eprintln!("- Residency: {}", cli.residency);
    eprintln!("- Max refines: {}", cli.max_refines);
    if cli.residency == Residency::Nvme && cli.index.is_none() {
        retrieval::bail("--residency nvme saves and serves its view at --index");
    }
    let placement = Placement {
        residency: cli.residency,
        memory_domains: &cli.memory_domains,
        devices: &cli.devices,
        nvme_paths: &cli.nvme_paths,
        max_refines: cli.max_refines,
        checkpoint: cli.index.as_deref(),
    };
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
        let backend = match index_file {
            IndexFile::Load(path) => USearchBackend::load(path, expansion_search, threads, placement),
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
                placement,
            ),
        };
        // A zero `--connectivity` lets the library derive it, so the description reads the header's.
        let connectivity = backend
            .as_ref()
            .ok()
            .and_then(|backend| backend.index.describe().ok())
            .map_or(connectivity, |described| described.metadata.connectivity);
        let mut description = format!(
            "usearch · {data_type} · {metric} · d={dimensions} · M={connectivity} · ef={expansion_add}/{expansion_search} · {threads} threads"
        );
        if !cli.memory_domains.is_empty() {
            write!(description, " · domains {}", spell_list(&cli.memory_domains)).expect("a String takes any write");
        }
        write!(description, " · {}", cli.residency).expect("a String takes any write");
        summary.record(run_config(&description, index_file, backend, &mut state, dimensions));
    }

    summary.finish();
}
