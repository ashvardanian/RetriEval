//! Qdrant benchmark binary.
//!
//! ## Prerequisites
//!
//! Requires Docker — the benchmark auto-manages a `qdrant/qdrant` container.
//!
//! ## Build & Install
//!
//! ```sh
//! cargo install --path . --features qdrant-backend
//! ```
//!
//! ## Examples
//!
//! ```sh
//! retri-eval-qdrant \
//!     --base-vectors datasets/wiki_1M/base.1M.fbin \
//!     --query-vectors datasets/wiki_1M/query.public.100K.fbin \
//!     --query-neighbors datasets/wiki_1M/groundtruth.public.100K.ibin \
//!     --metric ip \
//!     --output results/
//! ```

use std::{collections::HashMap, fmt, time::Duration};

use clap::Parser;
use itertools::iproduct;
use qdrant_client::{
    qdrant::{
        point_id, quantization_config::Quantization, BinaryQuantization, CreateCollectionBuilder, Datatype,
        Distance as QdrantDistance, HnswConfigDiffBuilder, Memory, PointId, PointStruct, QuantizationType,
        ScalarQuantization, SearchParamsBuilder, SearchPointsBuilder, UpsertPointsBuilder, VectorParamsBuilder,
    },
    Qdrant,
};
use serde_json::json;

use retrieval::{
    docker::ContainerHandle, run_config, spell_duration, spell_list, Backend, BatchKeys, BenchState, CommonArgs, Hits,
    IndexFile, Key, Port, SweepSummary, UnwrapOrBail, Vectors,
};

const COLLECTION: &str = "bench";

/// Qdrant's own id for `key`: a number when it fits 64 bits, a UUID otherwise.
fn point_id(key: Key) -> PointId {
    u64::try_from(key).map_or_else(|_| PointId::from(key.to_string()), PointId::from)
}

/// Qdrant distance functions, as `--metric` spells them.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum Metric {
    Ip,
    Cos,
    #[value(alias = "l2sq")]
    L2,
    Manhattan,
}

impl From<Metric> for QdrantDistance {
    fn from(metric: Metric) -> Self {
        match metric {
            Metric::Ip => Self::Dot,
            Metric::Cos => Self::Cosine,
            Metric::L2 => Self::Euclid,
            Metric::Manhattan => Self::Manhattan,
        }
    }
}

impl fmt::Display for Metric {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        retrieval::spell_value(self, formatter)
    }
}

/// Qdrant named-vector storage datatype. Server auto-converts from the f32
/// upserts the client sends — no wire-format change needed on our side.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum DataType {
    F32,
    F16,
    U8,
}

impl From<DataType> for Datatype {
    fn from(data_type: DataType) -> Self {
        match data_type {
            DataType::F32 => Self::Float32,
            DataType::F16 => Self::Float16,
            DataType::U8 => Self::Uint8,
        }
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        retrieval::spell_value(self, formatter)
    }
}

/// Server-side quantization — `binary` is deterministic `sign(x)` per dimensions;
/// `scalar` is one-pass per-dimensions min/max mapping to int8. Both stay inside the
/// "no learned codebook" constraint. `product` (k-means) is deliberately not
/// offered here.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum QuantizationKind {
    None,
    Binary,
    Scalar,
}

impl QuantizationKind {
    fn config(self) -> Option<Quantization> {
        match self {
            Self::None => None,
            Self::Binary => Some(Quantization::Binary(BinaryQuantization {
                memory: Some(Memory::Pinned as i32),
                ..Default::default()
            })),
            Self::Scalar => Some(Quantization::Scalar(ScalarQuantization {
                r#type: QuantizationType::Int8 as i32,
                quantile: Some(0.99),
                memory: Some(Memory::Pinned as i32),
                ..Default::default()
            })),
        }
    }
}

impl fmt::Display for QuantizationKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        retrieval::spell_value(self, formatter)
    }
}

// #region CLI

#[derive(Parser, Debug)]
#[command(name = "retri-eval-qdrant", about = "Benchmark Qdrant")]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,

    /// Distance metrics (comma-separated for sweep)
    #[arg(
        long,
        env = "RETRIEVAL_METRIC",
        value_delimiter = ',',
        value_enum,
        default_value = "l2"
    )]
    metric: Vec<Metric>,

    /// Storage data types (comma-separated for sweep)
    #[arg(
        long,
        env = "RETRIEVAL_DATA_TYPE",
        value_delimiter = ',',
        value_enum,
        default_value = "f32"
    )]
    data_type: Vec<DataType>,

    /// Server-side quantization (comma-separated for sweep)
    #[arg(
        long,
        env = "RETRIEVAL_QUANTIZATION",
        value_delimiter = ',',
        value_enum,
        default_value = "none"
    )]
    quantization: Vec<QuantizationKind>,

    /// HNSW connectivity M (comma-separated for sweep)
    #[arg(long, env = "RETRIEVAL_CONNECTIVITY", value_delimiter = ',', default_value = "16", value_parser = retrieval::parse_count_flag)]
    connectivity: Vec<usize>,

    /// HNSW expansion factor during indexing (comma-separated for sweep)
    #[arg(long, env = "RETRIEVAL_EXPANSION_ADD", value_delimiter = ',', default_value = "128", value_parser = retrieval::parse_count_flag)]
    expansion_add: Vec<usize>,

    /// HNSW expansion factor during search — Qdrant's `hnsw_ef` search param
    /// (comma-separated for sweep)
    #[arg(long, env = "RETRIEVAL_EXPANSION_SEARCH", value_delimiter = ',', default_value = "64", value_parser = retrieval::parse_count_flag)]
    expansion_search: Vec<usize>,

    /// Time limit for container start and readiness, like 120s
    #[arg(long, env = "RETRIEVAL_STARTUP_TIME_LIMIT", default_value = "120s", value_parser = retrieval::parse_duration_flag)]
    startup_time_limit: Duration,

    /// gRPC port for Qdrant
    #[arg(long, env = "RETRIEVAL_GRPC_PORT", default_value = "6334", value_parser = retrieval::parse_port)]
    grpc_port: Port,

    /// HTTP port for Qdrant
    #[arg(long, env = "RETRIEVAL_HTTP_PORT", default_value = "6333", value_parser = retrieval::parse_port)]
    http_port: Port,

    /// Vectors per upsert request (distinct from the shared `--inserts-per-core`
    /// / `--queries-per-core`, which pace the harness's add/search loops)
    #[arg(long, env = "RETRIEVAL_VECTORS_PER_UPSERT", default_value_t = 10_000, value_parser = retrieval::parse_count_flag)]
    vectors_per_upsert: usize,
}

// #region Backend

/// One config's collection on the shared server.
struct QdrantBackend<'a> {
    scratch: Vec<f32, std::alloc::System>,
    client: &'a Qdrant,
    container: &'a ContainerHandle,
    runtime: &'a tokio::runtime::Runtime,
    vectors_per_upsert: usize,
    /// Qdrant's `hnsw_ef`, set per request rather than on the collection, so it sweeps without a rebuild.
    expansion_search: usize,
    metadata: HashMap<String, serde_json::Value>,
}

/// The settings that shape one config's collection.
#[derive(Clone, Copy)]
struct CollectionConfig {
    dimensions: usize,
    metric: Metric,
    data_type: DataType,
    quantization: QuantizationKind,
    connectivity: usize,
    expansion_add: usize,
    expansion_search: usize,
}

impl<'a> QdrantBackend<'a> {
    /// Recreates the benchmark collection for `config` on the running server.
    fn new(
        client: &'a Qdrant,
        container: &'a ContainerHandle,
        runtime: &'a tokio::runtime::Runtime,
        vectors_per_upsert: usize,
        config: CollectionConfig,
    ) -> Result<Self, String> {
        let vector_params = VectorParamsBuilder::new(config.dimensions as u64, QdrantDistance::from(config.metric))
            .datatype(Datatype::from(config.data_type));
        let mut create = CreateCollectionBuilder::new(COLLECTION)
            .vectors_config(vector_params)
            .hnsw_config(
                HnswConfigDiffBuilder::default()
                    .m(config.connectivity as u64)
                    .ef_construct(config.expansion_add as u64),
            );
        if let Some(quantization) = config.quantization.config() {
            create = create.quantization_config(quantization);
        }
        runtime
            .block_on(async {
                let _ = client.delete_collection(COLLECTION).await;
                client.create_collection(create).await
            })
            .map_err(|error| format!("Qdrant create collection: {error}"))?;
        Ok(Self {
            scratch: Vec::new_in(std::alloc::System),
            client,
            container,
            runtime,
            vectors_per_upsert,
            expansion_search: config.expansion_search,
            metadata: HashMap::from([
                ("backend".into(), json!("qdrant")),
                ("metric".into(), json!(config.metric.to_string())),
                ("data_type".into(), json!(config.data_type.to_string())),
                ("quantization".into(), json!(config.quantization.to_string())),
                ("connectivity".into(), json!(config.connectivity)),
                ("expansion_add".into(), json!(config.expansion_add)),
                ("expansion_search".into(), json!(config.expansion_search)),
            ]),
        })
    }
}

impl Backend for QdrantBackend<'_> {
    fn metadata(&self) -> HashMap<String, serde_json::Value> {
        self.metadata.clone()
    }

    fn add(&mut self, keys: BatchKeys<'_>, vectors: Vectors) -> Result<(), String> {
        let dimensions = vectors.dimensions();
        let data = vectors.to_f32_in(&mut self.scratch)?;
        let mut keys = keys.keys();
        self.runtime.block_on(async {
            for rows in data.chunks(self.vectors_per_upsert * dimensions) {
                let points: Vec<PointStruct> = keys
                    .by_ref()
                    .zip(rows.chunks_exact(dimensions))
                    .map(|(key, row)| {
                        PointStruct::new(
                            point_id(key),
                            row.to_vec(),
                            HashMap::<String, qdrant_client::qdrant::Value>::new(),
                        )
                    })
                    .collect();
                self.client
                    .upsert_points(UpsertPointsBuilder::new(COLLECTION, points).wait(true))
                    .await
                    .map_err(|error| format!("Qdrant upsert failed: {error}"))?;
            }
            Ok(())
        })
    }

    fn search(&mut self, queries: Vectors, mut hits: Hits<'_>) -> Result<(), String> {
        let count = hits.count();
        let dimensions = queries.dimensions();
        let data = queries.to_f32_in(&mut self.scratch)?;
        let searches: Vec<_> = data
            .chunks_exact(dimensions)
            .map(|row| {
                SearchPointsBuilder::new(COLLECTION, row.to_vec(), count as u64)
                    .params(SearchParamsBuilder::default().hnsw_ef(self.expansion_search as u64))
                    .build()
            })
            .collect();
        let response = self
            .runtime
            .block_on(
                self.client
                    .search_batch_points(qdrant_client::qdrant::SearchBatchPointsBuilder::new(
                        COLLECTION, searches,
                    )),
            )
            .map_err(|error| format!("Qdrant batch search failed: {error}"))?;
        if response.result.len() != queries.len() {
            return Err("Qdrant batch response count mismatch".into());
        }
        for (result, row) in response.result.iter().zip(hits.rows()) {
            row.write(result.result.iter().map(|point| {
                let key = match point.id.as_ref().and_then(|id| id.point_id_options.as_ref()) {
                    Some(point_id::PointIdOptions::Num(id)) => Key::from(*id),
                    Some(point_id::PointIdOptions::Uuid(id)) => id.parse()?,
                    None => return Err("Qdrant returned a point without an ID".to_string()),
                };
                if point.score.is_finite() {
                    Ok((key, point.score))
                } else {
                    Err("Qdrant returned a nonfinite score".into())
                }
            }))?;
        }
        Ok(())
    }

    fn memory_bytes(&self) -> usize {
        self.runtime.block_on(self.container.memory_usage_bytes()) as usize
    }
}

// #region main

/// Reject argument combinations Qdrant will refuse or silently ignore, before a
/// container is started.
fn validate(cli: &Cli) -> Result<(), String> {
    let smallest_expansion = cli.expansion_add.iter().min();
    let largest_connectivity = cli.connectivity.iter().max();
    if let (Some(&expansion_add), Some(&connectivity)) = (smallest_expansion, largest_connectivity) {
        if expansion_add < connectivity {
            return Err(format!(
                "--expansion-add {expansion_add} is below --connectivity {connectivity}; \
                 ef_construct below M gives Qdrant fewer candidates than edges to keep"
            ));
        }
    }
    // Binary quantization thresholds at zero, so it is meaningless on a storage
    // type that is already unsigned and offset — Qdrant accepts the pair and
    // returns noise.
    if cli.quantization.contains(&QuantizationKind::Binary) && cli.data_type.contains(&DataType::U8) {
        return Err("--quantization binary with --data-type u8 quantizes an already-unsigned type".into());
    }
    Ok(())
}

fn main() {
    let cli: Cli = retrieval::parse_cli();
    validate(&cli).unwrap_or_bail("invalid arguments");
    let mut state = BenchState::load(&cli.common).unwrap_or_bail("benchmark state");
    eprintln!("- Metrics: {}", spell_list(&cli.metric));
    eprintln!("- Data types: {}", spell_list(&cli.data_type));
    eprintln!("- Quantization: {}", spell_list(&cli.quantization));
    eprintln!("- Connectivity: {}", spell_list(&cli.connectivity));
    eprintln!("- Expansion add: {}", spell_list(&cli.expansion_add));
    eprintln!("- Expansion search: {}", spell_list(&cli.expansion_search));
    eprintln!("- Startup time limit: {}", spell_duration(cli.startup_time_limit));
    eprintln!("- gRPC port: {}", cli.grpc_port);
    eprintln!("- HTTP port: {}", cli.http_port);
    eprintln!("- Vectors per upsert: {}", cli.vectors_per_upsert);
    let dimensions_sweep = cli.common.dimensions_sweep(state.dimensions());

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap_or_bail("tokio runtime");
    let container = runtime
        .block_on(ContainerHandle::start(
            "qdrant/qdrant:v1.19.1",
            "retrieval-qdrant",
            &[(cli.http_port.into(), 6333), (cli.grpc_port.into(), 6334)],
            &[],
            &[],
        ))
        .unwrap_or_bail("Qdrant container");
    let ready = runtime.block_on(container.wait_for_http(
        &format!("http://localhost:{}/healthz", cli.http_port),
        cli.startup_time_limit,
    ));
    let client = ready.map_err(|error| error.to_string()).and_then(|()| {
        Qdrant::from_url(&format!("http://localhost:{}", cli.grpc_port))
            .build()
            .map_err(|error| error.to_string())
    });

    let mut summary = SweepSummary::default();
    match &client {
        Ok(client) => {
            for (&dimensions, &metric, &data_type, &quantization, &connectivity, &expansion_add, &expansion_search) in iproduct!(
                &dimensions_sweep,
                &cli.metric,
                &cli.data_type,
                &cli.quantization,
                &cli.connectivity,
                &cli.expansion_add,
                &cli.expansion_search
            ) {
                let description = format!(
                    "qdrant · {metric} · data_type={data_type} · quant={quantization} · \
                     M={connectivity} · ef={expansion_add}/{expansion_search} · {dimensions}d"
                );
                let config = CollectionConfig {
                    dimensions,
                    metric,
                    data_type,
                    quantization,
                    connectivity,
                    expansion_add,
                    expansion_search,
                };
                let backend = QdrantBackend::new(client, &container, &runtime, cli.vectors_per_upsert, config);
                summary.record(run_config(
                    &description,
                    IndexFile::None,
                    backend,
                    &mut state,
                    dimensions,
                ));
            }
        }
        Err(error) => eprintln!("Qdrant did not come up: {error}"),
    }
    runtime.block_on(container.stop()).ok();
    summary.finish();
}
