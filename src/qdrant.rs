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
        Distance as QdrantDistance, HnswConfigDiffBuilder, Memory, PointStruct, QuantizationType, ScalarQuantization,
        SearchParamsBuilder, SearchPointsBuilder, UpsertPointsBuilder, VectorParamsBuilder,
    },
    Qdrant,
};
use serde_json::json;

use retrieval::{
    bail, docker::ContainerHandle, spell_duration, spell_list, try_run_config, Backend, BenchState, CommonArgs,
    Distance, Key, Port, SweepSummary, UnwrapOrBail, Vectors,
};

const COLLECTION: &str = "bench";

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
    #[arg(long, value_delimiter = ',', value_enum, default_value = "l2")]
    metric: Vec<Metric>,

    /// Storage data types (comma-separated for sweep)
    #[arg(long, value_delimiter = ',', value_enum, default_value = "f32")]
    data_type: Vec<DataType>,

    /// Server-side quantization (comma-separated for sweep)
    #[arg(long, value_delimiter = ',', value_enum, default_value = "none")]
    quantization: Vec<QuantizationKind>,

    /// HNSW connectivity M (comma-separated for sweep)
    #[arg(long, value_delimiter = ',', default_value = "16", value_parser = retrieval::parse_count_flag)]
    connectivity: Vec<usize>,

    /// HNSW expansion factor during indexing (comma-separated for sweep)
    #[arg(long, value_delimiter = ',', default_value = "128", value_parser = retrieval::parse_count_flag)]
    expansion_add: Vec<usize>,

    /// HNSW expansion factor during search — Qdrant's `hnsw_ef` search param
    /// (comma-separated for sweep)
    #[arg(long, value_delimiter = ',', default_value = "64", value_parser = retrieval::parse_count_flag)]
    expansion_search: Vec<usize>,

    /// Time limit for container start and readiness, like 120s
    #[arg(long, default_value = "120s", value_parser = retrieval::parse_duration_flag)]
    startup_time_limit: Duration,

    /// gRPC port for Qdrant
    #[arg(long, default_value = "6334", value_parser = retrieval::parse_port)]
    grpc_port: Port,

    /// HTTP port for Qdrant
    #[arg(long, default_value = "6333", value_parser = retrieval::parse_port)]
    http_port: Port,

    /// Vectors per upsert request (distinct from the shared `--vectors-per-add`
    /// / `--queries-per-search`, which pace the harness's add/search loops)
    #[arg(long, default_value_t = 10_000, value_parser = retrieval::parse_count_flag)]
    vectors_per_upsert: usize,
}

// #region Backend

struct QdrantBackend {
    scratch: Vec<f32, std::alloc::System>,
    client: Qdrant,
    container: Option<ContainerHandle>,
    runtime: tokio::runtime::Handle,
    batch_size: usize,
    /// Qdrant's `hnsw_ef`. Set per request rather than on the collection, so it
    /// can be swept without rebuilding the index.
    expansion_search: usize,
    description: String,
    metadata: std::collections::HashMap<String, serde_json::Value>,
}

impl Backend for QdrantBackend {
    fn description(&self) -> String {
        self.description.clone()
    }

    fn metadata(&self) -> std::collections::HashMap<String, serde_json::Value> {
        self.metadata.clone()
    }

    fn add(&mut self, keys: &[Key], vectors: Vectors) -> Result<(), String> {
        let data = vectors.data.to_f32_in(&mut self.scratch)?;
        let dimensions = vectors.dimensions;
        let num_vectors = data.len() / dimensions;

        self.runtime.block_on(async {
            for batch_start in (0..num_vectors).step_by(self.batch_size) {
                let batch_end = (batch_start + self.batch_size).min(num_vectors);
                let points: Vec<PointStruct> = (batch_start..batch_end)
                    .map(|i| {
                        let vec = data[i * dimensions..(i + 1) * dimensions].to_vec();
                        let empty: HashMap<String, qdrant_client::qdrant::Value> = HashMap::new();
                        PointStruct::new(keys[i] as u64, vec, empty)
                    })
                    .collect();
                self.client
                    .upsert_points(UpsertPointsBuilder::new(COLLECTION, points).wait(true))
                    .await
                    .map_err(|e| format!("Qdrant upsert failed: {e}"))?;
            }
            Ok::<(), String>(())
        })
    }

    fn search(
        &mut self,
        queries: Vectors,
        count: usize,
        out_keys: &mut [Key],
        out_distances: &mut [Distance],
        out_counts: &mut [usize],
    ) -> Result<(), String> {
        let data = queries.data.to_f32_in(&mut self.scratch)?;
        let dimensions = queries.dimensions;
        let num_vectors = data.len() / dimensions;

        self.runtime.block_on(async {
            let searches = data
                .chunks_exact(dimensions)
                .map(|row| {
                    SearchPointsBuilder::new(COLLECTION, row.to_vec(), count as u64)
                        .params(SearchParamsBuilder::default().hnsw_ef(self.expansion_search as u64))
                        .build()
                })
                .collect::<Vec<_>>();
            let response = self
                .client
                .search_batch_points(qdrant_client::qdrant::SearchBatchPointsBuilder::new(
                    COLLECTION, searches,
                ))
                .await
                .map_err(|e| format!("Qdrant batch search failed: {e}"))?;
            if response.result.len() != num_vectors {
                return Err("Qdrant batch response count mismatch".into());
            }
            for (query_index, result) in response.result.iter().enumerate() {
                let offset = query_index * count;
                let keys = &mut out_keys[offset..offset + count];
                let distances = &mut out_distances[offset..offset + count];
                keys.fill(Key::MAX);
                distances.fill(Distance::INFINITY);
                if result.result.len() > count {
                    return Err("Qdrant returned too many hits".into());
                }
                for (rank, point) in result.result.iter().enumerate() {
                    let Some(point_id::PointIdOptions::Num(id)) =
                        point.id.as_ref().and_then(|id| id.point_id_options.as_ref())
                    else {
                        return Err("Qdrant returned an invalid numeric ID".into());
                    };
                    keys[rank] = Key::try_from(*id).map_err(|_| "Qdrant key overflow")?;
                    if !point.score.is_finite() {
                        return Err("Qdrant returned a nonfinite score".into());
                    }
                    distances[rank] = point.score;
                }
                out_counts[query_index] = result.result.len();
            }
            Ok(())
        })
    }

    fn memory_bytes(&self) -> usize {
        self.container
            .as_ref()
            .map(|c| self.runtime.block_on(c.memory_usage_bytes()) as usize)
            .unwrap_or(0)
    }
}

impl Drop for QdrantBackend {
    fn drop(&mut self) {
        if let Some(c) = self.container.take() {
            let _ = self.runtime.block_on(c.stop());
        }
    }
}

// #region main

/// Reject argument combinations Qdrant will refuse or silently ignore, before a
/// container is started.
fn validate(cli: &Cli) -> Result<(), String> {
    for &connectivity in &cli.connectivity {
        for &expansion_add in &cli.expansion_add {
            if expansion_add < connectivity {
                return Err(format!(
                    "--expansion-add {expansion_add} is below --connectivity {connectivity}; \
                     ef_construct below M gives Qdrant fewer candidates than edges to keep"
                ));
            }
        }
    }
    // Binary quantization thresholds at zero, so it is meaningless on a storage
    // type that is already unsigned and offset — Qdrant accepts the pair and
    // returns noise.
    for &quantization in &cli.quantization {
        for &data_type in &cli.data_type {
            if quantization == QuantizationKind::Binary && data_type == DataType::U8 {
                return Err("--quantization binary with --data-type u8 quantizes an already-unsigned type".into());
            }
        }
    }
    Ok(())
}

fn main() {
    let cli: Cli = retrieval::parse_cli();
    validate(&cli).unwrap_or_bail("invalid arguments");

    if cli.common.index.is_some() {
        bail("--index is not supported for this backend");
    }

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio");
    let timeout = cli.startup_time_limit;

    let handle = runtime.block_on(async {
        let handle = ContainerHandle::start(
            "qdrant/qdrant:v1.19.1",
            "retrieval-qdrant",
            &[(cli.http_port.into(), 6333), (cli.grpc_port.into(), 6334)],
            &[],
            &[],
            timeout,
        )
        .await
        .expect("docker start");
        handle
            .wait_for_http(&format!("http://localhost:{}/healthz", cli.http_port), timeout)
            .await
            .expect("health");
        handle
    });

    let client = Qdrant::from_url(&format!("http://localhost:{}", cli.grpc_port))
        .build()
        .expect("qdrant client");

    let mut state = BenchState::load(&cli.common).unwrap_or_else(|e| {
        eprintln!("Failed to load benchmark state: {e}");
        std::process::exit(1);
    });
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
    if cli.common.dims.len() > 1 {
        retrieval::bail("--dims sweep with >1 value isn't supported on Qdrant; rerun the binary per dimensions");
    }
    let dimensions = cli.common.dims.first().copied().unwrap_or_else(|| state.dimensions());
    state.check_dimensions(dimensions).unwrap_or_bail("invalid --dims");

    let mut container_slot = Some(handle);
    let configs = iproduct!(
        &cli.metric,
        &cli.data_type,
        &cli.quantization,
        &cli.connectivity,
        &cli.expansion_add,
        &cli.expansion_search
    );
    let num_configs = configs.clone().count();
    let mut summary = SweepSummary::default();
    for (idx, (&metric, &data_type, &quantization, connectivity, expansion_add, expansion_search)) in
        configs.enumerate()
    {
        let is_last = idx + 1 == num_configs;

        runtime.block_on(async {
            let _ = client.delete_collection(COLLECTION).await;
            let mut vector_params = VectorParamsBuilder::new(dimensions as u64, QdrantDistance::from(metric));
            vector_params = vector_params.datatype(Datatype::from(data_type));
            let mut create = CreateCollectionBuilder::new(COLLECTION)
                .vectors_config(vector_params)
                .hnsw_config(
                    HnswConfigDiffBuilder::default()
                        .m(*connectivity as u64)
                        .ef_construct(*expansion_add as u64),
                );
            if let Some(q) = quantization.config() {
                create = create.quantization_config(q);
            }
            client.create_collection(create).await.expect("create collection");
        });

        // The container handle is held by the backend so it tears down on
        // Drop — for the final config, we give it the real handle; earlier
        // configs stay running (we reuse the same container across sweeps).
        let container_for_this_run = if is_last { container_slot.take() } else { None };

        let description = format!(
            "qdrant · {metric} · data_type={data_type} · quant={quantization} · \
             M={connectivity} · ef={expansion_add}/{expansion_search} · {dimensions}d"
        );

        let backend = QdrantBackend {
            scratch: Vec::new_in(std::alloc::System),
            client: client.clone(),
            container: container_for_this_run,
            runtime: runtime.handle().clone(),
            batch_size: cli.vectors_per_upsert,
            expansion_search: *expansion_search,
            description: description.clone(),
            metadata: {
                let mut metadata = std::collections::HashMap::new();
                metadata.insert("backend".into(), json!("qdrant"));
                metadata.insert("metric".into(), json!(metric.to_string()));
                metadata.insert("data_type".into(), json!(data_type.to_string()));
                metadata.insert("quantization".into(), json!(quantization.to_string()));
                metadata.insert("connectivity".into(), json!(connectivity));
                metadata.insert("expansion_add".into(), json!(expansion_add));
                metadata.insert("expansion_search".into(), json!(expansion_search));
                metadata
            },
        };

        summary.record(try_run_config(
            &description,
            Ok::<_, String>(backend),
            &mut state,
            dimensions,
        ));
    }
    summary.print();
}
