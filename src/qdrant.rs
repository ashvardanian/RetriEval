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
//!     --vectors datasets/wiki_1M/base.1M.fbin \
//!     --queries datasets/wiki_1M/query.public.100K.fbin \
//!     --neighbors datasets/wiki_1M/groundtruth.public.100K.ibin \
//!     --metric ip \
//!     --output results/
//! ```

use std::collections::HashMap;
use std::time::Duration;

use clap::Parser;
use itertools::iproduct;
use qdrant_client::qdrant::{
    point_id, quantization_config::Quantization, BinaryQuantization, CreateCollectionBuilder, Datatype,
    Distance as QdrantDistance, HnswConfigDiffBuilder, PointStruct, QuantizationType, ScalarQuantization,
    SearchPointsBuilder, UpsertPointsBuilder, VectorParamsBuilder,
};
use qdrant_client::Qdrant;
use retrieval::docker::ContainerHandle;
use retrieval::{bail, run, Backend, BenchState, CommonArgs, Distance, Key, UnwrapOrBail, Vectors};
use serde_json::json;

const COLLECTION: &str = "bench";

fn parse_qdrant_distance(s: &str) -> Result<QdrantDistance, String> {
    match s {
        "ip" => Ok(QdrantDistance::Dot),
        "cos" => Ok(QdrantDistance::Cosine),
        "l2sq" | "l2" => Ok(QdrantDistance::Euclid),
        "manhattan" => Ok(QdrantDistance::Manhattan),
        _ => Err(format!(
            "unknown Qdrant distance metric: {s} (supported: ip, cos, l2, manhattan)"
        )),
    }
}

/// Qdrant named-vector storage datatype. Server auto-converts from the f32
/// upserts the client sends — no wire-format change needed on our side.
fn parse_qdrant_datatype(s: &str) -> Result<Datatype, String> {
    match s {
        "f32" => Ok(Datatype::Float32),
        "f16" => Ok(Datatype::Float16),
        "u8" => Ok(Datatype::Uint8),
        _ => Err(format!("unknown Qdrant data_type: {s} (supported: f32, f16, u8)")),
    }
}

/// Server-side quantization — `binary` is deterministic `sign(x)` per dimensions;
/// `scalar` is one-pass per-dimensions min/max mapping to int8. Both stay inside the
/// "no learned codebook" constraint. `product` (k-means) is deliberately not
/// offered here.
fn parse_qdrant_quantization(s: &str) -> Result<Option<Quantization>, String> {
    match s {
        "none" => Ok(None),
        "binary" => Ok(Some(Quantization::Binary(BinaryQuantization {
            always_ram: Some(true),
            encoding: None,
            query_encoding: None,
        }))),
        "scalar" => Ok(Some(Quantization::Scalar(ScalarQuantization {
            r#type: QuantizationType::Int8 as i32,
            quantile: Some(0.99),
            always_ram: Some(true),
        }))),
        _ => Err(format!(
            "unknown Qdrant quantization: {s} (supported: none, binary, scalar)"
        )),
    }
}

// #region CLI

#[derive(Parser, Debug)]
#[command(name = "retri-eval-qdrant", about = "Benchmark Qdrant")]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,

    /// Distance metric (comma-separated for sweep): ip, cos, l2, manhattan
    #[arg(long, value_delimiter = ',', default_value = "l2")]
    metric: Vec<String>,

    /// Storage data_type (comma-separated for sweep): f32, f16, u8
    #[arg(long, value_delimiter = ',', default_value = "f32")]
    data_type: Vec<String>,

    /// Quantization (comma-separated for sweep): none, binary, scalar
    #[arg(long, value_delimiter = ',', default_value = "none")]
    quantization: Vec<String>,

    #[arg(long, value_delimiter = ',', default_value_t = 16)]
    connectivity: usize,

    #[arg(long, value_delimiter = ',', default_value_t = 128)]
    expansion_add: usize,

    /// Docker timeout in seconds
    #[arg(long, default_value_t = 120)]
    docker_timeout: u64,

    /// gRPC port for Qdrant
    #[arg(long, default_value_t = 6334)]
    grpc_port: u16,

    /// HTTP port for Qdrant
    #[arg(long, default_value_t = 6333)]
    http_port: u16,

    /// Batch size for upsert operations
    #[arg(long, default_value_t = 10_000)]
    batch_size: usize,
}

// #region Backend

struct QdrantBackend {
    client: Qdrant,
    container: Option<ContainerHandle>,
    runtime: tokio::runtime::Handle,
    batch_size: usize,
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
        let data = vectors.data.to_f32();
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
        &self,
        queries: Vectors,
        count: usize,
        out_keys: &mut [Key],
        out_distances: &mut [Distance],
        out_counts: &mut [usize],
    ) -> Result<(), String> {
        let data = queries.data.to_f32();
        let dimensions = queries.dimensions;
        let num_vectors = data.len() / dimensions;

        self.runtime.block_on(async {
            for query_index in 0..num_vectors {
                let query = data[query_index * dimensions..(query_index + 1) * dimensions].to_vec();
                let response = self
                    .client
                    .search_points(SearchPointsBuilder::new(COLLECTION, query, count as u64))
                    .await
                    .map_err(|e| format!("Qdrant search failed: {e}"))?;

                let offset = query_index * count;
                let mut found = 0;
                for (rank, point) in response.result.iter().enumerate().take(count) {
                    let id = match &point.id {
                        Some(id) => match &id.point_id_options {
                            Some(point_id::PointIdOptions::Num(numeric_id)) => *numeric_id as Key,
                            _ => Key::MAX,
                        },
                        None => Key::MAX,
                    };
                    out_keys[offset + rank] = id;
                    out_distances[offset + rank] = point.score;
                    if id != Key::MAX {
                        found += 1;
                    }
                }
                for rank in response.result.len()..count {
                    out_keys[offset + rank] = Key::MAX;
                    out_distances[offset + rank] = Distance::INFINITY;
                }
                out_counts[query_index] = found;
            }
            Ok::<(), String>(())
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

fn main() {
    let cli = Cli::parse();

    // Validate sweep axes up front so we fail fast on bad CLI input.
    let sweeps: Vec<(String, Datatype, Option<Quantization>, QdrantDistance)> =
        iproduct!(&cli.metric, &cli.data_type, &cli.quantization)
            .map(|(metric, data_type, quantization)| {
                let parsed_metric = parse_qdrant_distance(metric).unwrap_or_bail("metric");
                let parsed_data_type = parse_qdrant_datatype(data_type).unwrap_or_bail("data type");
                let parsed_quantization = parse_qdrant_quantization(quantization).unwrap_or_bail("quantization");
                (quantization.clone(), parsed_data_type, parsed_quantization, parsed_metric)
            })
            .enumerate()
            .map(|(_, v)| v)
            .collect();

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio");
    let timeout = Duration::from_secs(cli.docker_timeout);

    let handle = runtime.block_on(async {
        let handle = ContainerHandle::start(
            "qdrant/qdrant:v1.17.1",
            "retrieval-qdrant",
            &vec![(cli.http_port, 6333), (cli.grpc_port, 6334)],
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
    if cli.common.dimensions.len() > 1 {
        retrieval::bail("--dimensions sweep with >1 value isn't supported on Qdrant; rerun the binary per dimensions");
    }
    let dimensions = cli.common.dimensions.first().copied().unwrap_or_else(|| state.dimensions());
    state.check_dimensions(dimensions).unwrap_or_bail("invalid --dimensions");

    let mut container_slot = Some(handle);
    let num_configs = cli.metric.len() * cli.data_type.len() * cli.quantization.len();
    for (idx, ((metric_str, dtype_str, quant_str), (_raw_q, dtype_enum, quant_opt, metric_enum))) in
        iproduct!(&cli.metric, &cli.data_type, &cli.quantization)
            .zip(sweeps.into_iter())
            .enumerate()
    {
        let is_last = idx + 1 == num_configs;

        runtime.block_on(async {
            let _ = client.delete_collection(COLLECTION).await;
            let mut vector_params = VectorParamsBuilder::new(dimensions as u64, metric_enum);
            vector_params = vector_params.datatype(dtype_enum);
            let mut create = CreateCollectionBuilder::new(COLLECTION)
                .vectors_config(vector_params)
                .hnsw_config(
                    HnswConfigDiffBuilder::default()
                        .m(cli.connectivity as u64)
                        .ef_construct(cli.expansion_add as u64),
                );
            if let Some(q) = quant_opt {
                create = create.quantization_config(q);
            }
            client.create_collection(create).await.expect("create collection");
        });

        // The container handle is held by the backend so it tears down on
        // Drop — for the final config, we give it the real handle; earlier
        // configs stay running (we reuse the same container across sweeps).
        let container_for_this_run = if is_last { container_slot.take() } else { None };

        let description = format!(
            "qdrant · {metric_str} · data_type={dtype_str} · quant={quant_str} · M={} · ef={} · {dimensions}d",
            cli.connectivity, cli.expansion_add
        );

        let mut backend = QdrantBackend {
            client: client.clone(),
            container: container_for_this_run,
            runtime: runtime.handle().clone(),
            batch_size: cli.batch_size,
            description,
            metadata: {
                let mut metadata = std::collections::HashMap::new();
                metadata.insert("backend".into(), json!("qdrant"));
                metadata.insert("metric".into(), json!(metric_str));
                metadata.insert("data_type".into(), json!(dtype_str));
                metadata.insert("quantization".into(), json!(quant_str));
                metadata.insert("dimensions".into(), json!(dimensions));
                metadata.insert("connectivity".into(), json!(cli.connectivity));
                metadata.insert("expansion_add".into(), json!(cli.expansion_add));
                metadata
            },
        };

        run(&mut backend, &mut state, dimensions).unwrap_or_else(|e| {
            eprintln!("Benchmark failed: {e}");
            std::process::exit(1);
        });
    }
    eprintln!("Benchmark complete.");
}
