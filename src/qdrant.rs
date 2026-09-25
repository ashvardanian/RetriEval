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

use std::collections::HashMap;
use std::time::Duration;

use clap::Parser;
use itertools::iproduct;
use qdrant_client::qdrant::{
    point_id, quantization_config::Quantization, BinaryQuantization, CreateCollectionBuilder, Datatype,
    Distance as QdrantDistance, HnswConfigDiffBuilder, Memory, PointStruct, QuantizationType, ScalarQuantization,
    SearchParamsBuilder, SearchPointsBuilder, UpsertPointsBuilder, VectorParamsBuilder,
};
use qdrant_client::Qdrant;
use retrieval::docker::ContainerHandle;
use retrieval::{
    bail, try_run_config, Backend, BenchState, CommonArgs, Distance, Key, SweepSummary, UnwrapOrBail, Vectors,
};
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
            memory: Some(Memory::Pinned as i32),
            ..Default::default()
        }))),
        "scalar" => Ok(Some(Quantization::Scalar(ScalarQuantization {
            r#type: QuantizationType::Int8 as i32,
            quantile: Some(0.99),
            memory: Some(Memory::Pinned as i32),
            ..Default::default()
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

    /// HNSW connectivity M (comma-separated for sweep)
    #[arg(long, value_delimiter = ',', default_value = "16")]
    connectivity: Vec<usize>,

    /// HNSW expansion factor during indexing (comma-separated for sweep)
    #[arg(long, value_delimiter = ',', default_value = "128")]
    expansion_add: Vec<usize>,

    /// HNSW expansion factor during search — Qdrant's `hnsw_ef` search param
    /// (comma-separated for sweep)
    #[arg(long, value_delimiter = ',', default_value = "64")]
    expansion_search: Vec<usize>,

    /// Docker timeout in seconds
    #[arg(long, default_value_t = 120)]
    docker_timeout: u64,

    /// gRPC port for Qdrant
    #[arg(long, default_value_t = 6334)]
    grpc_port: u16,

    /// HTTP port for Qdrant
    #[arg(long, default_value_t = 6333)]
    http_port: u16,

    /// Vectors per upsert request (distinct from the shared `--batch-size-add`
    /// / `--batch-size-search`, which pace the harness's add/search loops)
    #[arg(long, default_value_t = 10_000)]
    batch_size_upsert: usize,
}

// #region Backend

struct QdrantBackend {
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
                    .search_points(
                        SearchPointsBuilder::new(COLLECTION, query, count as u64)
                            .params(SearchParamsBuilder::default().hnsw_ef(self.expansion_search as u64)),
                    )
                    .await
                    .map_err(|e| format!("Qdrant search failed: {e}"))?;

                let offset = query_index * count;
                let hits = response.result.iter().filter_map(|point| {
                    let numeric_id = match point.id.as_ref()?.point_id_options.as_ref()? {
                        point_id::PointIdOptions::Num(numeric_id) => *numeric_id,
                        _ => return None,
                    };
                    Some((numeric_id as Key, point.score))
                });
                out_counts[query_index] = retrieval::write_row(
                    hits,
                    &mut out_keys[offset..offset + count],
                    &mut out_distances[offset..offset + count],
                );
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

/// Reject argument combinations Qdrant will refuse or silently ignore, before a
/// container is started.
fn validate(cli: &Cli) -> Result<(), String> {
    for &expansion_search in &cli.expansion_search {
        if expansion_search == 0 {
            return Err("--expansion-search must be greater than 0".into());
        }
    }
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
    for quantization in &cli.quantization {
        for data_type in &cli.data_type {
            if quantization == "binary" && data_type == "u8" {
                return Err("--quantization binary with --data-type u8 quantizes an already-unsigned type".into());
            }
        }
    }
    if cli.batch_size_upsert == 0 {
        return Err("--batch-size-upsert must be greater than 0".into());
    }
    Ok(())
}

fn main() {
    let cli = Cli::parse();
    validate(&cli).unwrap_or_bail("invalid arguments");

    if cli.common.index.is_some() {
        bail("--index is not supported for this backend");
    }

    // Reject bad CLI input before starting a container. The loop below re-parses
    // rather than carrying a parallel vector — one `iproduct!` zipped against a
    // second is only correct while both enumerate identically, which silently
    // stops holding the moment a sweep axis is added to one of them.
    for (metric, data_type, quantization) in iproduct!(&cli.metric, &cli.data_type, &cli.quantization) {
        parse_qdrant_distance(metric).unwrap_or_bail("metric");
        parse_qdrant_datatype(data_type).unwrap_or_bail("data type");
        parse_qdrant_quantization(quantization).unwrap_or_bail("quantization");
    }

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio");
    let timeout = Duration::from_secs(cli.docker_timeout);

    let handle = runtime.block_on(async {
        let handle = ContainerHandle::start(
            "qdrant/qdrant:v1.19.1",
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
    let dimensions = cli
        .common
        .dimensions
        .first()
        .copied()
        .unwrap_or_else(|| state.dimensions());
    state
        .check_dimensions(dimensions)
        .unwrap_or_bail("invalid --dimensions");

    let mut container_slot = Some(handle);
    let configs: Vec<_> = iproduct!(
        &cli.metric,
        &cli.data_type,
        &cli.quantization,
        &cli.connectivity,
        &cli.expansion_add,
        &cli.expansion_search
    )
    .collect();
    let num_configs = configs.len();
    let mut summary = SweepSummary::default();
    for (idx, (metric_str, dtype_str, quant_str, connectivity, expansion_add, expansion_search)) in
        configs.into_iter().enumerate()
    {
        let is_last = idx + 1 == num_configs;
        let metric_enum = parse_qdrant_distance(metric_str).unwrap_or_bail("metric");
        let dtype_enum = parse_qdrant_datatype(dtype_str).unwrap_or_bail("data type");
        let quant_opt = parse_qdrant_quantization(quant_str).unwrap_or_bail("quantization");

        runtime.block_on(async {
            let _ = client.delete_collection(COLLECTION).await;
            let mut vector_params = VectorParamsBuilder::new(dimensions as u64, metric_enum);
            vector_params = vector_params.datatype(dtype_enum);
            let mut create = CreateCollectionBuilder::new(COLLECTION)
                .vectors_config(vector_params)
                .hnsw_config(
                    HnswConfigDiffBuilder::default()
                        .m(*connectivity as u64)
                        .ef_construct(*expansion_add as u64),
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
            "qdrant · {metric_str} · data_type={dtype_str} · quant={quant_str} · \
             M={connectivity} · ef={expansion_add}/{expansion_search} · {dimensions}d"
        );

        let backend = QdrantBackend {
            client: client.clone(),
            container: container_for_this_run,
            runtime: runtime.handle().clone(),
            batch_size: cli.batch_size_upsert,
            expansion_search: *expansion_search,
            description: description.clone(),
            metadata: {
                let mut metadata = std::collections::HashMap::new();
                metadata.insert("backend".into(), json!("qdrant"));
                metadata.insert("metric".into(), json!(metric_str));
                metadata.insert("data_type".into(), json!(dtype_str));
                metadata.insert("quantization".into(), json!(quant_str));
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
