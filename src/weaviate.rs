//! Weaviate benchmark binary.
//!
//! ## Prerequisites
//!
//! Requires Docker — the benchmark auto-manages a `semitechnologies/weaviate` container.
//!
//! ## Build & Install
//!
//! ```sh
//! cargo install --path . --features weaviate-backend
//! ```
//!
//! ## Examples
//!
//! ```sh
//! retri-eval-weaviate \
//!     --vectors datasets/wiki_1M/base.1M.fbin \
//!     --queries datasets/wiki_1M/query.public.100K.fbin \
//!     --neighbors datasets/wiki_1M/groundtruth.public.100K.ibin \
//!     --metric cos --quantization none,binary \
//!     --output results/
//! ```

use std::time::Duration;

use clap::Parser;
use itertools::iproduct;
use retrieval::docker::ContainerHandle;
use retrieval::{bail, run, Backend, BenchState, CommonArgs, Distance, Key, UnwrapOrBail, Vectors};
use serde_json::json;
use weaviate_community::collections::objects::Object;
use weaviate_community::collections::query::RawQuery;
use weaviate_community::collections::schema::*;
use weaviate_community::WeaviateClient;

const CLASS_NAME: &str = "Bench";

fn parse_weaviate_distance(s: &str) -> Result<DistanceMetric, String> {
    match s {
        "ip" => Ok(DistanceMetric::DOT),
        "cos" => Ok(DistanceMetric::COSINE),
        "l2sq" | "l2" => Ok(DistanceMetric::L2SQUARED),
        _ => Err(format!(
            "unknown Weaviate metric: {s} (supported: ip, cos, l2; Hamming needs bit-packed vectors \
             which Weaviate doesn't natively store)"
        )),
    }
}

/// Weaviate distance metric -> the string Weaviate's REST schema expects.
/// Used when we bypass the typed client and hand-craft the JSON body for BQ.
fn distance_as_json_str(m: DistanceMetric) -> &'static str {
    match m {
        DistanceMetric::DOT => "dot",
        DistanceMetric::COSINE => "cosine",
        DistanceMetric::L2SQUARED => "l2-squared",
        DistanceMetric::HAMMING => "hamming",
        DistanceMetric::MANHATTAN => "manhattan",
    }
}

fn parse_weaviate_quantization(s: &str) -> Result<WeaviateQuant, String> {
    match s {
        "none" => Ok(WeaviateQuant::None),
        "binary" => Ok(WeaviateQuant::Binary),
        _ => Err(format!(
            "unknown Weaviate quantization: {s} (supported: none, binary; sq/pq not in this pass)"
        )),
    }
}

#[derive(Clone, Copy)]
enum WeaviateQuant {
    None,
    Binary,
}

#[derive(Parser, Debug)]
#[command(name = "retri-eval-weaviate", about = "Benchmark Weaviate")]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,

    /// Distance metric (comma-separated for sweep): ip, cos, l2
    #[arg(long, value_delimiter = ',', default_value = "l2")]
    metric: Vec<String>,

    /// Server-side quantization (comma-separated for sweep): none, binary
    #[arg(long, value_delimiter = ',', default_value = "none")]
    quantization: Vec<String>,

    #[arg(long, default_value_t = 16)]
    connectivity: usize,

    #[arg(long, default_value_t = 128)]
    expansion_add: usize,

    #[arg(long, default_value_t = 64)]
    expansion_search: usize,

    #[arg(long, default_value_t = 120)]
    docker_timeout: u64,

    /// Weaviate HTTP port
    #[arg(long, default_value_t = 8080)]
    port: u16,
}

struct WeaviateBackend {
    client: WeaviateClient,
    container: Option<ContainerHandle>,
    runtime: tokio::runtime::Handle,
    description: String,
    metadata: std::collections::HashMap<String, serde_json::Value>,
}

impl Backend for WeaviateBackend {
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
            for vector_index in 0..num_vectors {
                // Weaviate community crate requires Vec<f64> by ownership — allocation unavoidable
                let row: Vec<f64> = data[vector_index * dimensions..(vector_index + 1) * dimensions]
                    .iter()
                    .map(|&value| value as f64)
                    .collect();
                let obj = Object::builder(CLASS_NAME, serde_json::json!({ "idx": keys[vector_index] as i64 }))
                    .with_vector(row)
                    .build();
                self.client
                    .objects
                    .create(&obj, None)
                    .await
                    .map_err(|e| format!("Weaviate insert failed: {e}"))?;
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
                let query: Vec<f64> = data[query_index * dimensions..(query_index + 1) * dimensions]
                    .iter()
                    .map(|&value| value as f64)
                    .collect();
                let gql = format!(
                    "{{ Get {{ {CLASS_NAME}(nearVector: {{ vector: {query:?} }} limit: {count}) \
                     {{ idx _additional {{ distance }} }} }} }}"
                );
                let response = self
                    .client
                    .query
                    .raw(RawQuery::new(&gql))
                    .await
                    .map_err(|e| format!("Weaviate query failed: {e}"))?;

                let offset = query_index * count;
                let mut found = 0;
                if let Some(items) = response
                    .pointer(&format!("/data/Get/{CLASS_NAME}"))
                    .and_then(|v| v.as_array())
                {
                    for (rank, item) in items.iter().enumerate().take(count) {
                        let stored_index = item.get("idx").and_then(|v| v.as_i64()).unwrap_or(-1);
                        let distance_value = item
                            .pointer("/_additional/distance")
                            .and_then(|d| d.as_f64())
                            .unwrap_or(f64::INFINITY) as f32;
                        out_keys[offset + rank] = if stored_index >= 0 { stored_index as Key } else { Key::MAX };
                        out_distances[offset + rank] = distance_value;
                        if stored_index >= 0 {
                            found += 1;
                        }
                    }
                }
                for rank in found..count {
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

impl Drop for WeaviateBackend {
    fn drop(&mut self) {
        if let Some(c) = self.container.take() {
            let _ = self.runtime.block_on(c.stop());
        }
    }
}

/// Create the Weaviate class. When `quant == Binary` we bypass the typed
/// `VectorIndexConfig::builder` path — the `weaviate-community` 0.2 crate has
/// no `.with_bq(...)` method — and POST the schema directly via `reqwest`.
/// When `quant == None` we use the existing typed builder so the rest of the
/// crate's schema validation keeps running.
async fn create_class(
    client: &WeaviateClient,
    http_base: &str,
    metric: DistanceMetric,
    quant: WeaviateQuant,
    max_connections: u64,
    ef_construction: u64,
    ef: i64,
) -> Result<(), String> {
    let _ = client.schema.delete(CLASS_NAME).await;
    match quant {
        WeaviateQuant::None => {
            let class = Class::builder(CLASS_NAME)
                .with_description("Benchmark vectors")
                .with_vectorizer("none")
                .with_vector_index_type(VectorIndexType::HNSW)
                .with_vector_index_config(
                    VectorIndexConfig::builder()
                        .with_distance(metric)
                        .with_ef(ef)
                        .with_ef_construction(ef_construction)
                        .with_max_connections(max_connections)
                        .build(),
                )
                .with_properties(Properties::new(vec![Property::builder("idx", vec!["int"]).build()]))
                .build();
            client
                .schema
                .create_class(&class)
                .await
                .map_err(|e| format!("create class failed: {e}"))?;
        }
        WeaviateQuant::Binary => {
            let body = json!({
                "class": CLASS_NAME,
                "description": "Benchmark vectors",
                "vectorizer": "none",
                "vectorIndexType": "hnsw",
                "vectorIndexConfig": {
                    "distance": distance_as_json_str(metric),
                    "ef": ef,
                    "efConstruction": ef_construction,
                    "maxConnections": max_connections,
                    "bq": { "enabled": true }
                },
                "properties": [
                    { "name": "idx", "dataType": ["int"] }
                ]
            });
            let resp = reqwest::Client::new()
                .post(format!("{http_base}/v1/schema"))
                .json(&body)
                .send()
                .await
                .map_err(|e| format!("POST /v1/schema failed: {e}"))?;
            if !resp.status().is_success() {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                return Err(format!("POST /v1/schema -> {status}: {text}"));
            }
        }
    }
    Ok(())
}

fn main() {
    let cli = Cli::parse();

    for m in &cli.metric {
        parse_weaviate_distance(m).unwrap_or_bail("metric");
    }
    for quantization in &cli.quantization {
        parse_weaviate_quantization(quantization).unwrap_or_bail("quantization");
    }

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio");
    let timeout = Duration::from_secs(cli.docker_timeout);

    let handle = runtime.block_on(async {
        let handle = ContainerHandle::start(
            "semitechnologies/weaviate:1.36.10",
            "retrieval-weaviate",
            &vec![(cli.port, 8080), (50051, 50051)],
            &[
                "QUERY_DEFAULTS_LIMIT=25".into(),
                "AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true".into(),
                "PERSISTENCE_DATA_PATH=/var/lib/weaviate".into(),
                "DEFAULT_VECTORIZER_MODULE=none".into(),
                "CLUSTER_HOSTNAME=node1".into(),
            ],
            timeout,
        )
        .await
        .expect("docker start");
        handle
            .wait_for_http(&format!("http://localhost:{}/v1/.well-known/ready", cli.port), timeout)
            .await
            .expect("weaviate not ready");
        handle
    });

    let http_base = format!("http://localhost:{}", cli.port);
    let client = WeaviateClient::new(&http_base, None, None).expect("weaviate client");

    let mut state = BenchState::load(&cli.common).unwrap_or_else(|e| {
        eprintln!("Failed to load benchmark state: {e}");
        std::process::exit(1);
    });
    if cli.common.dimensions.len() > 1 {
        retrieval::bail("--dimensions sweep with >1 value isn't supported on Weaviate; rerun the binary per dimensions");
    }
    let dimensions = cli.common.dimensions.first().copied().unwrap_or_else(|| state.dimensions());
    state.check_dimensions(dimensions).unwrap_or_bail("invalid --dimensions");

    let mut container_slot = Some(handle);
    let num_configs = cli.metric.len() * cli.quantization.len();
    for (idx, (metric_str, quant_str)) in iproduct!(&cli.metric, &cli.quantization).enumerate() {
        let is_last = idx + 1 == num_configs;
        let metric = parse_weaviate_distance(metric_str).expect("validated above");
        let quant = parse_weaviate_quantization(quant_str).expect("validated above");

        runtime.block_on(async {
            create_class(
                &client,
                &http_base,
                metric,
                quant,
                cli.connectivity as u64,
                cli.expansion_add as u64,
                cli.expansion_search as i64,
            )
            .await
            .expect("create class");
        });

        let container_for_this_run = if is_last { container_slot.take() } else { None };

        let mut backend = WeaviateBackend {
            client: WeaviateClient::new(&http_base, None, None).expect("weaviate client"),
            container: container_for_this_run,
            runtime: runtime.handle().clone(),
            description: format!(
                "weaviate · {metric_str} · quant={quant_str} · M={} · ef={}/{} · {dimensions}d",
                cli.connectivity, cli.expansion_add, cli.expansion_search,
            ),
            metadata: {
                let mut metadata = std::collections::HashMap::new();
                metadata.insert("backend".into(), json!("weaviate"));
                metadata.insert("metric".into(), json!(metric_str));
                metadata.insert("quantization".into(), json!(quant_str));
                metadata.insert("connectivity".into(), json!(cli.connectivity));
                metadata.insert("expansion_add".into(), json!(cli.expansion_add));
                metadata.insert("expansion_search".into(), json!(cli.expansion_search));
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
