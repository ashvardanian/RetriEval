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
//!     --base-vectors datasets/wiki_1M/base.1M.fbin \
//!     --query-vectors datasets/wiki_1M/query.public.100K.fbin \
//!     --query-neighbors datasets/wiki_1M/groundtruth.public.100K.ibin \
//!     --metric cos --quantization none,binary \
//!     --output results/
//! ```

use std::time::Duration;

use clap::Parser;
use itertools::iproduct;
use retrieval::docker::ContainerHandle;
use retrieval::{
    bail, try_run_config, Backend, BenchState, CommonArgs, Distance, Key, SweepSummary, UnwrapOrBail, Vectors,
};
use serde_json::{json, Value};

const CLASS_NAME: &str = "Bench";

/// CLI metric -> the distance name Weaviate's REST schema expects.
fn parse_weaviate_distance(s: &str) -> Result<&'static str, String> {
    match s {
        "ip" => Ok("dot"),
        "cos" => Ok("cosine"),
        "l2sq" | "l2" => Ok("l2-squared"),
        _ => Err(format!(
            "unknown Weaviate metric: {s} (supported: ip, cos, l2; Hamming needs bit-packed vectors \
             which Weaviate doesn't natively store)"
        )),
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

    /// HNSW connectivity M (comma-separated for sweep)
    #[arg(long, value_delimiter = ',', default_value = "16")]
    connectivity: Vec<usize>,

    /// HNSW expansion factor during indexing (comma-separated for sweep)
    #[arg(long, value_delimiter = ',', default_value = "128")]
    expansion_add: Vec<usize>,

    /// HNSW expansion factor during search (comma-separated for sweep).
    /// Class-level on Weaviate, so each value rebuilds the class.
    #[arg(long, value_delimiter = ',', default_value = "64")]
    expansion_search: Vec<usize>,

    #[arg(long, default_value_t = 120)]
    docker_timeout: u64,

    /// Weaviate HTTP port
    #[arg(long, default_value_t = 8080)]
    port: u16,
}

struct WeaviateBackend {
    http: reqwest::Client,
    http_base: String,
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
        let url = format!("{}/v1/objects", self.http_base);
        self.runtime.block_on(async {
            for (row, &key) in data.chunks_exact(vectors.dimensions).zip(keys) {
                let object = json!({ "class": CLASS_NAME, "properties": { "idx": key as i64 }, "vector": row });
                post_json(&self.http, &url, &object)
                    .await
                    .map_err(|e| format!("Weaviate insert failed: {e}"))?;
            }
            Ok(())
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
        let url = format!("{}/v1/graphql", self.http_base);

        self.runtime.block_on(async {
            for (query_index, query) in data.chunks_exact(queries.dimensions).enumerate() {
                let gql = format!(
                    "{{ Get {{ {CLASS_NAME}(nearVector: {{ vector: {query:?} }} limit: {count}) \
                     {{ idx _additional {{ distance }} }} }} }}"
                );
                let response = post_json(&self.http, &url, &json!({ "query": gql }))
                    .await
                    .map_err(|e| format!("Weaviate query failed: {e}"))?;
                // GraphQL reports a failed query as 200 OK with an `errors` array.
                if let Some(errors) = response.get("errors") {
                    return Err(format!("Weaviate query failed: {errors}"));
                }

                let offset = query_index * count;
                let items = response
                    .pointer(&format!("/data/Get/{CLASS_NAME}"))
                    .and_then(|v| v.as_array());
                let hits = items.into_iter().flatten().filter_map(|item| {
                    let stored_index = item.get("idx").and_then(|v| v.as_i64())?;
                    let key = u32::try_from(stored_index).ok()?;
                    let distance = item
                        .pointer("/_additional/distance")
                        .and_then(|d| d.as_f64())
                        .unwrap_or(f64::INFINITY) as Distance;
                    Some((key as Key, distance))
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

impl Drop for WeaviateBackend {
    fn drop(&mut self) {
        if let Some(c) = self.container.take() {
            let _ = self.runtime.block_on(c.stop());
        }
    }
}

/// POST `body` as JSON and return the parsed response, turning a non-2xx status into an error.
async fn post_json(http: &reqwest::Client, url: &str, body: &Value) -> Result<Value, String> {
    let response = http
        .post(url)
        .json(body)
        .send()
        .await
        .map_err(|e| format!("POST {url} failed: {e}"))?;
    let status = response.status();
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        return Err(format!("POST {url} -> {status}: {text}"));
    }
    response
        .json()
        .await
        .map_err(|e| format!("POST {url} returned invalid JSON: {e}"))
}

/// Drop and recreate the benchmark class with the given HNSW parameters.
async fn create_class(
    http: &reqwest::Client,
    http_base: &str,
    distance: &str,
    quant: WeaviateQuant,
    max_connections: u64,
    ef_construction: u64,
    ef: i64,
) -> Result<(), String> {
    let _ = http.delete(format!("{http_base}/v1/schema/{CLASS_NAME}")).send().await;
    let body = json!({
        "class": CLASS_NAME,
        "description": "Benchmark vectors",
        "vectorizer": "none",
        "vectorIndexType": "hnsw",
        "vectorIndexConfig": {
            "distance": distance,
            "ef": ef,
            "efConstruction": ef_construction,
            "maxConnections": max_connections,
            "bq": { "enabled": matches!(quant, WeaviateQuant::Binary) }
        },
        "properties": [
            { "name": "idx", "dataType": ["int"] }
        ]
    });
    post_json(http, &format!("{http_base}/v1/schema"), &body)
        .await
        .map(drop)
}

/// Reject argument combinations Weaviate will refuse, before a container starts.
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
                     efConstruction below maxConnections leaves the graph under-linked"
                ));
            }
        }
    }
    Ok(())
}

fn main() {
    let cli = Cli::parse();
    validate(&cli).unwrap_or_bail("invalid arguments");

    if cli.common.index.is_some() {
        bail("--index is not supported for this backend");
    }

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
            "semitechnologies/weaviate:1.39.7",
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
    let http = reqwest::Client::new();

    let mut state = BenchState::load(&cli.common).unwrap_or_else(|e| {
        eprintln!("Failed to load benchmark state: {e}");
        std::process::exit(1);
    });
    if cli.common.dimensions.len() > 1 {
        retrieval::bail(
            "--dimensions sweep with >1 value isn't supported on Weaviate; rerun the binary per dimensions",
        );
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
        &cli.quantization,
        &cli.connectivity,
        &cli.expansion_add,
        &cli.expansion_search
    )
    .collect();
    let num_configs = configs.len();
    let mut summary = SweepSummary::default();
    for (idx, (metric_str, quant_str, connectivity, expansion_add, expansion_search)) in configs.into_iter().enumerate()
    {
        let is_last = idx + 1 == num_configs;
        let metric = parse_weaviate_distance(metric_str).expect("validated above");
        let quant = parse_weaviate_quantization(quant_str).expect("validated above");

        runtime.block_on(async {
            create_class(
                &http,
                &http_base,
                metric,
                quant,
                *connectivity as u64,
                *expansion_add as u64,
                *expansion_search as i64,
            )
            .await
            .expect("create class");
        });

        let container_for_this_run = if is_last { container_slot.take() } else { None };

        let description = format!(
            "weaviate · {metric_str} · quant={quant_str} · \
             M={connectivity} · ef={expansion_add}/{expansion_search} · {dimensions}d"
        );
        let backend = WeaviateBackend {
            http: http.clone(),
            http_base: http_base.clone(),
            container: container_for_this_run,
            runtime: runtime.handle().clone(),
            description: description.clone(),
            metadata: {
                let mut metadata = std::collections::HashMap::new();
                metadata.insert("backend".into(), json!("weaviate"));
                metadata.insert("metric".into(), json!(metric_str));
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
