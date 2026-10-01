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

use std::{
    fmt::{self, Write},
    time::Duration,
};

use clap::Parser;
use itertools::iproduct;
use serde_json::{json, Value};

use retrieval::{
    bail, docker::ContainerHandle, spell_duration, spell_list, try_run_config, Backend, BenchState, CommonArgs,
    Distance, Key, Port, SweepSummary, UnwrapOrBail, Vectors,
};

const CLASS_NAME: &str = "Bench";

/// Weaviate distances, as `--metric` spells them; Hamming needs bit-packed vectors Weaviate doesn't store.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum Metric {
    Ip,
    Cos,
    #[value(alias = "l2sq")]
    L2,
}

impl Metric {
    /// The distance name Weaviate's REST schema expects.
    fn schema_name(self) -> &'static str {
        match self {
            Self::Ip => "dot",
            Self::Cos => "cosine",
            Self::L2 => "l2-squared",
        }
    }
}

impl fmt::Display for Metric {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        retrieval::spell_value(self, formatter)
    }
}

/// Server-side quantization, as `--quantization` spells it.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum WeaviateQuant {
    None,
    Binary,
}

impl fmt::Display for WeaviateQuant {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        retrieval::spell_value(self, formatter)
    }
}

#[derive(Parser, Debug)]
#[command(name = "retri-eval-weaviate", about = "Benchmark Weaviate")]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,

    /// Distance metrics (comma-separated for sweep)
    #[arg(long, value_delimiter = ',', value_enum, default_value = "l2")]
    metric: Vec<Metric>,

    /// Server-side quantization (comma-separated for sweep)
    #[arg(long, value_delimiter = ',', value_enum, default_value = "none")]
    quantization: Vec<WeaviateQuant>,

    /// HNSW connectivity M (comma-separated for sweep)
    #[arg(long, value_delimiter = ',', default_value = "16", value_parser = retrieval::parse_count_flag)]
    connectivity: Vec<usize>,

    /// HNSW expansion factor during indexing (comma-separated for sweep)
    #[arg(long, value_delimiter = ',', default_value = "128", value_parser = retrieval::parse_count_flag)]
    expansion_add: Vec<usize>,

    /// HNSW expansion factor during search (comma-separated for sweep).
    /// Class-level on Weaviate, so each value rebuilds the class.
    #[arg(long, value_delimiter = ',', default_value = "64", value_parser = retrieval::parse_count_flag)]
    expansion_search: Vec<usize>,

    /// Time limit for container start and readiness, like 120s
    #[arg(long, default_value = "120s", value_parser = retrieval::parse_duration_flag)]
    startup_time_limit: Duration,

    /// Weaviate HTTP port
    #[arg(long, default_value = "8080", value_parser = retrieval::parse_port)]
    port: Port,
}

struct WeaviateBackend {
    scratch: Vec<f32, std::alloc::System>,
    query: String,
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
        let data = vectors.data.to_f32_in(&mut self.scratch)?;
        let objects: Vec<_> = data
            .chunks_exact(vectors.dimensions)
            .zip(keys)
            .map(|(row, &key)| json!({"class": CLASS_NAME, "properties": {"idx": key}, "vector": row}))
            .collect();
        let response = self.runtime.block_on(post_json(
            &self.http,
            &format!("{}/v1/batch/objects", self.http_base),
            &json!({"objects": objects}),
        ))?;
        let results = response.as_array().ok_or("Weaviate batch response is not an array")?;
        if results.len() != keys.len() {
            return Err("Weaviate batch response length mismatch".into());
        }
        for result in results {
            if result.pointer("/result/status").and_then(Value::as_str) != Some("SUCCESS") {
                return Err(format!("Weaviate batch insert failed: {result}"));
            }
        }
        Ok(())
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
        self.query.clear();
        self.query.push_str("{ Get {");
        for (i, row) in data.chunks_exact(queries.dimensions).enumerate() {
            write!(self.query, "q{i}: {CLASS_NAME}(nearVector: {{ vector: {row:?} }} limit: {count}) {{ idx _additional {{ distance }} }} ").unwrap();
        }
        self.query.push_str("} }");
        let response = self.runtime.block_on(post_json(
            &self.http,
            &format!("{}/v1/graphql", self.http_base),
            &json!({"query": self.query}),
        ))?;
        if let Some(errors) = response.get("errors") {
            return Err(format!("Weaviate query failed: {errors}"));
        }
        for (i, output_count) in out_counts.iter_mut().enumerate() {
            let name = format!("q{i}");
            let items = response
                .get("data")
                .and_then(|v| v.get("Get"))
                .and_then(|v| v.get(&name))
                .and_then(Value::as_array)
                .ok_or("Weaviate query response missing results")?;
            let keys = &mut out_keys[i * count..(i + 1) * count];
            let distances = &mut out_distances[i * count..(i + 1) * count];
            keys.fill(Key::MAX);
            distances.fill(Distance::INFINITY);
            if items.len() > count {
                return Err("Weaviate returned too many results".into());
            }
            for (j, item) in items.iter().enumerate() {
                keys[j] = item
                    .get("idx")
                    .and_then(Value::as_u64)
                    .and_then(|k| Key::try_from(k).ok())
                    .ok_or("Weaviate returned invalid key")?;
                distances[j] = item
                    .pointer("/_additional/distance")
                    .and_then(Value::as_f64)
                    .ok_or("Weaviate returned invalid distance")? as Distance;
                if !distances[j].is_finite() {
                    return Err("Weaviate returned nonfinite distance".into());
                }
            }
            *output_count = items.len();
        }
        Ok(())
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
            "semitechnologies/weaviate:1.39.7",
            "retrieval-weaviate",
            &[(cli.port.into(), 8080), (50051, 50051)],
            &[
                "QUERY_DEFAULTS_LIMIT=25".into(),
                "AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true".into(),
                "PERSISTENCE_DATA_PATH=/var/lib/weaviate".into(),
                "DEFAULT_VECTORIZER_MODULE=none".into(),
                "CLUSTER_HOSTNAME=node1".into(),
            ],
            &[],
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
    eprintln!("- Metrics: {}", spell_list(&cli.metric));
    eprintln!("- Quantization: {}", spell_list(&cli.quantization));
    eprintln!("- Connectivity: {}", spell_list(&cli.connectivity));
    eprintln!("- Expansion add: {}", spell_list(&cli.expansion_add));
    eprintln!("- Expansion search: {}", spell_list(&cli.expansion_search));
    eprintln!("- Startup time limit: {}", spell_duration(cli.startup_time_limit));
    eprintln!("- Port: {}", cli.port);
    if cli.common.dims.len() > 1 {
        retrieval::bail("--dims sweep with >1 value isn't supported on Weaviate; rerun the binary per dimensions");
    }
    let dimensions = cli.common.dims.first().copied().unwrap_or_else(|| state.dimensions());
    state.check_dimensions(dimensions).unwrap_or_bail("invalid --dims");

    let mut container_slot = Some(handle);
    let configs = iproduct!(
        &cli.metric,
        &cli.quantization,
        &cli.connectivity,
        &cli.expansion_add,
        &cli.expansion_search
    );
    let num_configs = configs.clone().count();
    let mut summary = SweepSummary::default();
    for (idx, (&metric, &quant, connectivity, expansion_add, expansion_search)) in configs.enumerate() {
        let is_last = idx + 1 == num_configs;

        runtime.block_on(async {
            create_class(
                &http,
                &http_base,
                metric.schema_name(),
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
            "weaviate · {metric} · quant={quant} · \
             M={connectivity} · ef={expansion_add}/{expansion_search} · {dimensions}d"
        );
        let backend = WeaviateBackend {
            scratch: Vec::new_in(std::alloc::System),
            query: String::new(),
            http: http.clone(),
            http_base: http_base.clone(),
            container: container_for_this_run,
            runtime: runtime.handle().clone(),
            description: description.clone(),
            metadata: {
                let mut metadata = std::collections::HashMap::new();
                metadata.insert("backend".into(), json!("weaviate"));
                metadata.insert("metric".into(), json!(metric.to_string()));
                metadata.insert("quantization".into(), json!(quant.to_string()));
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
