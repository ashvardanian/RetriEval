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
    docker::ContainerHandle, run_config, spell_duration, spell_list, Backend, BatchKeys, BenchState, CommonArgs,
    Distance, Hits, IndexFile, Key, Port, SweepSummary, UnwrapOrBail, Vectors,
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
enum WeaviateQuantization {
    None,
    Binary,
}

impl fmt::Display for WeaviateQuantization {
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
    #[arg(
        long,
        env = "RETRIEVAL_METRIC",
        value_delimiter = ',',
        value_enum,
        default_value = "l2"
    )]
    metric: Vec<Metric>,

    /// Server-side quantization (comma-separated for sweep)
    #[arg(
        long,
        env = "RETRIEVAL_QUANTIZATION",
        value_delimiter = ',',
        value_enum,
        default_value = "none"
    )]
    quantization: Vec<WeaviateQuantization>,

    /// HNSW connectivity M (comma-separated for sweep)
    #[arg(long, env = "RETRIEVAL_CONNECTIVITY", value_delimiter = ',', default_value = "16", value_parser = retrieval::parse_count_flag)]
    connectivity: Vec<usize>,

    /// HNSW expansion factor during indexing (comma-separated for sweep)
    #[arg(long, env = "RETRIEVAL_EXPANSION_ADD", value_delimiter = ',', default_value = "128", value_parser = retrieval::parse_count_flag)]
    expansion_add: Vec<usize>,

    /// HNSW expansion factor during search (comma-separated for sweep).
    /// Class-level on Weaviate, so each value rebuilds the class.
    #[arg(long, env = "RETRIEVAL_EXPANSION_SEARCH", value_delimiter = ',', default_value = "64", value_parser = retrieval::parse_count_flag)]
    expansion_search: Vec<usize>,

    /// Time limit for container start and readiness, like 120s
    #[arg(long, env = "RETRIEVAL_STARTUP_TIME_LIMIT", default_value = "120s", value_parser = retrieval::parse_duration_flag)]
    startup_time_limit: Duration,

    /// Weaviate HTTP port
    #[arg(long, env = "RETRIEVAL_PORT", default_value = "8080", value_parser = retrieval::parse_port)]
    port: Port,
}

/// One config's class on the shared server.
struct WeaviateBackend<'a> {
    scratch: Vec<f32, std::alloc::System>,
    query: String,
    http: &'a reqwest::Client,
    http_base: &'a str,
    container: &'a ContainerHandle,
    runtime: &'a tokio::runtime::Runtime,
    metadata: std::collections::HashMap<String, serde_json::Value>,
}

/// The settings that shape one config's class.
#[derive(Clone, Copy)]
struct ClassSettings {
    metric: Metric,
    quantization: WeaviateQuantization,
    connectivity: usize,
    expansion_add: usize,
    expansion_search: usize,
}

impl<'a> WeaviateBackend<'a> {
    /// Drops and recreates the benchmark class for `settings` on the running server.
    fn new(
        http: &'a reqwest::Client,
        http_base: &'a str,
        container: &'a ContainerHandle,
        runtime: &'a tokio::runtime::Runtime,
        settings: ClassSettings,
    ) -> Result<Self, String> {
        runtime.block_on(create_class(http, http_base, settings))?;
        Ok(Self {
            scratch: Vec::new_in(std::alloc::System),
            query: String::new(),
            http,
            http_base,
            container,
            runtime,
            metadata: std::collections::HashMap::from([
                ("backend".into(), json!("weaviate")),
                ("metric".into(), json!(settings.metric.to_string())),
                ("quantization".into(), json!(settings.quantization.to_string())),
                ("connectivity".into(), json!(settings.connectivity)),
                ("expansion_add".into(), json!(settings.expansion_add)),
                ("expansion_search".into(), json!(settings.expansion_search)),
            ]),
        })
    }
}

impl Backend for WeaviateBackend<'_> {
    fn metadata(&self) -> std::collections::HashMap<String, serde_json::Value> {
        self.metadata.clone()
    }

    fn add(&mut self, keys: BatchKeys<'_>, vectors: Vectors) -> Result<(), String> {
        let dimensions = vectors.dimensions();
        let data = vectors.to_f32_in(&mut self.scratch)?;
        // Every key spells as a UUID, integers zero-padded, so each one is the object's own id.
        self.query.clear();
        self.query.push_str(r#"{"objects":["#);
        for (index, (row, key)) in data.chunks_exact(dimensions).zip(keys.keys()).enumerate() {
            let separator = if index == 0 { "" } else { "," };
            write!(
                self.query,
                r#"{separator}{{"class":"{CLASS_NAME}","id":"{key}","vector":{row:?}}}"#
            )
            .expect("a String takes any write");
        }
        self.query.push_str("]}");
        let response = self.runtime.block_on(post_json(
            self.http,
            &format!("{}/v1/batch/objects", self.http_base),
            self.query.clone(),
        ))?;
        let results = response.as_array().ok_or("Weaviate batch response is not an array")?;
        if results.len() != keys.len() {
            return Err("Weaviate batch response length mismatch".into());
        }
        match results
            .iter()
            .find(|result| result.pointer("/result/status").and_then(Value::as_str) != Some("SUCCESS"))
        {
            Some(failed) => Err(format!("Weaviate batch insert failed: {failed}")),
            None => Ok(()),
        }
    }

    fn search(&mut self, queries: Vectors, mut hits: Hits<'_>) -> Result<(), String> {
        let count = hits.count();
        let dimensions = queries.dimensions();
        let data = queries.to_f32_in(&mut self.scratch)?;
        self.query.clear();
        self.query.push_str("{ Get {");
        for (query_index, row) in data.chunks_exact(dimensions).enumerate() {
            write!(self.query, "q{query_index}: {CLASS_NAME}(nearVector: {{ vector: {row:?} }} limit: {count}) {{ _additional {{ id distance }} }} ").expect("a String takes any write");
        }
        self.query.push_str("} }");
        let response = self.runtime.block_on(post_json(
            self.http,
            &format!("{}/v1/graphql", self.http_base),
            json!({"query": self.query}).to_string(),
        ))?;
        if let Some(errors) = response.get("errors") {
            return Err(format!("Weaviate query failed: {errors}"));
        }
        for (query_index, row) in hits.rows().enumerate() {
            let items = response
                .pointer(&format!("/data/Get/q{query_index}"))
                .and_then(Value::as_array)
                .ok_or("Weaviate query response missing results")?;
            row.write(items.iter().map(|item| {
                let key: Key = item
                    .pointer("/_additional/id")
                    .and_then(Value::as_str)
                    .ok_or("Weaviate returned an object without an id")?
                    .parse()?;
                let distance = item
                    .pointer("/_additional/distance")
                    .and_then(Value::as_f64)
                    .map(|distance| distance as Distance)
                    .filter(|distance| distance.is_finite())
                    .ok_or("Weaviate returned an invalid distance")?;
                Ok((key, distance))
            }))?;
        }
        Ok(())
    }

    fn memory_bytes(&self) -> usize {
        self.runtime.block_on(self.container.memory_usage_bytes()) as usize
    }
}

/// POST `body` as JSON and return the parsed response, turning a non-2xx status into an error.
async fn post_json(http: &reqwest::Client, url: &str, body: String) -> Result<Value, String> {
    let response = http
        .post(url)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .map_err(|error| format!("POST {url} failed: {error}"))?;
    let status = response.status();
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        return Err(format!("POST {url} -> {status}: {text}"));
    }
    response
        .json()
        .await
        .map_err(|error| format!("POST {url} returned invalid JSON: {error}"))
}

/// Drop and recreate the benchmark class with the given HNSW parameters.
async fn create_class(http: &reqwest::Client, http_base: &str, settings: ClassSettings) -> Result<(), String> {
    let _ = http.delete(format!("{http_base}/v1/schema/{CLASS_NAME}")).send().await;
    let body = json!({
        "class": CLASS_NAME,
        "description": "Benchmark vectors",
        "vectorizer": "none",
        "vectorIndexType": "hnsw",
        "vectorIndexConfig": {
            "distance": settings.metric.schema_name(),
            "ef": settings.expansion_search,
            "efConstruction": settings.expansion_add,
            "maxConnections": settings.connectivity,
            "bq": { "enabled": matches!(settings.quantization, WeaviateQuantization::Binary) }
        },
        "properties": []
    });
    post_json(http, &format!("{http_base}/v1/schema"), body.to_string())
        .await
        .map(drop)
}

/// Reject argument combinations Weaviate will refuse, before a container starts.
fn validate(cli: &Cli) -> Result<(), String> {
    let smallest_expansion = cli.expansion_add.iter().min();
    let largest_connectivity = cli.connectivity.iter().max();
    if let (Some(&expansion_add), Some(&connectivity)) = (smallest_expansion, largest_connectivity) {
        if expansion_add < connectivity {
            return Err(format!(
                "--expansion-add {expansion_add} is below --connectivity {connectivity}; \
                 efConstruction below maxConnections leaves the graph under-linked"
            ));
        }
    }
    Ok(())
}

fn main() {
    let cli: Cli = retrieval::parse_cli();
    validate(&cli).unwrap_or_bail("invalid arguments");
    let mut state = BenchState::load(&cli.common).unwrap_or_bail("benchmark state");
    eprintln!("- Metrics: {}", spell_list(&cli.metric));
    eprintln!("- Quantization: {}", spell_list(&cli.quantization));
    eprintln!("- Connectivity: {}", spell_list(&cli.connectivity));
    eprintln!("- Expansion add: {}", spell_list(&cli.expansion_add));
    eprintln!("- Expansion search: {}", spell_list(&cli.expansion_search));
    eprintln!("- Startup time limit: {}", spell_duration(cli.startup_time_limit));
    eprintln!("- Port: {}", cli.port);
    let dimensions_sweep = cli.common.dimensions_sweep(state.dimensions());

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap_or_bail("tokio runtime");
    let container = runtime
        .block_on(ContainerHandle::start(
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
        ))
        .unwrap_or_bail("Weaviate container");
    let http_base = format!("http://localhost:{}", cli.port);
    let http = reqwest::Client::new();
    let ready =
        runtime.block_on(container.wait_for_http(&format!("{http_base}/v1/.well-known/ready"), cli.startup_time_limit));

    let mut summary = SweepSummary::default();
    match ready {
        Ok(()) => {
            for (&dimensions, &metric, &quantization, &connectivity, &expansion_add, &expansion_search) in iproduct!(
                &dimensions_sweep,
                &cli.metric,
                &cli.quantization,
                &cli.connectivity,
                &cli.expansion_add,
                &cli.expansion_search
            ) {
                let description = format!(
                    "weaviate · {metric} · quant={quantization} · \
                     M={connectivity} · ef={expansion_add}/{expansion_search} · {dimensions}d"
                );
                let settings = ClassSettings {
                    metric,
                    quantization,
                    connectivity,
                    expansion_add,
                    expansion_search,
                };
                let backend = WeaviateBackend::new(&http, &http_base, &container, &runtime, settings);
                summary.record(run_config(
                    &description,
                    IndexFile::None,
                    backend,
                    &mut state,
                    dimensions,
                ));
            }
        }
        Err(error) => eprintln!("Weaviate did not come up: {error}"),
    }
    runtime.block_on(container.stop()).ok();
    summary.finish();
}
