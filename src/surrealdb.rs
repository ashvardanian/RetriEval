//! SurrealDB native F32 HNSW benchmark binary.
//!
//! ## Prerequisites
//!
//! Requires Docker; manages a pinned `surrealdb/surrealdb:v3.3.0` container.
//!
//! ## Build & Install
//!
//! ```sh
//! cargo install --path . --no-default-features --features surrealdb-backend
//! ```
//!
//! ## Examples
//!
//! ```sh
//! retri-eval-surrealdb --base-vectors data/base.fbin \
//!     --query-vectors data/query.fbin --query-neighbors data/neighbors.ibin \
//!     --metric l2,cos --connectivity 16 --expansion-add 128 --expansion-search 64
//! ```

use std::{
    alloc::System,
    collections::HashMap,
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

const IMAGE: &str = "surrealdb/surrealdb:v3.3.0";
const INDEX: &str = "bench_hnsw";

/// SurrealDB HNSW distances, as `--metric` spells them.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum Metric {
    L2,
    Cos,
}

impl Metric {
    /// The `DEFINE INDEX ... HNSW DIST` token.
    fn token(self) -> &'static str {
        match self {
            Self::L2 => "EUCLIDEAN",
            Self::Cos => "COSINE",
        }
    }
}

impl fmt::Display for Metric {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        retrieval::spell_value(self, formatter)
    }
}

#[derive(Parser, Debug)]
#[command(name = "retri-eval-surrealdb", about = "Benchmark SurrealDB native F32 HNSW")]
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

    /// HNSW connectivity M, at least 2 (comma-separated for sweep)
    #[arg(long, env = "RETRIEVAL_CONNECTIVITY", value_delimiter = ',', default_value = "16", value_parser = retrieval::parse_count_flag)]
    connectivity: Vec<usize>,

    /// HNSW expansion factor during indexing, SurrealDB's `EFC` (comma-separated for sweep)
    #[arg(long, env = "RETRIEVAL_EXPANSION_ADD", value_delimiter = ',', default_value = "128", value_parser = retrieval::parse_count_flag)]
    expansion_add: Vec<usize>,

    /// HNSW expansion factor during search, the `EF` of each KNN query (comma-separated for sweep)
    #[arg(long, env = "RETRIEVAL_EXPANSION_SEARCH", value_delimiter = ',', default_value = "64", value_parser = retrieval::parse_count_flag)]
    expansion_search: Vec<usize>,

    /// Time limit for container start and readiness, and for each HTTP request, like 120s
    #[arg(long, env = "RETRIEVAL_STARTUP_TIME_LIMIT", default_value = "120s", value_parser = retrieval::parse_duration_flag)]
    startup_time_limit: Duration,

    /// SurrealDB HTTP port
    #[arg(long, env = "RETRIEVAL_PORT", default_value = "8000", value_parser = retrieval::parse_port)]
    port: Port,
}

fn sql_body_capacity(dimensions: usize, rows: usize) -> Result<usize, String> {
    dimensions
        .checked_mul(20)
        .and_then(|bytes| bytes.checked_add(256))
        .and_then(|bytes| bytes.checked_mul(rows))
        .and_then(|bytes| bytes.checked_add(1024))
        .map(|bytes| bytes.max(1 << 20))
        .ok_or_else(|| "SurrealDB request exceeds address space".into())
}

struct SurrealServer {
    container: ContainerHandle,
    runtime: tokio::runtime::Handle,
    http: reqwest::Client,
    sql_url: String,
    token: String,
    sql_body_limit: usize,
}

impl SurrealServer {
    async fn start(
        runtime: tokio::runtime::Handle,
        port: Port,
        timeout: Duration,
        sql_body_limit: usize,
    ) -> Result<Self, String> {
        let container = ContainerHandle::start(
            IMAGE,
            "retrieval-surrealdb",
            &[(port.into(), 8000)],
            &[format!("SURREAL_HTTP_MAX_SQL_BODY_SIZE={sql_body_limit}")],
            &[
                "start".into(),
                "--log".into(),
                "warn".into(),
                "--user".into(),
                "root".into(),
                "--pass".into(),
                "root".into(),
                "memory".into(),
            ],
        )
        .await
        .map_err(|error| error.to_string())?;
        let base = format!("http://127.0.0.1:{port}");
        let setup = async {
            container
                .wait_for_http(&format!("{base}/health"), timeout)
                .await
                .map_err(|error| error.to_string())?;
            let http = reqwest::Client::builder()
                .timeout(timeout)
                .build()
                .map_err(|error| error.to_string())?;
            let response: Value = http
                .post(format!("{base}/signin"))
                .header("Accept", "application/json")
                .json(&json!({"user":"root","pass":"root"}))
                .send()
                .await
                .map_err(|error| error.to_string())?
                .error_for_status()
                .map_err(|error| error.to_string())?
                .json()
                .await
                .map_err(|error| error.to_string())?;
            let token = response
                .get("token")
                .and_then(Value::as_str)
                .ok_or("SurrealDB sign-in returned no token")?
                .to_owned();
            Ok::<_, String>((http, token))
        }
        .await;
        match setup {
            Ok((http, token)) => Ok(Self {
                container,
                runtime,
                http,
                sql_url: format!("{base}/sql"),
                token,
                sql_body_limit,
            }),
            Err(error) => {
                let _ = container.stop().await;
                Err(error)
            }
        }
    }
    async fn query(&self, sql: &str) -> Result<Value, String> {
        let response = self
            .http
            .post(&self.sql_url)
            .bearer_auth(&self.token)
            .header("Accept", "application/json")
            .header("surreal-ns", "retrieval")
            .header("surreal-db", "benchmark")
            .body(sql.to_owned())
            .send()
            .await
            .map_err(|error| error.to_string())?;
        let status = response.status();
        if !status.is_success() {
            return Err(format!(
                "SurrealDB HTTP {status}: {}",
                response.text().await.unwrap_or_default()
            ));
        }
        let body: Value = response.json().await.map_err(|error| error.to_string())?;
        let statements = body.as_array().ok_or("SurrealDB SQL response is not an array")?;
        for statement in statements {
            if statement.get("status").and_then(Value::as_str) != Some("OK") {
                return Err(format!("SurrealDB statement failed: {statement}"));
            }
        }
        Ok(body)
    }
}
impl Drop for SurrealServer {
    fn drop(&mut self) {
        let _ = self.runtime.block_on(self.container.stop());
    }
}

fn uses_hnsw(plan: &Value) -> bool {
    match plan {
        Value::Object(fields) => {
            fields.get("operator").and_then(Value::as_str) == Some("KnnScan")
                && fields
                    .get("attributes")
                    .and_then(|attributes| attributes.get("index"))
                    .and_then(Value::as_str)
                    == Some(INDEX)
                || fields.values().any(uses_hnsw)
        }
        Value::Array(values) => values.iter().any(uses_hnsw),
        _ => false,
    }
}
fn append_search(sql: &mut String, row: &[f32], count: usize, expansion: usize) {
    write!(sql, "SELECT <string> record::id(id) AS key, vector::distance::knn() AS distance FROM bench WHERE embedding <|{count},{expansion}|> {row:?} ORDER BY distance")
        .expect("a String takes any write");
}

struct SurrealBackend<'a> {
    server: &'a SurrealServer,
    scratch: Vec<f32, System>,
    sql: String,
    expansion_search: usize,
    metadata: HashMap<String, Value>,
}
impl<'a> SurrealBackend<'a> {
    fn new(
        server: &'a SurrealServer,
        dimensions: usize,
        metric: Metric,
        connectivity: usize,
        expansion_add: usize,
        expansion_search: usize,
        batch_capacity: usize,
    ) -> Result<Self, String> {
        let distance = metric.token();
        let schema = format!("DEFINE NAMESPACE IF NOT EXISTS retrieval; DEFINE DATABASE IF NOT EXISTS benchmark; REMOVE TABLE IF EXISTS bench; DEFINE TABLE bench SCHEMALESS; DEFINE INDEX {INDEX} ON bench FIELDS embedding HNSW DIMENSION {dimensions} TYPE F32 DIST {distance} EFC {expansion_add} M {connectivity};");
        server.runtime.block_on(server.query(&schema))?;
        let mut probe = Vec::with_capacity_in(dimensions, System);
        probe.resize(dimensions, 1.0);
        let mut query = String::new();
        append_search(&mut query, &probe, 1, expansion_search);
        query.push_str(" EXPLAIN;");
        let plan = server.runtime.block_on(server.query(&query))?;
        if !uses_hnsw(&plan) {
            return Err(format!("SurrealDB did not select native HNSW: {plan}"));
        }
        let elements = dimensions
            .checked_mul(batch_capacity)
            .ok_or("SurrealDB batch dimensions exceed address space")?;
        let mut scratch = Vec::new_in(System);
        scratch.try_reserve(elements).map_err(|error| error.to_string())?;
        let bytes = sql_body_capacity(dimensions, batch_capacity)?;
        query.try_reserve(bytes).map_err(|error| error.to_string())?;
        Ok(Self {
            server,
            scratch,
            sql: query,
            expansion_search,
            metadata: HashMap::from([
                ("backend".into(), json!("surrealdb")),
                ("image".into(), json!(IMAGE)),
                ("metric".into(), json!(metric.to_string())),
                ("data_type".into(), json!("f32")),
                ("index".into(), json!("HNSW")),
                ("connectivity".into(), json!(connectivity)),
                ("expansion_add".into(), json!(expansion_add)),
                ("expansion_search".into(), json!(expansion_search)),
                ("sql_body_limit_bytes".into(), json!(server.sql_body_limit)),
                ("index_plan_verified".into(), json!(true)),
                ("storage".into(), json!("memory")),
                ("batch_add".into(), json!("native INSERT array")),
                (
                    "batch_search".into(),
                    json!("SQL statements pipelined in one HTTP request"),
                ),
            ]),
        })
    }
}
impl Backend for SurrealBackend<'_> {
    fn metadata(&self) -> HashMap<String, Value> {
        self.metadata.clone()
    }

    fn add(&mut self, keys: BatchKeys<'_>, vectors: Vectors) -> Result<(), String> {
        let dimensions = vectors.dimensions();
        let values = vectors.to_f32_in(&mut self.scratch)?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err("SurrealDB requires finite vectors".into());
        }
        self.sql.clear();
        self.sql.push_str("INSERT INTO bench [");
        // Every key spells as a UUID, integers zero-padded, so each one is a native record id.
        for (index, (row, key)) in values.chunks_exact(dimensions).zip(keys.keys()).enumerate() {
            let separator = if index == 0 { "" } else { "," };
            write!(self.sql, "{separator}{{id:bench:u\"{key}\",embedding:{row:?}}}").expect("a String takes any write");
        }
        self.sql.push_str("] RETURN VALUE id;");
        let response = self.server.runtime.block_on(self.server.query(&self.sql))?;
        let inserted = response
            .pointer("/0/result")
            .and_then(Value::as_array)
            .ok_or("missing SurrealDB insert results")?;
        if inserted.len() == keys.len() {
            Ok(())
        } else {
            Err("SurrealDB insert count mismatch".into())
        }
    }

    fn search(&mut self, queries: Vectors, mut hits: Hits<'_>) -> Result<(), String> {
        let count = hits.count();
        let dimensions = queries.dimensions();
        let values = queries.to_f32_in(&mut self.scratch)?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err("SurrealDB requires finite queries".into());
        }
        self.sql.clear();
        for row in values.chunks_exact(dimensions) {
            append_search(&mut self.sql, row, count, self.expansion_search);
            self.sql.push(';');
        }
        let response = self.server.runtime.block_on(self.server.query(&self.sql))?;
        let statements = response.as_array().ok_or("missing SurrealDB query results")?;
        if statements.len() != queries.len() {
            return Err("SurrealDB query result count mismatch".into());
        }
        for (statement, row) in statements.iter().zip(hits.rows()) {
            let neighbors = statement
                .get("result")
                .and_then(Value::as_array)
                .ok_or("invalid SurrealDB neighbors")?;
            row.write(neighbors.iter().enumerate().map(|(rank, neighbor)| {
                let key: Key = neighbor
                    .get("key")
                    .and_then(Value::as_str)
                    .ok_or("invalid SurrealDB neighbor key")?
                    .parse()?;
                let distance = neighbor
                    .get("distance")
                    .and_then(Value::as_f64)
                    .map(|distance| distance as Distance)
                    .filter(|distance| distance.is_finite())
                    .ok_or("invalid SurrealDB neighbor distance")?;
                if neighbors[..rank]
                    .iter()
                    .any(|earlier| earlier.get("key") == neighbor.get("key"))
                {
                    return Err("duplicate SurrealDB neighbor".into());
                }
                Ok((key, distance))
            }))?;
        }
        Ok(())
    }

    fn memory_bytes(&self) -> usize {
        self.server.runtime.block_on(self.server.container.memory_usage_bytes()) as usize
    }
}

/// Reject argument combinations SurrealDB will refuse, before a container is started.
fn validate(cli: &Cli) -> Result<(), String> {
    if let Some(&connectivity) = cli.connectivity.iter().find(|&&connectivity| connectivity < 2) {
        return Err(format!(
            "--connectivity {connectivity} is below the 2 SurrealDB's HNSW needs"
        ));
    }
    let smallest_expansion = cli.expansion_add.iter().min();
    let largest_connectivity = cli.connectivity.iter().max();
    if let (Some(&expansion_add), Some(&connectivity)) = (smallest_expansion, largest_connectivity) {
        if expansion_add < connectivity {
            return Err(format!(
                "--expansion-add {expansion_add} is below --connectivity {connectivity}; \
                 SurrealDB cannot build a graph with fewer candidates than edges"
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
    eprintln!("- Connectivity: {}", spell_list(&cli.connectivity));
    eprintln!("- Expansion add: {}", spell_list(&cli.expansion_add));
    eprintln!("- Expansion search: {}", spell_list(&cli.expansion_search));
    eprintln!("- Startup time limit: {}", spell_duration(cli.startup_time_limit));
    eprintln!("- Port: {}", cli.port);
    let dimensions_sweep = cli.common.dimensions_sweep(state.dimensions());
    let rows_per_request = cli.common.inserts_per_core.max(cli.common.queries_per_core);
    let widest = dimensions_sweep
        .iter()
        .copied()
        .max()
        .expect("the sweep holds at least the native width");
    let sql_body_limit = sql_body_capacity(widest, rows_per_request).unwrap_or_bail("SQL body limit");
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap_or_bail("tokio runtime");
    let server = runtime
        .block_on(SurrealServer::start(
            runtime.handle().clone(),
            cli.port,
            cli.startup_time_limit,
            sql_body_limit,
        ))
        .unwrap_or_bail("SurrealDB start");
    let mut summary = SweepSummary::default();
    for (&dimensions, &metric, &connectivity, &expansion_add, &expansion_search) in iproduct!(
        &dimensions_sweep,
        &cli.metric,
        &cli.connectivity,
        &cli.expansion_add,
        &cli.expansion_search
    ) {
        let description = format!(
            "surrealdb · {metric} · F32 HNSW · M={connectivity} · ef={expansion_add}/{expansion_search} · {dimensions}d"
        );
        let backend = SurrealBackend::new(
            &server,
            dimensions,
            metric,
            connectivity,
            expansion_add,
            expansion_search,
            rows_per_request,
        );
        summary.record(run_config(
            &description,
            IndexFile::None,
            backend,
            &mut state,
            dimensions,
        ));
    }
    drop(server);
    summary.finish();
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{sql_body_capacity, uses_hnsw};

    #[test]
    fn sql_body_limit_covers_default_batches_and_checks_overflow() {
        assert_eq!(sql_body_capacity(8, 16).unwrap(), 1 << 20);
        assert_eq!(sql_body_capacity(1536, 10_000).unwrap(), 309_761_024);
        assert!(sql_body_capacity(usize::MAX, 1).is_err());
        assert!(sql_body_capacity(1, usize::MAX).is_err());
    }
    #[test]
    fn accepts_only_the_named_hnsw_plan() {
        assert!(uses_hnsw(
            &json!({"children":[{"operator":"KnnScan","attributes":{"index":"bench_hnsw"}}]})
        ));
        assert!(!uses_hnsw(
            &json!({"operator":"TableScan","attributes":{"index":"bench_hnsw"}})
        ));
        assert!(!uses_hnsw(
            &json!({"operator":"KnnScan","attributes":{"index":"other"}})
        ));
    }
}
