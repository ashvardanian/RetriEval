//! Redis/RediSearch benchmark binary.
//!
//! ## Prerequisites
//!
//! Requires Docker — the benchmark auto-manages a `redis/redis-stack` container.
//!
//! ## Build & Install
//!
//! ```sh
//! cargo install --path . --features redis-backend
//! ```
//!
//! ## Examples
//!
//! ```sh
//! retri-eval-redis \
//!     --base-vectors datasets/wiki_1M/base.1M.fbin \
//!     --query-vectors datasets/wiki_1M/query.public.100K.fbin \
//!     --query-neighbors datasets/wiki_1M/groundtruth.public.100K.ibin \
//!     --metric ip \
//!     --output results/
//! ```

use std::{fmt, fmt::Write, time::Duration};

use clap::Parser;
use itertools::iproduct;
use serde_json::json;

use retrieval::{
    docker::ContainerHandle, run_config, spell_duration, spell_list, Backend, BenchState, CastScratch, CommonArgs,
    Distance, Hits, IndexFile, Key, Port, SweepSummary, UnwrapOrBail, Vectors,
};

const INDEX_NAME: &str = "bench_idx";
const PREFIX: &str = "vec:";

/// RediSearch distance metrics, as `--metric` spells them; Hamming and Jaccard aren't exposed on vector indexes.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum Metric {
    Ip,
    Cos,
    #[value(alias = "l2sq")]
    L2,
}

impl Metric {
    /// The `FT.CREATE ... DISTANCE_METRIC` token.
    fn token(self) -> &'static str {
        match self {
            Self::Ip => "IP",
            Self::Cos => "COSINE",
            Self::L2 => "L2",
        }
    }
}

impl fmt::Display for Metric {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        retrieval::spell_value(self, formatter)
    }
}

/// RediSearch vector types, as `--data-type` spells them.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum DataType {
    F32,
    F64,
    F16,
    Bf16,
    U8,
    I8,
}

impl From<DataType> for CastScratch {
    fn from(data_type: DataType) -> Self {
        match data_type {
            DataType::F64 => Self::F64(Vec::new_in(std::alloc::System)),
            DataType::F32 => Self::F32(Vec::new_in(std::alloc::System)),
            DataType::Bf16 => Self::BF16(Vec::new_in(std::alloc::System)),
            DataType::F16 => Self::F16(Vec::new_in(std::alloc::System)),
            DataType::I8 => Self::I8(Vec::new_in(std::alloc::System)),
            DataType::U8 => Self::U8(Vec::new_in(std::alloc::System)),
        }
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        retrieval::spell_value(self, formatter)
    }
}

/// RediSearch `FT.CREATE VECTOR ... TYPE <name>` strings and the bytes each element takes on the wire.
#[derive(Clone, Copy)]
struct RedisDataType {
    token: &'static str,
    bytes_per_element: usize,
}

impl From<DataType> for RedisDataType {
    fn from(data_type: DataType) -> Self {
        let (token, bytes_per_element) = match data_type {
            DataType::F32 => ("FLOAT32", 4),
            DataType::F64 => ("FLOAT64", 8),
            DataType::F16 => ("FLOAT16", 2),
            DataType::Bf16 => ("BFLOAT16", 2),
            DataType::U8 => ("UINT8", 1),
            DataType::I8 => ("INT8", 1),
        };
        Self {
            token,
            bytes_per_element,
        }
    }
}

// #region CLI

#[derive(Parser, Debug)]
#[command(name = "retri-eval-redis", about = "Benchmark Redis/RediSearch")]
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

    /// FT.CREATE VECTOR TYPE (comma-separated for sweep)
    #[arg(
        long,
        env = "RETRIEVAL_DATA_TYPE",
        value_delimiter = ',',
        value_enum,
        default_value = "f32"
    )]
    data_type: Vec<DataType>,

    /// HNSW connectivity M (comma-separated for sweep)
    #[arg(long, env = "RETRIEVAL_CONNECTIVITY", value_delimiter = ',', default_value = "16", value_parser = retrieval::parse_count_flag)]
    connectivity: Vec<usize>,

    /// HNSW expansion factor during indexing (comma-separated for sweep)
    #[arg(long, env = "RETRIEVAL_EXPANSION_ADD", value_delimiter = ',', default_value = "128", value_parser = retrieval::parse_count_flag)]
    expansion_add: Vec<usize>,

    /// HNSW expansion factor during search — RediSearch's `EF_RUNTIME`
    /// (comma-separated for sweep). RediSearch rejects values below the
    /// requested neighbor count, so this is raised to it when smaller.
    #[arg(long, env = "RETRIEVAL_EXPANSION_SEARCH", value_delimiter = ',', default_value = "64", value_parser = retrieval::parse_count_flag)]
    expansion_search: Vec<usize>,

    /// Time limit for container start and readiness, like 120s
    #[arg(long, env = "RETRIEVAL_STARTUP_TIME_LIMIT", default_value = "120s", value_parser = retrieval::parse_duration_flag)]
    startup_time_limit: Duration,

    /// Redis port
    #[arg(long, env = "RETRIEVAL_PORT", default_value = "6379", value_parser = retrieval::parse_port)]
    port: Port,

    /// Vectors per pipeline flush (distinct from the shared `--inserts-per-core`
    /// / `--queries-per-core`, which pace the harness's add/search loops)
    #[arg(long, env = "RETRIEVAL_VECTORS_PER_UPSERT", default_value_t = 1_000, value_parser = retrieval::parse_count_flag)]
    vectors_per_upsert: usize,
}

// #region Backend

/// One config's index on the shared server.
struct RedisBackend<'a> {
    connection: redis::Connection,
    wire: CastScratch,
    /// One key at a time, spelled into the same buffer before the pipeline copies it.
    key: String,
    pipeline: redis::Pipeline,
    container: &'a ContainerHandle,
    runtime: &'a tokio::runtime::Runtime,
    vectors_per_upsert: usize,
    data_type: RedisDataType,
    /// RediSearch's `EF_RUNTIME`, applied per query rather than at index
    /// creation so it can be swept without rebuilding.
    expansion_search: usize,
    metadata: std::collections::HashMap<String, serde_json::Value>,
}

/// The settings that shape one config's index.
#[derive(Clone, Copy)]
struct IndexSettings {
    dimensions: usize,
    metric: Metric,
    data_type: DataType,
    connectivity: usize,
    expansion_add: usize,
    expansion_search: usize,
}

impl<'a> RedisBackend<'a> {
    /// Empties the server and creates the benchmark index for `settings`.
    fn new(
        client: &redis::Client,
        container: &'a ContainerHandle,
        runtime: &'a tokio::runtime::Runtime,
        vectors_per_upsert: usize,
        settings: IndexSettings,
    ) -> Result<Self, String> {
        let data_type = RedisDataType::from(settings.data_type);
        let mut connection = client
            .get_connection()
            .map_err(|error| format!("Redis connection: {error}"))?;
        let () = redis::cmd("FLUSHALL")
            .query(&mut connection)
            .map_err(|error| format!("Redis FLUSHALL: {error}"))?;
        let () = redis::cmd("FT.CREATE")
            .arg(INDEX_NAME)
            .arg("ON")
            .arg("HASH")
            .arg("PREFIX")
            .arg("1")
            .arg(PREFIX)
            .arg("SCHEMA")
            .arg("vector")
            .arg("VECTOR")
            .arg("HNSW")
            .arg("10")
            .arg("TYPE")
            .arg(data_type.token)
            .arg("DIM")
            .arg(settings.dimensions)
            .arg("DISTANCE_METRIC")
            .arg(settings.metric.token())
            .arg("M")
            .arg(settings.connectivity)
            .arg("EF_CONSTRUCTION")
            .arg(settings.expansion_add)
            .query(&mut connection)
            .map_err(|error| format!("Redis FT.CREATE: {error}"))?;
        Ok(Self {
            connection,
            wire: settings.data_type.into(),
            key: String::new(),
            pipeline: redis::pipe(),
            container,
            runtime,
            vectors_per_upsert,
            data_type,
            expansion_search: settings.expansion_search,
            metadata: std::collections::HashMap::from([
                ("backend".into(), json!("redis")),
                ("metric".into(), json!(settings.metric.to_string())),
                ("data_type".into(), json!(settings.data_type.to_string())),
                ("bytes_per_element".into(), json!(data_type.bytes_per_element)),
                ("connectivity".into(), json!(settings.connectivity)),
                ("expansion_add".into(), json!(settings.expansion_add)),
                ("expansion_search".into(), json!(settings.expansion_search)),
            ]),
        })
    }
}

impl Backend for RedisBackend<'_> {
    fn metadata(&self) -> std::collections::HashMap<String, serde_json::Value> {
        self.metadata.clone()
    }

    fn add(&mut self, keys: &[Key], vectors: Vectors) -> Result<(), String> {
        let bytes_per_row = vectors.dimensions() * self.data_type.bytes_per_element;
        let encoded = self.wire.encode(&vectors)?;
        let batches = keys
            .chunks(self.vectors_per_upsert)
            .zip(encoded.chunks(self.vectors_per_upsert * bytes_per_row));
        for (keys, rows) in batches {
            self.pipeline.clear();
            for (&key, row) in keys.iter().zip(rows.chunks_exact(bytes_per_row)) {
                self.key.clear();
                write!(self.key, "{PREFIX}{key}").expect("a String takes any write");
                self.pipeline.cmd("HSET").arg(&self.key).arg("vector").arg(row).ignore();
            }
            let () = self
                .pipeline
                .query(&mut self.connection)
                .map_err(|error| format!("Redis HSET failed: {error}"))?;
        }
        Ok(())
    }

    fn search(&mut self, queries: Vectors, mut hits: Hits<'_>) -> Result<(), String> {
        let count = hits.count();
        let bytes_per_row = queries.dimensions() * self.data_type.bytes_per_element;
        // `EF_RUNTIME` below the requested count is rejected by RediSearch.
        let expansion_runtime = self.expansion_search.max(count);
        let query = format!("*=>[KNN {count} @vector $BLOB EF_RUNTIME {expansion_runtime}]");
        let encoded = self.wire.encode(&queries)?;
        self.pipeline.clear();
        for query_bytes in encoded.chunks_exact(bytes_per_row) {
            self.pipeline
                .cmd("FT.SEARCH")
                .arg(INDEX_NAME)
                .arg(&query)
                .arg("PARAMS")
                .arg(2)
                .arg("BLOB")
                .arg(query_bytes)
                .arg("RETURN")
                .arg(1)
                .arg("__vector_score")
                .arg("DIALECT")
                .arg(2)
                .arg("SORTBY")
                .arg("__vector_score")
                .arg("ASC")
                .arg("LIMIT")
                .arg(0)
                .arg(count);
        }
        let responses: Vec<redis::Value> = self
            .pipeline
            .query(&mut self.connection)
            .map_err(|error| format!("FT.SEARCH batch failed: {error}"))?;
        if responses.len() != queries.len() {
            return Err("Redis batch response count mismatch".into());
        }
        for (response, row) in responses.iter().zip(hits.rows()) {
            row.write(decode_ft_search(response)?)?;
        }
        Ok(())
    }

    fn memory_bytes(&self) -> usize {
        self.runtime.block_on(self.container.memory_usage_bytes()) as usize
    }
}

fn redis_text(value: &redis::Value) -> Option<&str> {
    match value {
        redis::Value::BulkString(v) => std::str::from_utf8(v).ok(),
        redis::Value::SimpleString(v) => Some(v),
        _ => None,
    }
}
/// The hits of one `FT.SEARCH` reply, best first.
fn decode_ft_search(
    value: &redis::Value,
) -> Result<impl Iterator<Item = Result<(Key, Distance), String>> + '_, String> {
    let redis::Value::Array(items) = value else {
        return Err("invalid FT.SEARCH response".into());
    };
    if items.is_empty() || !(items.len() - 1).is_multiple_of(2) {
        return Err("malformed FT.SEARCH response".into());
    }
    Ok(items[1..].as_chunks::<2>().0.iter().map(|[key, fields]| {
        let key = redis_text(key)
            .and_then(|text| text.strip_prefix(PREFIX))
            .and_then(|text| text.parse::<Key>().ok())
            .ok_or("invalid Redis result key")?;
        let redis::Value::Array(fields) = fields else {
            return Err("invalid Redis result fields".into());
        };
        let distance = fields
            .as_chunks::<2>()
            .0
            .iter()
            .find(|[name, _]| redis_text(name) == Some("__vector_score"))
            .and_then(|[_, value]| redis_text(value))
            .and_then(|text| text.parse::<Distance>().ok())
            .filter(|distance| distance.is_finite())
            .ok_or("invalid Redis result distance")?;
        Ok((key, distance))
    }))
}

// #region main

/// Reject argument combinations RediSearch will refuse or silently reinterpret,
/// before a container is started.
fn validate(cli: &Cli) -> Result<(), String> {
    let smallest_expansion = cli.expansion_add.iter().min();
    let largest_connectivity = cli.connectivity.iter().max();
    if let (Some(&expansion_add), Some(&connectivity)) = (smallest_expansion, largest_connectivity) {
        if expansion_add < connectivity {
            return Err(format!(
                "--expansion-add {expansion_add} is below --connectivity {connectivity}; \
                 RediSearch cannot build a graph with fewer candidates than edges"
            ));
        }
    }
    Ok(())
}

/// Polls PING until Redis answers or `timeout` passes.
fn wait_for_ping(client: &redis::Client, timeout: Duration) -> Result<(), String> {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        let ready = client
            .get_connection_with_timeout(Duration::from_millis(500))
            .and_then(|mut connection| redis::cmd("PING").query::<String>(&mut connection))
            .is_ok_and(|reply| reply == "PONG");
        if ready {
            return Ok(());
        }
        if std::time::Instant::now() >= deadline {
            return Err("Redis did not answer PING before the startup deadline".into());
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn main() {
    let cli: Cli = retrieval::parse_cli();
    validate(&cli).unwrap_or_bail("invalid arguments");
    let mut state = BenchState::load(&cli.common).unwrap_or_bail("benchmark state");
    eprintln!("- Metrics: {}", spell_list(&cli.metric));
    eprintln!("- Data types: {}", spell_list(&cli.data_type));
    eprintln!("- Connectivity: {}", spell_list(&cli.connectivity));
    eprintln!("- Expansion add: {}", spell_list(&cli.expansion_add));
    eprintln!("- Expansion search: {}", spell_list(&cli.expansion_search));
    eprintln!("- Startup time limit: {}", spell_duration(cli.startup_time_limit));
    eprintln!("- Port: {}", cli.port);
    eprintln!("- Vectors per upsert: {}", cli.vectors_per_upsert);
    let dimensions_sweep = cli.common.dimensions_sweep(state.dimensions());

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap_or_bail("tokio runtime");
    let container = runtime
        .block_on(ContainerHandle::start(
            "redis:8.10",
            "retrieval-redis",
            &[(cli.port.into(), 6379)],
            &[],
            &[],
        ))
        .unwrap_or_bail("Redis container");
    let client = redis::Client::open(format!("redis://localhost:{}/", cli.port))
        .map_err(|error| error.to_string())
        .and_then(|client| wait_for_ping(&client, cli.startup_time_limit).map(|()| client));

    let mut summary = SweepSummary::default();
    match &client {
        Ok(client) => {
            for (&dimensions, &metric, &data_type, &connectivity, &expansion_add, &expansion_search) in iproduct!(
                &dimensions_sweep,
                &cli.metric,
                &cli.data_type,
                &cli.connectivity,
                &cli.expansion_add,
                &cli.expansion_search
            ) {
                let description = format!(
                    "redis · {metric} · data_type={data_type} · \
                     M={connectivity} · ef={expansion_add}/{expansion_search} · {dimensions}d"
                );
                let settings = IndexSettings {
                    dimensions,
                    metric,
                    data_type,
                    connectivity,
                    expansion_add,
                    expansion_search,
                };
                let backend = RedisBackend::new(client, &container, &runtime, cli.vectors_per_upsert, settings);
                summary.record(run_config(
                    &description,
                    IndexFile::None,
                    backend,
                    &mut state,
                    dimensions,
                ));
            }
        }
        Err(error) => eprintln!("Redis did not come up: {error}"),
    }
    runtime.block_on(container.stop()).ok();
    summary.finish();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_borrows_scores_and_rejects_missing_fields() {
        let text = |s: &str| redis::Value::BulkString(s.as_bytes().to_vec());
        let response = redis::Value::Array(vec![
            redis::Value::Int(1),
            text(&format!("{PREFIX}42")),
            redis::Value::Array(vec![text("__vector_score"), text("0.25")]),
        ]);
        let hits: Result<Vec<_>, _> = decode_ft_search(&response).unwrap().collect();
        assert_eq!(hits.unwrap(), [(42, 0.25)]);
        let malformed = redis::Value::Array(vec![
            redis::Value::Int(1),
            text(&format!("{PREFIX}42")),
            redis::Value::Array(vec![]),
        ]);
        assert!(decode_ft_search(&malformed).unwrap().any(|hit| hit.is_err()));
    }
}
