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

use std::{fmt, time::Duration};

use clap::Parser;
use itertools::iproduct;
use serde_json::json;

use retrieval::{
    bail, docker::ContainerHandle, pod_slice_as_bytes, spell_duration, spell_list, try_run_config, Backend, BenchState,
    CommonArgs, Distance, Key, Port, SweepSummary, UnwrapOrBail, VectorSlice, Vectors,
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

impl fmt::Display for DataType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        retrieval::spell_value(self, formatter)
    }
}

/// RediSearch `FT.CREATE VECTOR ... TYPE <name>` strings.
/// Redis 8+ adds the half-float and narrow-int dtypes; older redis-stack
/// images only accept the four float formats. Bytes-per-element is fixed by
/// the type and used to size the RESP payload.
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

enum WireScratch {
    F32(Vec<f32, std::alloc::System>),
    F64(Vec<f64, std::alloc::System>),
    F16(Vec<numkong::f16, std::alloc::System>),
    BF16(Vec<numkong::bf16, std::alloc::System>),
    U8(Vec<u8, std::alloc::System>),
    I8(Vec<i8, std::alloc::System>),
}
impl WireScratch {
    fn new(data_type: RedisDataType) -> Self {
        match data_type.token {
            "FLOAT32" => Self::F32(Vec::new_in(std::alloc::System)),
            "FLOAT64" => Self::F64(Vec::new_in(std::alloc::System)),
            "FLOAT16" => Self::F16(Vec::new_in(std::alloc::System)),
            "BFLOAT16" => Self::BF16(Vec::new_in(std::alloc::System)),
            "UINT8" => Self::U8(Vec::new_in(std::alloc::System)),
            "INT8" => Self::I8(Vec::new_in(std::alloc::System)),
            _ => unreachable!(),
        }
    }
    fn encode<'a>(&'a mut self, source: &'a VectorSlice<'_>, start: usize, end: usize) -> &'a [u8] {
        match (&mut *self, source) {
            (Self::F32(_), VectorSlice::F32(data)) => return unsafe { pod_slice_as_bytes(&data[start..end]) },
            (Self::I8(_), VectorSlice::I8(data)) => return unsafe { pod_slice_as_bytes(&data[start..end]) },
            (Self::U8(_), VectorSlice::U8(data)) => return &data[start..end],
            _ => {}
        }
        match self {
            Self::F32(v) => cast_into(source, start, end, v),
            Self::F64(v) => cast_into(source, start, end, v),
            Self::F16(v) => cast_into(source, start, end, v),
            Self::BF16(v) => cast_into(source, start, end, v),
            Self::U8(v) => cast_into(source, start, end, v),
            Self::I8(v) => cast_into(source, start, end, v),
        }
    }
}
fn cast_into<'a, T: numkong::CastDtype + Copy + Default>(
    source: &VectorSlice<'_>,
    start: usize,
    end: usize,
    target: &'a mut Vec<T, std::alloc::System>,
) -> &'a [u8] {
    target.resize(end - start, T::default());
    let result = match source {
        VectorSlice::F32(v) => numkong::cast(&v[start..end], target),
        VectorSlice::I8(v) => numkong::cast(&v[start..end], target),
        VectorSlice::U8(v) => numkong::cast(&v[start..end], target),
        VectorSlice::B1x8(_) => unreachable!(),
    };
    result.expect("matching conversion lengths");
    unsafe { pod_slice_as_bytes(target) }
}

// #region CLI

#[derive(Parser, Debug)]
#[command(name = "retri-eval-redis", about = "Benchmark Redis/RediSearch")]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,

    /// Distance metrics (comma-separated for sweep)
    #[arg(long, value_delimiter = ',', value_enum, default_value = "l2")]
    metric: Vec<Metric>,

    /// FT.CREATE VECTOR TYPE (comma-separated for sweep); u8 and i8 need Redis 8+
    #[arg(long, value_delimiter = ',', value_enum, default_value = "f32")]
    data_type: Vec<DataType>,

    /// HNSW connectivity M (comma-separated for sweep)
    #[arg(long, value_delimiter = ',', default_value = "16", value_parser = retrieval::parse_count_flag)]
    connectivity: Vec<usize>,

    /// HNSW expansion factor during indexing (comma-separated for sweep)
    #[arg(long, value_delimiter = ',', default_value = "128", value_parser = retrieval::parse_count_flag)]
    expansion_add: Vec<usize>,

    /// HNSW expansion factor during search — RediSearch's `EF_RUNTIME`
    /// (comma-separated for sweep). RediSearch rejects values below the
    /// requested neighbor count, so this is raised to it when smaller.
    #[arg(long, value_delimiter = ',', default_value = "64", value_parser = retrieval::parse_count_flag)]
    expansion_search: Vec<usize>,

    /// Time limit for container start and readiness, like 120s
    #[arg(long, default_value = "120s", value_parser = retrieval::parse_duration_flag)]
    startup_time_limit: Duration,

    /// Redis port
    #[arg(long, default_value = "6379", value_parser = retrieval::parse_port)]
    port: Port,

    /// Vectors per pipeline flush (distinct from the shared `--vectors-per-add`
    /// / `--queries-per-search`, which pace the harness's add/search loops)
    #[arg(long, default_value_t = 1_000, value_parser = retrieval::parse_count_flag)]
    vectors_per_upsert: usize,
}

// #region Backend

struct RedisBackend {
    connection: redis::Connection,
    wire: WireScratch,
    pipeline: redis::Pipeline,
    container: Option<ContainerHandle>,
    runtime: tokio::runtime::Handle,
    batch_size: usize,
    data_type: RedisDataType,
    /// RediSearch's `EF_RUNTIME`, applied per query rather than at index
    /// creation so it can be swept without rebuilding.
    expansion_search: usize,
    description: String,
    metadata: std::collections::HashMap<String, serde_json::Value>,
}

impl Backend for RedisBackend {
    fn description(&self) -> String {
        self.description.clone()
    }

    fn metadata(&self) -> std::collections::HashMap<String, serde_json::Value> {
        self.metadata.clone()
    }

    fn add(&mut self, keys: &[Key], vectors: Vectors) -> Result<(), String> {
        if matches!(vectors.data, retrieval::VectorSlice::B1x8(_)) {
            return Err("This backend does not support packed binary input".into());
        }
        let dimensions = vectors.dimensions;
        let num_vectors = vectors.len();
        let bytes_per_row = dimensions * self.data_type.bytes_per_element;

        for batch_start in (0..num_vectors).step_by(self.batch_size) {
            let batch_end = (batch_start + self.batch_size).min(num_vectors);
            let batch_rows = batch_end - batch_start;
            let encoded = self
                .wire
                .encode(&vectors.data, batch_start * dimensions, batch_end * dimensions);

            self.pipeline.clear();
            let pipe = &mut self.pipeline;
            for row_within_batch in 0..batch_rows {
                let global_index = batch_start + row_within_batch;
                let key = format!("{PREFIX}{}", keys[global_index]);
                pipe.cmd("HSET")
                    .arg(&key)
                    .arg("vector")
                    .arg(&encoded[row_within_batch * bytes_per_row..(row_within_batch + 1) * bytes_per_row])
                    .ignore();
            }
            let _: () = pipe
                .query(&mut self.connection)
                .map_err(|e| format!("Redis HSET failed: {e}"))?;
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
        if matches!(queries.data, retrieval::VectorSlice::B1x8(_)) {
            return Err("This backend does not support packed binary input".into());
        }
        let dimensions = queries.dimensions;
        let num_vectors = queries.len();
        let bytes_per_row = dimensions * self.data_type.bytes_per_element;
        // `EF_RUNTIME` below the requested count is rejected by RediSearch.
        let ef_runtime = self.expansion_search.max(count);
        let query_str = format!("*=>[KNN {count} @vector $BLOB EF_RUNTIME {ef_runtime}]");

        let encoded = self.wire.encode(&queries.data, 0, num_vectors * dimensions);
        self.pipeline.clear();

        for query_index in 0..num_vectors {
            let query_bytes = &encoded[query_index * bytes_per_row..(query_index + 1) * bytes_per_row];
            self.pipeline
                .cmd("FT.SEARCH")
                .arg(INDEX_NAME)
                .arg(&query_str)
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
            .map_err(|e| format!("FT.SEARCH batch failed: {e}"))?;
        if responses.len() != num_vectors {
            return Err("Redis batch response count mismatch".into());
        }
        for (query_index, response) in responses.iter().enumerate() {
            let offset = query_index * count;
            out_counts[query_index] = decode_ft_search(
                response,
                &mut out_keys[offset..offset + count],
                &mut out_distances[offset..offset + count],
            )?;
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

impl Drop for RedisBackend {
    fn drop(&mut self) {
        if let Some(c) = self.container.take() {
            let _ = self.runtime.block_on(c.stop());
        }
    }
}

fn redis_text(value: &redis::Value) -> Option<&str> {
    match value {
        redis::Value::BulkString(v) => std::str::from_utf8(v).ok(),
        redis::Value::SimpleString(v) => Some(v),
        _ => None,
    }
}
fn decode_ft_search(value: &redis::Value, keys: &mut [Key], distances: &mut [Distance]) -> Result<usize, String> {
    keys.fill(Key::MAX);
    distances.fill(Distance::INFINITY);
    let redis::Value::Array(items) = value else {
        return Err("invalid FT.SEARCH response".into());
    };
    if items.is_empty() || (items.len() - 1) % 2 != 0 {
        return Err("malformed FT.SEARCH response".into());
    }
    let mut found = 0;
    for pair in items[1..].as_chunks::<2>().0.iter().take(keys.len()) {
        let key = redis_text(&pair[0])
            .and_then(|s| s.strip_prefix(PREFIX))
            .and_then(|s| s.parse::<Key>().ok())
            .ok_or("invalid Redis result key")?;
        let redis::Value::Array(fields) = &pair[1] else {
            return Err("invalid Redis result fields".into());
        };
        let distance = fields
            .as_chunks::<2>()
            .0
            .iter()
            .find(|p| redis_text(&p[0]) == Some("__vector_score"))
            .and_then(|p| redis_text(&p[1]))
            .and_then(|s| s.parse::<Distance>().ok())
            .filter(|d| d.is_finite())
            .ok_or("invalid Redis result distance")?;
        keys[found] = key;
        distances[found] = distance;
        found += 1;
    }
    Ok(found)
}

// #region main

/// Reject argument combinations RediSearch will refuse or silently reinterpret,
/// before a container is started.
fn validate(cli: &Cli) -> Result<(), String> {
    for &connectivity in &cli.connectivity {
        for &expansion_add in &cli.expansion_add {
            if expansion_add < connectivity {
                return Err(format!(
                    "--expansion-add {expansion_add} is below --connectivity {connectivity}; \
                     RediSearch cannot build a graph with fewer candidates than edges"
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
            "redis:8.10",
            "retrieval-redis",
            &[(cli.port.into(), 6379)],
            &[],
            &[],
            timeout,
        )
        .await
        .expect("docker start");
        handle
            .wait_for_tcp("localhost", cli.port.into(), timeout)
            .await
            .expect("redis not ready");
        handle
    });

    let mut state = BenchState::load(&cli.common).unwrap_or_else(|e| {
        eprintln!("Failed to load benchmark state: {e}");
        std::process::exit(1);
    });
    eprintln!("- Metrics: {}", spell_list(&cli.metric));
    eprintln!("- Data types: {}", spell_list(&cli.data_type));
    eprintln!("- Connectivity: {}", spell_list(&cli.connectivity));
    eprintln!("- Expansion add: {}", spell_list(&cli.expansion_add));
    eprintln!("- Expansion search: {}", spell_list(&cli.expansion_search));
    eprintln!("- Startup time limit: {}", spell_duration(cli.startup_time_limit));
    eprintln!("- Port: {}", cli.port);
    eprintln!("- Vectors per upsert: {}", cli.vectors_per_upsert);
    if cli.common.dims.len() > 1 {
        retrieval::bail("--dims sweep with >1 value isn't supported on Redis; rerun the binary per dimensions");
    }
    let dimensions = cli.common.dims.first().copied().unwrap_or_else(|| state.dimensions());
    state.check_dimensions(dimensions).unwrap_or_bail("invalid --dims");

    let redis_url = format!("redis://localhost:{}/", cli.port);
    let client = redis::Client::open(redis_url.as_str()).expect("redis client");

    let deadline = std::time::Instant::now() + timeout;
    loop {
        let ready = client
            .get_connection_with_timeout(Duration::from_millis(500))
            .and_then(|mut connection| redis::cmd("PING").query::<String>(&mut connection))
            .is_ok_and(|reply| reply == "PONG");
        if ready {
            break;
        }
        if std::time::Instant::now() >= deadline {
            runtime.block_on(handle.stop()).ok();
            bail("Redis did not answer PING before the startup deadline");
        }
        std::thread::sleep(Duration::from_millis(100));
    }

    let mut container_slot = Some(handle);
    let configs = iproduct!(
        &cli.metric,
        &cli.data_type,
        &cli.connectivity,
        &cli.expansion_add,
        &cli.expansion_search
    );
    let num_configs = configs.clone().count();
    let mut summary = SweepSummary::default();
    for (idx, (&metric, &data_type_name, connectivity, expansion_add, expansion_search)) in configs.enumerate() {
        let is_last = idx + 1 == num_configs;
        let data_type = RedisDataType::from(data_type_name);

        let mut conn = client.get_connection().expect("redis connection");
        let _: () = redis::cmd("FLUSHALL").query(&mut conn).expect("FLUSHALL");
        let _: () = redis::cmd("FT.CREATE")
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
            .arg(dimensions)
            .arg("DISTANCE_METRIC")
            .arg(metric.token())
            .arg("M")
            .arg(*connectivity)
            .arg("EF_CONSTRUCTION")
            .arg(*expansion_add)
            .query(&mut conn)
            .expect("FT.CREATE");

        let container_for_this_run = if is_last { container_slot.take() } else { None };

        let description = format!(
            "redis · {metric} · data_type={data_type_name} · \
             M={connectivity} · ef={expansion_add}/{expansion_search} · {dimensions}d"
        );
        let backend = RedisBackend {
            connection: conn,
            wire: WireScratch::new(data_type),
            pipeline: redis::pipe(),
            container: container_for_this_run,
            runtime: runtime.handle().clone(),
            batch_size: cli.vectors_per_upsert,
            data_type,
            expansion_search: *expansion_search,
            description: description.clone(),
            metadata: {
                let mut metadata = std::collections::HashMap::new();
                metadata.insert("backend".into(), json!("redis"));
                metadata.insert("metric".into(), json!(metric.to_string()));
                metadata.insert("data_type".into(), json!(data_type_name.to_string()));
                metadata.insert("bytes_per_element".into(), json!(data_type.bytes_per_element));
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
        let mut keys = [0; 2];
        let mut distances = [0.0; 2];
        assert_eq!(decode_ft_search(&response, &mut keys, &mut distances).unwrap(), 1);
        assert_eq!(keys, [42, Key::MAX]);
        assert_eq!(distances, [0.25, Distance::INFINITY]);
        let malformed = redis::Value::Array(vec![
            redis::Value::Int(1),
            text(&format!("{PREFIX}42")),
            redis::Value::Array(vec![]),
        ]);
        assert!(decode_ft_search(&malformed, &mut keys, &mut distances).is_err());
    }
    #[test]
    fn conversion_reuses_typed_wire_allocation() {
        let mut wire = WireScratch::F32(Vec::new_in(std::alloc::System));
        let source = VectorSlice::I8(&[1, -2, 3]);
        let pointer = wire.encode(&source, 0, 3).as_ptr();
        assert_eq!(wire.encode(&source, 0, 2).as_ptr(), pointer);
    }
}
