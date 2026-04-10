//! Redis/RediSearch benchmark binary.
//!
//! ```sh
//! cargo run --release --bin retri-eval-redis --features redis-backend -- \
//!     --vectors datasets/wiki_1M/base.1M.fbin \
//!     --queries datasets/wiki_1M/query.public.100K.fbin \
//!     --neighbors datasets/wiki_1M/groundtruth.public.100K.ibin \
//!     --metric ip \
//!     --output wiki-1M-redis.jsonl
//! ```

use std::cell::RefCell;
use std::time::Duration;

use clap::Parser;
use retrieval::docker::ContainerHandle;
use retrieval::{run, Backend, BenchState, CommonArgs, Distance, Key, Vectors};

const INDEX_NAME: &str = "bench_idx";
const PREFIX: &str = "vec:";

// #region Local metric mapping

fn parse_redis_metric(s: &str) -> Result<&'static str, String> {
    match s {
        "ip" => Ok("IP"),
        "cos" => Ok("COSINE"),
        "l2sq" | "l2" => Ok("L2"),
        _ => Err(format!("unknown Redis distance metric: {s}")),
    }
}

// #region CLI

#[derive(Parser, Debug)]
#[command(name = "retri-eval-redis", about = "Benchmark Redis/RediSearch")]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,

    #[arg(long, default_value = "l2")]
    metric: String,

    #[arg(long, default_value_t = 16)]
    connectivity: usize,

    #[arg(long, default_value_t = 128)]
    expansion_add: usize,

    #[arg(long, default_value_t = 120)]
    docker_timeout: u64,

    /// Redis port
    #[arg(long, default_value_t = 6379)]
    port: u16,

    /// Batch size for pipeline operations
    #[arg(long, default_value_t = 1_000)]
    batch_size: usize,
}

// #region Backend

struct RedisBackend {
    connection: RefCell<redis::Connection>,
    container: Option<ContainerHandle>,
    runtime: tokio::runtime::Handle,
    batch_size: usize,
    description: String,
    metadata: std::collections::HashMap<String, serde_json::Value>,
}

unsafe impl Send for RedisBackend {}

impl Backend for RedisBackend {
    fn description(&self) -> String {
        self.description.clone()
    }

    fn metadata(&self) -> std::collections::HashMap<String, serde_json::Value> {
        self.metadata.clone()
    }

    fn add(&mut self, _keys: &[Key], vectors: Vectors) -> Result<(), String> {
        let data = vectors.data.to_f32();
        let dimensions = vectors.dimensions;
        let num_vectors = data.len() / dimensions;

        let mut vec_bytes = vec![0u8; dimensions * 4];

        for batch_start in (0..num_vectors).step_by(self.batch_size) {
            let batch_end = (batch_start + self.batch_size).min(num_vectors);
            let mut pipe = redis::pipe();
            for i in batch_start..batch_end {
                for (j, f) in data[i * dimensions..(i + 1) * dimensions]
                    .iter()
                    .enumerate()
                {
                    vec_bytes[j * 4..(j + 1) * 4].copy_from_slice(&f.to_le_bytes());
                }
                let key = format!("{PREFIX}{i}");
                pipe.cmd("HSET").arg(&key).arg("vector").arg(&vec_bytes[..]);
            }
            let _: () = pipe
                .query(&mut *self.connection.borrow_mut())
                .map_err(|e| format!("Redis HSET failed: {e}"))?;
        }
        Ok(())
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
        let query_str = format!("*=>[KNN {count} @vector $BLOB]");
        let mut query_bytes = vec![0u8; dimensions * 4];

        for q in 0..num_vectors {
            for (i, f) in data[q * dimensions..(q + 1) * dimensions]
                .iter()
                .enumerate()
            {
                query_bytes[i * 4..(i + 1) * 4].copy_from_slice(&f.to_le_bytes());
            }

            let raw: redis::Value = redis::cmd("FT.SEARCH")
                .arg(INDEX_NAME)
                .arg(&query_str)
                .arg("PARAMS")
                .arg("2")
                .arg("BLOB")
                .arg(&query_bytes[..])
                .arg("DIALECT")
                .arg("2")
                .arg("SORTBY")
                .arg("__vector_score")
                .arg("ASC")
                .arg("LIMIT")
                .arg("0")
                .arg(count)
                .query(&mut *self.connection.borrow_mut())
                .map_err(|e| format!("FT.SEARCH failed: {e}"))?;

            let offset = q * count;
            let pairs = parse_ft_search(&raw);
            let found = pairs.len().min(count);
            for (j, (id, score)) in pairs.iter().enumerate().take(count) {
                out_keys[offset + j] = *id;
                out_distances[offset + j] = *score;
            }
            for j in found..count {
                out_keys[offset + j] = Key::MAX;
                out_distances[offset + j] = Distance::INFINITY;
            }
            out_counts[q] = found;
        }
        Ok(())
    }

    fn memory_bytes(&self) -> usize {
        0
    }
}

impl Drop for RedisBackend {
    fn drop(&mut self) {
        if let Some(c) = self.container.take() {
            let _ = self.runtime.block_on(c.stop());
        }
    }
}

fn parse_ft_search(value: &redis::Value) -> Vec<(Key, Distance)> {
    let mut pairs = Vec::new();
    if let redis::Value::Array(ref items) = value {
        let mut i = 1;
        while i + 1 < items.len() {
            let key_str = match &items[i] {
                redis::Value::BulkString(b) => String::from_utf8_lossy(b).to_string(),
                redis::Value::SimpleString(s) => s.clone(),
                _ => {
                    i += 2;
                    continue;
                }
            };
            let id: Key = key_str
                .strip_prefix(PREFIX)
                .and_then(|s| s.parse().ok())
                .unwrap_or(Key::MAX);
            let mut score = 0.0f32;
            if let redis::Value::Array(ref fields) = items[i + 1] {
                let mut j = 0;
                while j + 1 < fields.len() {
                    if let redis::Value::BulkString(name) = &fields[j] {
                        if name == b"__vector_score" {
                            if let redis::Value::BulkString(val) = &fields[j + 1] {
                                score = std::str::from_utf8(val)
                                    .ok()
                                    .and_then(|s| s.parse().ok())
                                    .unwrap_or(0.0);
                            }
                        }
                    }
                    j += 2;
                }
            }
            pairs.push((id, score));
            i += 2;
        }
    }
    pairs
}

// #region main

fn main() {
    let cli = Cli::parse();
    let distance_metric = parse_redis_metric(&cli.metric).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio");
    let timeout = Duration::from_secs(cli.docker_timeout);

    let handle = runtime.block_on(async {
        let handle = ContainerHandle::start(
            "redis/redis-stack:latest",
            "retrieval-redis",
            &vec![(cli.port, 6379)],
            &[],
            timeout,
        )
        .await
        .expect("docker start");
        handle
            .wait_for_tcp("localhost", cli.port, timeout)
            .await
            .expect("redis not ready");
        handle
    });

    let mut state = BenchState::load(&cli.common).unwrap_or_else(|e| {
        eprintln!("Failed to load benchmark state: {e}");
        std::process::exit(1);
    });
    let dimensions = state.dimensions();

    let redis_url = format!("redis://localhost:{}/", cli.port);
    let client = redis::Client::open(redis_url.as_str()).expect("redis client");
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
        .arg("FLOAT32")
        .arg("DIM")
        .arg(dimensions)
        .arg("DISTANCE_METRIC")
        .arg(distance_metric)
        .arg("M")
        .arg(cli.connectivity)
        .arg("EF_CONSTRUCTION")
        .arg(cli.expansion_add)
        .query(&mut conn)
        .expect("FT.CREATE");

    let mut backend = RedisBackend {
        connection: RefCell::new(conn),
        container: Some(handle),
        runtime: runtime.handle().clone(),
        batch_size: cli.batch_size,
        description: format!(
            "redis · {} · M={} · ef={} · {dimensions}d",
            cli.metric, cli.connectivity, cli.expansion_add,
        ),
        metadata: {
            use serde_json::json;
            let mut metadata = std::collections::HashMap::new();
            metadata.insert("backend".into(), json!("redis"));
            metadata.insert("metric".into(), json!(&cli.metric));
            metadata.insert("connectivity".into(), json!(cli.connectivity));
            metadata.insert("expansion_add".into(), json!(cli.expansion_add));
            m
        },
    };

    run(&mut backend, &mut state).unwrap_or_else(|e| {
        eprintln!("Benchmark failed: {e}");
        std::process::exit(1);
    });
    eprintln!("Benchmark complete.");
}
