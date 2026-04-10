//! Redis/RediSearch benchmark binary.
//!
//! ```sh
//! cargo run --release --bin bench-redis --features redis-backend -- \
//!     --vectors datasets/wiki_1M/base.1M.fbin \
//!     --queries datasets/wiki_1M/query.public.100K.fbin \
//!     --neighbors datasets/wiki_1M/groundtruth.public.100K.ibin \
//!     --metric ip \
//!     --output wiki-1M-redis.jsonl
//! ```

use std::cell::RefCell;
use std::time::Duration;

use clap::Parser;
use usearch_bench::docker::ContainerHandle;
use usearch_bench::{run, Backend, BenchState, CommonArgs, Distance, Key, Vectors};

const INDEX_NAME: &str = "bench_idx";
const PREFIX: &str = "vec:";
const BATCH_SIZE: usize = 1_000;

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
#[command(name = "bench-redis", about = "Benchmark Redis/RediSearch")]
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
}

// #region Backend

struct RedisBackend {
    connection: RefCell<redis::Connection>,
    container: Option<ContainerHandle>,
    runtime: tokio::runtime::Handle,
    description: String,
}

unsafe impl Send for RedisBackend {}

impl Backend for RedisBackend {
    fn description(&self) -> String {
        self.description.clone()
    }

    fn add(&mut self, _keys: &[Key], vectors: Vectors) -> Result<(), String> {
        let data = vectors.data.to_f32();
        let dimensions = vectors.dimensions;
        let num_vectors = data.len() / dimensions;

        for batch_start in (0..num_vectors).step_by(BATCH_SIZE) {
            let batch_end = (batch_start + BATCH_SIZE).min(num_vectors);
            let mut pipe = redis::pipe();
            for i in batch_start..batch_end {
                let vec_bytes: Vec<u8> = data[i * dimensions..(i + 1) * dimensions]
                    .iter()
                    .flat_map(|f| f.to_le_bytes())
                    .collect();
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

        for q in 0..num_vectors {
            let query_bytes: Vec<u8> = data[q * dimensions..(q + 1) * dimensions]
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();

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
                    if let redis::Value::BulkString(b) = &fields[j] {
                        if String::from_utf8_lossy(b) == "__vector_score" {
                            if let redis::Value::BulkString(b) = &fields[j + 1] {
                                score = String::from_utf8_lossy(b).parse().unwrap_or(0.0);
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
            "usearch-bench-redis",
            &vec![(6379, 6379)],
            &[],
            timeout,
        )
        .await
        .expect("docker start");
        handle
            .wait_for_tcp("localhost", 6379, timeout)
            .await
            .expect("redis not ready");
        handle
    });

    let mut state = BenchState::load(&cli.common).unwrap_or_else(|e| {
        eprintln!("Failed to load benchmark state: {e}");
        std::process::exit(1);
    });
    let dimensions = state.dimensions();

    let client = redis::Client::open("redis://localhost:6379/").expect("redis client");
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
        description: format!(
            "redis · {} · M={} · ef={} · {dimensions}d",
            cli.metric, cli.connectivity, cli.expansion_add,
        ),
    };

    run(&mut backend, &mut state).unwrap_or_else(|e| {
        eprintln!("Benchmark failed: {e}");
        std::process::exit(1);
    });
    eprintln!("Benchmark complete.");
}
