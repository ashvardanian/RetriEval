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
//!     --vectors datasets/wiki_1M/base.1M.fbin \
//!     --queries datasets/wiki_1M/query.public.100K.fbin \
//!     --neighbors datasets/wiki_1M/groundtruth.public.100K.ibin \
//!     --metric ip \
//!     --output results/
//! ```

use std::cell::RefCell;
use std::time::Duration;

use clap::Parser;
use itertools::iproduct;
use retrieval::docker::ContainerHandle;
use retrieval::{bail, pod_slice_as_bytes, run, Backend, BenchState, CommonArgs, Distance, Key, VectorSlice, Vectors};
use serde_json::json;

const INDEX_NAME: &str = "bench_idx";
const PREFIX: &str = "vec:";

fn parse_redis_metric(s: &str) -> Result<&'static str, String> {
    match s {
        "ip" => Ok("IP"),
        "cos" => Ok("COSINE"),
        "l2sq" | "l2" => Ok("L2"),
        _ => Err(format!(
            "unknown Redis metric: {s} (supported: ip, cos, l2; engine does not expose Hamming / \
             Jaccard on vector indexes)"
        )),
    }
}

/// RediSearch `FT.CREATE VECTOR ... TYPE <name>` strings.
/// Redis 8+ adds the half-float and narrow-int dtypes; older redis-stack
/// images only accept the four float formats. Bytes-per-element is fixed by
/// the type and used to size the RESP payload.
#[derive(Clone, Copy)]
struct RedisDtype {
    token: &'static str,
    bytes_per_element: usize,
}

fn parse_redis_dtype(s: &str) -> Result<RedisDtype, String> {
    match s {
        "f32" => Ok(RedisDtype {
            token: "FLOAT32",
            bytes_per_element: 4,
        }),
        "f64" => Ok(RedisDtype {
            token: "FLOAT64",
            bytes_per_element: 8,
        }),
        "f16" => Ok(RedisDtype {
            token: "FLOAT16",
            bytes_per_element: 2,
        }),
        "bf16" => Ok(RedisDtype {
            token: "BFLOAT16",
            bytes_per_element: 2,
        }),
        "u8" => Ok(RedisDtype {
            token: "UINT8",
            bytes_per_element: 1,
        }),
        "i8" => Ok(RedisDtype {
            token: "INT8",
            bytes_per_element: 1,
        }),
        _ => Err(format!(
            "unknown Redis data_type: {s} (supported on Redis 8+: f32, f64, f16, bf16, u8, i8)"
        )),
    }
}

/// Pipeline-batch of rows pre-encoded for the wire. Either borrows the source slice directly (source
/// element type matches target wire type) or owns a single `Vec<Target>` produced by one
/// `numkong::cast` from the source to the target type. Row bytes are served by indexing into the
/// stored buffer — `Cmd::write_arg` already memcpys into its own data buffer, so we hand it a
/// borrowed `&[u8]` and skip the per-row `Vec<u8>` the earlier design allocated.
enum EncodedBatch<'source> {
    Identity(&'source [u8]),
    CastF32(Vec<f32>),
    CastF64(Vec<f64>),
    CastF16(Vec<numkong::f16>),
    CastBF16(Vec<numkong::bf16>),
    CastU8(Vec<u8>),
    CastI8(Vec<i8>),
}

impl EncodedBatch<'_> {
    fn row_bytes(&self, row_within_batch: usize, bytes_per_row: usize) -> &[u8] {
        // SAFETY: every variant is a dense slice of POD elements; the stored length covers full rows.
        let all_bytes: &[u8] = match self {
            Self::Identity(bytes) => bytes,
            Self::CastF32(values) => unsafe { pod_slice_as_bytes(values) },
            Self::CastF64(values) => unsafe { pod_slice_as_bytes(values) },
            Self::CastF16(values) => unsafe { pod_slice_as_bytes(values) },
            Self::CastBF16(values) => unsafe { pod_slice_as_bytes(values) },
            Self::CastU8(values) => values,
            Self::CastI8(values) => unsafe { pod_slice_as_bytes(values) },
        };
        let start = row_within_batch * bytes_per_row;
        &all_bytes[start..start + bytes_per_row]
    }
}

/// Encode one pipeline's worth of rows for the wire in a single pass.
///
/// Fast paths: if the source element type already matches the target `TYPE` token, borrow the
/// source bytes directly — zero allocation, zero cast work. Otherwise allocate one `Vec<Target>`
/// for the whole batch and hand it to `numkong::cast` once; every row is then a borrow into the
/// owned buffer. Replaces the earlier per-row shape that allocated `2 × num_rows` vectors per
/// pipeline and round-tripped through `f32` even when source and target matched.
fn encode_batch<'source>(
    data_type: RedisDtype,
    source: &'source VectorSlice<'_>,
    start_row: usize,
    num_rows: usize,
    dimensions: usize,
) -> EncodedBatch<'source> {
    let elements = num_rows * dimensions;
    let start = start_row * dimensions;
    let end = start + elements;

    // Zero-copy identity paths: the source slice is already in the wire format Redis wants.
    // SAFETY: element types are POD, dense-packed row-major.
    match (source, data_type.token) {
        (VectorSlice::F32(data), "FLOAT32") => {
            return EncodedBatch::Identity(unsafe { pod_slice_as_bytes(&data[start..end]) })
        }
        (VectorSlice::I8(data), "INT8") => {
            return EncodedBatch::Identity(unsafe { pod_slice_as_bytes(&data[start..end]) })
        }
        (VectorSlice::U8(data), "UINT8") => return EncodedBatch::Identity(&data[start..end]),
        (VectorSlice::B1x8(_), _) => unreachable!("parse_redis_metric rejects bit-packed inputs"),
        _ => {}
    }

    // Cast path: one `numkong::cast` from source element type straight to target type. No f32 hop
    // even when source ≠ f32 (i.e. `.u8bin` + `--data_type f16` goes u8 → f16 directly).
    match data_type.token {
        "FLOAT32" => EncodedBatch::CastF32(cast_batch(source, start, end, elements)),
        "FLOAT64" => EncodedBatch::CastF64(cast_batch(source, start, end, elements)),
        "FLOAT16" => EncodedBatch::CastF16(cast_batch(source, start, end, elements)),
        "BFLOAT16" => EncodedBatch::CastBF16(cast_batch(source, start, end, elements)),
        "UINT8" => EncodedBatch::CastU8(cast_batch(source, start, end, elements)),
        "INT8" => EncodedBatch::CastI8(cast_batch(source, start, end, elements)),
        token => unreachable!("unknown Redis data_type token {token}"),
    }
}

/// Allocate an uninitialized `Vec<Target>` of exactly `elements` slots, dispatch on the source
/// variant, and let `numkong::cast` fill it in one SIMD batch. Skips the `vec![T::default(); n]`
/// zero-fill because `nk_cast` writes every element before we read it back.
fn cast_batch<Target>(source: &VectorSlice<'_>, start: usize, end: usize, elements: usize) -> Vec<Target>
where
    Target: numkong::CastDtype + Copy,
{
    let mut target: Vec<Target> = Vec::with_capacity(elements);
    // SAFETY: `target.as_mut_ptr()` points to `elements` valid-for-write slots (reserved by
    //         `with_capacity`). `numkong::cast` writes every slot via a C-side memcpy/SIMD store;
    //         we only reveal those slots through `set_len` after the cast returns. Target types
    //         are all POD so any bit pattern produced by the cast is a valid value.
    unsafe {
        let slot = std::slice::from_raw_parts_mut(target.as_mut_ptr(), elements);
        let cast_outcome = match source {
            VectorSlice::F32(data) => numkong::cast(&data[start..end], slot),
            VectorSlice::I8(data) => numkong::cast(&data[start..end], slot),
            VectorSlice::U8(data) => numkong::cast(&data[start..end], slot),
            VectorSlice::B1x8(_) => unreachable!(),
        };
        cast_outcome.expect("numkong::cast: source and target slice lengths must match (checked above)");
        target.set_len(elements);
    }
    target
}

// #region CLI

#[derive(Parser, Debug)]
#[command(name = "retri-eval-redis", about = "Benchmark Redis/RediSearch")]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,

    /// Distance metric (comma-separated for sweep): ip, cos, l2
    #[arg(long, value_delimiter = ',', default_value = "l2")]
    metric: Vec<String>,

    /// FT.CREATE VECTOR TYPE (comma-separated for sweep): f32, f64, f16, bf16, u8, i8
    /// (requires Redis 8+ for anything other than f32/f64/f16/bf16)
    #[arg(long, value_delimiter = ',', default_value = "f32")]
    data_type: Vec<String>,

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
    data_type: RedisDtype,
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

    fn add(&mut self, keys: &[Key], vectors: Vectors) -> Result<(), String> {
        let dimensions = vectors.dimensions;
        let num_vectors = vectors.len();
        let bytes_per_row = dimensions * self.data_type.bytes_per_element;

        for batch_start in (0..num_vectors).step_by(self.batch_size) {
            let batch_end = (batch_start + self.batch_size).min(num_vectors);
            let batch_rows = batch_end - batch_start;
            let encoded = encode_batch(self.data_type, &vectors.data, batch_start, batch_rows, dimensions);

            let mut pipe = redis::pipe();
            for row_within_batch in 0..batch_rows {
                let global_index = batch_start + row_within_batch;
                let key = format!("{PREFIX}{}", keys[global_index]);
                pipe.cmd("HSET")
                    .arg(&key)
                    .arg("vector")
                    .arg(encoded.row_bytes(row_within_batch, bytes_per_row));
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
        let dimensions = queries.dimensions;
        let num_vectors = queries.len();
        let bytes_per_row = dimensions * self.data_type.bytes_per_element;
        let query_str = format!("*=>[KNN {count} @vector $BLOB]");

        // FT.SEARCH is one request per query (no pipelining for vector search), so encode the whole
        // query batch once and index into it per iteration — single allocation for num_vectors rows.
        let encoded = encode_batch(self.data_type, &queries.data, 0, num_vectors, dimensions);

        for query_index in 0..num_vectors {
            let query_bytes = encoded.row_bytes(query_index, bytes_per_row);

            let raw: redis::Value = redis::cmd("FT.SEARCH")
                .arg(INDEX_NAME)
                .arg(&query_str)
                .arg("PARAMS")
                .arg("2")
                .arg("BLOB")
                .arg(query_bytes)
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

            let offset = query_index * count;
            let pairs = parse_ft_search(&raw);
            let found = pairs.len().min(count);
            for (rank, (id, score)) in pairs.iter().enumerate().take(count) {
                out_keys[offset + rank] = *id;
                out_distances[offset + rank] = *score;
            }
            for rank in found..count {
                out_keys[offset + rank] = Key::MAX;
                out_distances[offset + rank] = Distance::INFINITY;
            }
            out_counts[query_index] = found;
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

    for m in &cli.metric {
        parse_redis_metric(m).unwrap_or_else(|e| bail(&e));
    }
    for d in &cli.data_type {
        parse_redis_dtype(d).unwrap_or_else(|e| bail(&e));
    }

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio");
    let timeout = Duration::from_secs(cli.docker_timeout);

    let handle = runtime.block_on(async {
        let handle = ContainerHandle::start("redis:8.6", "retrieval-redis", &vec![(cli.port, 6379)], &[], timeout)
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
    if cli.common.dimensions.len() > 1 {
        retrieval::bail("--dimensions sweep with >1 value isn't supported on Redis; rerun the binary per dimensions");
    }
    let dimensions = cli.common.dimensions.first().copied().unwrap_or_else(|| state.dimensions());
    state
        .check_dimensions(dimensions)
        .unwrap_or_else(|e| retrieval::bail(&format!("invalid --dimensions: {e}")));

    let redis_url = format!("redis://localhost:{}/", cli.port);
    let client = redis::Client::open(redis_url.as_str()).expect("redis client");

    let mut container_slot = Some(handle);
    let num_configs = cli.metric.len() * cli.data_type.len();
    for (idx, (metric_str, dtype_str)) in iproduct!(&cli.metric, &cli.data_type).enumerate() {
        let is_last = idx + 1 == num_configs;
        let metric = parse_redis_metric(metric_str).expect("metric validated above");
        let data_type = parse_redis_dtype(dtype_str).expect("data_type validated above");

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
            .arg(metric)
            .arg("M")
            .arg(cli.connectivity)
            .arg("EF_CONSTRUCTION")
            .arg(cli.expansion_add)
            .query(&mut conn)
            .expect("FT.CREATE");

        let container_for_this_run = if is_last { container_slot.take() } else { None };

        let mut backend = RedisBackend {
            connection: RefCell::new(conn),
            container: container_for_this_run,
            runtime: runtime.handle().clone(),
            batch_size: cli.batch_size,
            data_type,
            description: format!(
                "redis · {metric_str} · data_type={dtype_str} · M={} · ef={} · {dimensions}d",
                cli.connectivity, cli.expansion_add,
            ),
            metadata: {
                let mut metadata = std::collections::HashMap::new();
                metadata.insert("backend".into(), json!("redis"));
                metadata.insert("metric".into(), json!(metric_str));
                metadata.insert("data_type".into(), json!(dtype_str));
                metadata.insert("bytes_per_element".into(), json!(data_type.bytes_per_element));
                metadata.insert("connectivity".into(), json!(cli.connectivity));
                metadata.insert("expansion_add".into(), json!(cli.expansion_add));
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
