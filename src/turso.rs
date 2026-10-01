//! Turso benchmark binary: exact F32 vector search on the native Rust engine, no Docker.
//!
//! ## Build & Install
//!
//! ```sh
//! cargo install --path . --no-default-features --features turso-backend
//! ```
//!
//! ## Examples
//!
//! ```sh
//! retri-eval-turso \
//!     --base-vectors datasets/wiki_1M/base.1M.fbin \
//!     --query-vectors datasets/wiki_1M/query.public.100K.fbin \
//!     --query-neighbors datasets/wiki_1M/groundtruth.public.100K.ibin \
//!     --metric cos
//! ```

use std::{alloc::System, collections::HashMap, fmt};

use clap::Parser;
use serde_json::{json, Value};

use retrieval::{run, Backend, BenchState, CommonArgs, Distance, Key, UnwrapOrBail, VectorSlice, Vectors};

/// Turso vector distances, as `--metric` spells them.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum Metric {
    L2,
    Cos,
}

impl Metric {
    /// The SQL function that scores a stored vector against a query.
    fn sql_function(self) -> &'static str {
        match self {
            Self::L2 => "vector_distance_l2",
            Self::Cos => "vector_distance_cos",
        }
    }
}

impl fmt::Display for Metric {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        retrieval::spell_value(self, formatter)
    }
}

#[derive(Parser, Debug)]
#[command(name = "retri-eval-turso", about = "Benchmark Turso exact F32 vector search")]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,

    /// Distance metric
    #[arg(long, value_enum, default_value = "l2")]
    metric: Metric,

    /// Database file, or `:memory:`; only its `bench` table is recreated
    #[arg(long, default_value = ":memory:", value_parser = clap::builder::NonEmptyStringValueParser::new())]
    db_path: String,
}

struct TursoBackend {
    connection: turso::Connection,
    runtime: tokio::runtime::Handle,
    metric: Metric,
    search_sql: String,
    wire: Vec<u8, System>,
}

impl TursoBackend {
    async fn open(path: &str, metric: Metric, runtime: tokio::runtime::Handle) -> Result<Self, String> {
        let database = turso::Builder::new_local(path)
            .build()
            .await
            .map_err(|e| e.to_string())?;
        let connection = database.connect().map_err(|e| e.to_string())?;
        connection.execute_batch("PRAGMA synchronous=FULL; DROP TABLE IF EXISTS bench; CREATE TABLE bench (id INTEGER PRIMARY KEY, vector BLOB NOT NULL)")
            .await.map_err(|e| e.to_string())?;
        Ok(Self {
            connection,
            runtime,
            metric,
            search_sql: format!(
                "SELECT id, {}(vector, ?1) AS distance FROM bench ORDER BY distance, id LIMIT ?2",
                metric.sql_function()
            ),
            wire: Vec::new_in(System),
        })
    }
}

fn encode(row: &[f32], wire: &mut Vec<u8, System>) -> Result<(), String> {
    wire.clear();
    for &value in row {
        if !value.is_finite() {
            return Err("Turso requires finite F32 vectors".into());
        }
        wire.extend_from_slice(&value.to_le_bytes());
    }
    Ok(())
}

impl Backend for TursoBackend {
    fn description(&self) -> String {
        format!("turso · {} · F32 exact scan", self.metric)
    }
    fn metadata(&self) -> HashMap<String, Value> {
        HashMap::from([
            ("backend".into(), json!("turso")),
            ("library_version".into(), json!("0.8.1")),
            ("metric".into(), json!(self.metric.to_string())),
            ("data_type".into(), json!("f32")),
            ("index_type".into(), json!("flat")),
            ("synchronous".into(), json!("FULL")),
        ])
    }
    fn add(&mut self, keys: &[Key], vectors: Vectors) -> Result<(), String> {
        let VectorSlice::F32(data) = vectors.data else {
            return Err("Turso supports F32 source vectors only".into());
        };
        self.runtime.block_on(async {
            let transaction = self.connection.transaction().await.map_err(|e| e.to_string())?;
            let mut statement = transaction
                .prepare("INSERT INTO bench VALUES (?1, ?2)")
                .await
                .map_err(|e| e.to_string())?;
            for (row, &key) in data.chunks_exact(vectors.dimensions).zip(keys) {
                encode(row, &mut self.wire)?;
                statement
                    .execute((i64::from(key), self.wire.as_slice()))
                    .await
                    .map_err(|e| e.to_string())?;
            }
            transaction.commit().await.map_err(|e| e.to_string())
        })
    }
    fn search(
        &mut self,
        queries: Vectors,
        count: usize,
        keys: &mut [Key],
        distances: &mut [Distance],
        counts: &mut [usize],
    ) -> Result<(), String> {
        let VectorSlice::F32(data) = queries.data else {
            return Err("Turso supports F32 source vectors only".into());
        };
        keys.fill(Key::MAX);
        distances.fill(Distance::INFINITY);
        counts.fill(0);
        self.runtime.block_on(async {
            let mut statement = self
                .connection
                .prepare_cached(&self.search_sql)
                .await
                .map_err(|e| e.to_string())?;
            for (i, query) in data.chunks_exact(queries.dimensions).enumerate() {
                encode(query, &mut self.wire)?;
                let mut rows = statement
                    .query((self.wire.as_slice(), count as i64))
                    .await
                    .map_err(|e| e.to_string())?;
                while let Some(row) = rows.next().await.map_err(|e| e.to_string())? {
                    let rank = counts[i];
                    if rank == count {
                        return Err("Turso exceeded result limit".into());
                    }
                    let id = row.get::<i64>(0).map_err(|e| e.to_string())?;
                    let distance = row.get::<f64>(1).map_err(|e| e.to_string())? as Distance;
                    if !distance.is_finite() {
                        return Err("Turso returned nonfinite distance".into());
                    }
                    keys[i * count + rank] = Key::try_from(id).map_err(|e| e.to_string())?;
                    distances[i * count + rank] = distance;
                    counts[i] += 1;
                }
            }
            Ok(())
        })
    }
    fn memory_bytes(&self) -> usize {
        0
    }
}

fn main() {
    let cli: Cli = retrieval::parse_cli();

    if cli.common.index.is_some() {
        retrieval::bail("--index is not supported for this backend");
    }

    let mut state = BenchState::load(&cli.common).unwrap_or_bail("benchmark state");
    eprintln!("- Metric: {}", cli.metric);
    eprintln!("- Database path: {}", cli.db_path);
    if cli.common.dims.len() > 1 {
        retrieval::bail("--dims sweep with >1 value isn't supported on Turso; rerun the binary per dimensions");
    }
    let dimensions = cli.common.dims.first().copied().unwrap_or_else(|| state.dimensions());
    state.check_dimensions(dimensions).unwrap_or_bail("invalid --dims");

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio");
    let mut backend = runtime
        .block_on(TursoBackend::open(&cli.db_path, cli.metric, runtime.handle().clone()))
        .unwrap_or_bail("open Turso");
    run(&mut backend, &mut state, dimensions).unwrap_or_bail("benchmark");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_search_reuses_wire_and_reports_short_rows() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let mut backend = runtime
            .block_on(TursoBackend::open(":memory:", Metric::L2, runtime.handle().clone()))
            .unwrap();
        backend
            .add(
                &[9, 2],
                Vectors {
                    data: VectorSlice::F32(&[3.0, 4.0, 0.0, 0.0]),
                    dimensions: 2,
                },
            )
            .unwrap();
        let pointer = backend.wire.as_ptr();
        let mut keys = [0; 3];
        let mut distances = [0.0; 3];
        let mut counts = [0];
        backend
            .search(
                Vectors {
                    data: VectorSlice::F32(&[0.0, 0.0]),
                    dimensions: 2,
                },
                3,
                &mut keys,
                &mut distances,
                &mut counts,
            )
            .unwrap();
        assert_eq!(keys, [2, 9, Key::MAX]);
        assert_eq!(distances, [0.0, 5.0, Distance::INFINITY]);
        assert_eq!(counts, [2]);
        assert_eq!(pointer, backend.wire.as_ptr());
    }
    #[test]
    fn cosine_distance_is_evaluated_by_engine() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let mut backend = runtime
            .block_on(TursoBackend::open(":memory:", Metric::Cos, runtime.handle().clone()))
            .unwrap();
        backend
            .add(
                &[9, 2],
                Vectors {
                    data: VectorSlice::F32(&[0.0, 1.0, 1.0, 0.0]),
                    dimensions: 2,
                },
            )
            .unwrap();
        let mut keys = [0; 2];
        let mut distances = [0.0; 2];
        let mut counts = [0];
        backend
            .search(
                Vectors {
                    data: VectorSlice::F32(&[1.0, 0.0]),
                    dimensions: 2,
                },
                2,
                &mut keys,
                &mut distances,
                &mut counts,
            )
            .unwrap();
        assert_eq!(keys, [2, 9]);
        assert!(distances[0].abs() < 1e-6);
        assert!((distances[1] - 1.0).abs() < 1e-6);
        assert_eq!(counts, [2]);
    }
}
