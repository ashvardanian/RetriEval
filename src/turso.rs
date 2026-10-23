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

use retrieval::{
    run_config, Backend, BatchKeys, BenchState, CommonArgs, Distance, Hits, IndexFile, SweepSummary, UnwrapOrBail,
    VectorSlice, Vectors,
};

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
    #[arg(long, env = "RETRIEVAL_METRIC", value_enum, default_value = "l2")]
    metric: Metric,

    /// Database file, or `:memory:`; only its `bench` table is recreated
    #[arg(long, env = "RETRIEVAL_DB_PATH", default_value = ":memory:", value_parser = clap::builder::NonEmptyStringValueParser::new())]
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
            .map_err(|error| error.to_string())?;
        let connection = database.connect().map_err(|error| error.to_string())?;
        connection.execute_batch("PRAGMA synchronous=FULL; DROP TABLE IF EXISTS bench; CREATE TABLE bench (id INTEGER PRIMARY KEY, vector BLOB NOT NULL)")
            .await.map_err(|error| error.to_string())?;
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

    /// Rows are the `INTEGER PRIMARY KEY`, SQLite's own rowid, and hits name them back as keys.
    fn add(&mut self, keys: BatchKeys<'_>, vectors: Vectors) -> Result<(), String> {
        let VectorSlice::F32(values) = vectors.values() else {
            return Err("Turso supports F32 source vectors only".into());
        };
        self.runtime.block_on(async {
            let transaction = self.connection.transaction().await.map_err(|error| error.to_string())?;
            let mut statement = transaction
                .prepare_cached("INSERT INTO bench VALUES (?1, ?2)")
                .await
                .map_err(|error| error.to_string())?;
            for (vector, row) in values.chunks_exact(vectors.dimensions()).zip(keys.rows()) {
                encode(vector, &mut self.wire)?;
                statement
                    .execute((row as i64, self.wire.as_slice()))
                    .await
                    .map_err(|error| error.to_string())?;
            }
            transaction.commit().await.map_err(|error| error.to_string())
        })
    }

    fn search(&mut self, queries: Vectors, mut hits: Hits<'_>) -> Result<(), String> {
        let VectorSlice::F32(values) = queries.values() else {
            return Err("Turso supports F32 source vectors only".into());
        };
        let count = hits.count();
        self.runtime.block_on(async {
            let mut statement = self
                .connection
                .prepare_cached(&self.search_sql)
                .await
                .map_err(|error| error.to_string())?;
            for (query, hit_row) in values.chunks_exact(queries.dimensions()).zip(hits.rows()) {
                encode(query, &mut self.wire)?;
                let mut rows = statement
                    .query((self.wire.as_slice(), count as i64))
                    .await
                    .map_err(|error| error.to_string())?;
                let mut found = Vec::with_capacity_in(count, System);
                while let Some(row) = rows.next().await.map_err(|error| error.to_string())? {
                    let id = row.get::<i64>(0).map_err(|error| error.to_string())?;
                    let distance = row.get::<f64>(1).map_err(|error| error.to_string())? as Distance;
                    found.push((id, distance));
                }
                hit_row.write_rows(found.into_iter().map(|(id, distance)| {
                    let row = usize::try_from(id).map_err(|error| error.to_string())?;
                    if distance.is_finite() {
                        Ok((row, distance))
                    } else {
                        Err("Turso returned nonfinite distance".into())
                    }
                }))?;
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
    let mut state = BenchState::load(&cli.common).unwrap_or_bail("benchmark state");
    eprintln!("- Metric: {}", cli.metric);
    eprintln!("- Database path: {}", cli.db_path);
    let dimensions_sweep = cli.common.dimensions_sweep(state.dimensions());

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap_or_bail("tokio runtime");
    let mut summary = SweepSummary::default();
    for &dimensions in &dimensions_sweep {
        let description = format!("turso · {} · F32 exact scan · {dimensions}d", cli.metric);
        let backend = runtime.block_on(TursoBackend::open(&cli.db_path, cli.metric, runtime.handle().clone()));
        summary.record(run_config(
            &description,
            IndexFile::None,
            backend,
            &mut state,
            dimensions,
        ));
    }
    summary.finish();
}

#[cfg(test)]
mod tests {
    use super::*;
    use retrieval::{BatchRows, Key, KeyType, Keys, SearchBuffers};

    static KEYS: Keys = Keys::Generated {
        count: 2,
        key_type: KeyType::UuidV7,
    };

    fn both_rows() -> BatchKeys<'static> {
        BatchKeys::new(&KEYS, BatchRows::Range(0..2))
    }

    fn open(metric: Metric, runtime: &tokio::runtime::Runtime) -> TursoBackend {
        runtime
            .block_on(TursoBackend::open(":memory:", metric, runtime.handle().clone()))
            .unwrap()
    }

    #[test]
    fn exact_search_reuses_wire_and_reports_short_rows() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let mut backend = open(Metric::L2, &runtime);
        backend
            .add(both_rows(), Vectors::new(VectorSlice::F32(&[3.0, 4.0, 0.0, 0.0]), 2))
            .unwrap();
        let pointer = backend.wire.as_ptr();
        let mut buffers = SearchBuffers::new(1, 3).unwrap();
        backend
            .search(
                Vectors::new(VectorSlice::F32(&[0.0, 0.0]), 2),
                buffers.hits(0..1, &KEYS),
            )
            .unwrap();
        let (keys, distances, found) = buffers.rows().next().unwrap();
        assert_eq!(keys, [KEYS.get(1), KEYS.get(0), Key::MISSING]);
        assert_eq!(distances, [0.0, 5.0, Distance::INFINITY]);
        assert_eq!(found, 2);
        assert_eq!(pointer, backend.wire.as_ptr());
    }

    #[test]
    fn cosine_distance_is_evaluated_by_engine() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let mut backend = open(Metric::Cos, &runtime);
        backend
            .add(both_rows(), Vectors::new(VectorSlice::F32(&[0.0, 1.0, 1.0, 0.0]), 2))
            .unwrap();
        let mut buffers = SearchBuffers::new(1, 2).unwrap();
        backend
            .search(
                Vectors::new(VectorSlice::F32(&[1.0, 0.0]), 2),
                buffers.hits(0..1, &KEYS),
            )
            .unwrap();
        let (keys, distances, found) = buffers.rows().next().unwrap();
        assert_eq!(keys, [KEYS.get(1), KEYS.get(0)]);
        assert!(distances[0].abs() < 1e-6);
        assert!((distances[1] - 1.0).abs() < 1e-6);
        assert_eq!(found, 2);
    }
}
