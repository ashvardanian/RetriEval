//! LanceDB benchmark binary (in-process, no Docker).
//!
//! ## Build & Install
//!
//! No extra system dependencies — LanceDB is an in-process library:
//!
//! ```sh
//! cargo install --path . --features lancedb-backend
//! ```
//!
//! ## Examples
//!
//! ```sh
//! retri-eval-lancedb \
//!     --vectors datasets/wiki_1M/base.1M.fbin \
//!     --queries datasets/wiki_1M/query.public.100K.fbin \
//!     --neighbors datasets/wiki_1M/groundtruth.public.100K.ibin \
//!     --metric ip \
//!     --output results/
//! ```

use std::sync::Arc;

use arrow_array::{Float32Array, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use clap::Parser;
use futures_util::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use retrieval::{run, Backend, BenchState, CommonArgs, Distance, Key, UnwrapOrBail, Vectors};
use serde_json::json;

const TABLE_NAME: &str = "bench";

// #region Local metric mapping

fn parse_lancedb_metric(s: &str) -> Result<lancedb::DistanceType, String> {
    match s {
        "ip" => Ok(lancedb::DistanceType::Dot),
        "cos" => Ok(lancedb::DistanceType::Cosine),
        "l2sq" | "l2" => Ok(lancedb::DistanceType::L2),
        _ => Err(format!("unknown LanceDB distance metric: {s}")),
    }
}

// #region CLI

#[derive(Parser, Debug)]
#[command(name = "retri-eval-lancedb", about = "Benchmark LanceDB")]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,

    #[arg(long, default_value = "l2")]
    metric: String,

    /// Path for LanceDB storage
    #[arg(long, default_value = "/tmp/retrieval-lancedb")]
    db_path: String,
}

// #region Backend

struct LanceDbBackend {
    db: lancedb::Connection,
    table: Option<lancedb::Table>,
    dimensions: usize,
    metric: String,
    runtime: tokio::runtime::Handle,
    description: String,
    metadata: std::collections::HashMap<String, serde_json::Value>,
}

impl LanceDbBackend {
    fn schema(&self) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    self.dimensions as i32,
                ),
                false,
            ),
        ]))
    }

    fn make_batch(&self, keys: &[Key], data: &[f32]) -> RecordBatch {
        let schema = self.schema();
        let ids = UInt64Array::from(keys.iter().map(|&k| k as u64).collect::<Vec<_>>());
        let values = Float32Array::from(data.to_vec());
        let list_field = Arc::new(Field::new("item", DataType::Float32, true));
        let vectors =
            arrow_array::FixedSizeListArray::try_new(list_field, self.dimensions as i32, Arc::new(values), None)
                .expect("vector array");
        RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(vectors)]).expect("batch")
    }
}

impl Backend for LanceDbBackend {
    fn description(&self) -> String {
        self.description.clone()
    }

    fn metadata(&self) -> std::collections::HashMap<String, serde_json::Value> {
        self.metadata.clone()
    }

    fn add(&mut self, keys: &[Key], vectors: Vectors) -> Result<(), String> {
        let data = vectors.data.to_f32();
        let batch = self.make_batch(keys, &data);

        self.runtime.block_on(async {
            match &self.table {
                None => {
                    let table = self
                        .db
                        .create_table(TABLE_NAME, vec![batch])
                        .execute()
                        .await
                        .map_err(|e| format!("LanceDB create: {e}"))?;
                    self.table = Some(table);
                }
                Some(table) => {
                    table
                        .add(vec![batch])
                        .execute()
                        .await
                        .map_err(|e| format!("LanceDB add: {e}"))?;
                }
            }
            Ok::<(), String>(())
        })
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
        let table = self.table.as_ref().ok_or("no table created")?;

        let lance_metric =
            parse_lancedb_metric(&self.metric).map_err(|e| format!("unsupported metric for LanceDB: {e}"))?;

        self.runtime.block_on(async {
            for query_index in 0..num_vectors {
                let query = data[query_index * dimensions..(query_index + 1) * dimensions].to_vec();
                let results = table
                    .vector_search(query)
                    .map_err(|e| format!("LanceDB query build: {e}"))?
                    .distance_type(lance_metric)
                    .limit(count)
                    .execute()
                    .await
                    .map_err(|e| format!("LanceDB search: {e}"))?;

                let batches: Vec<RecordBatch> = results
                    .try_collect()
                    .await
                    .map_err(|e| format!("LanceDB collect: {e}"))?;

                let offset = query_index * count;
                let mut found = 0;
                for batch in &batches {
                    let id_col: Option<&UInt64Array> =
                        batch.column_by_name("id").and_then(|c| c.as_any().downcast_ref());
                    let dist_col: Option<&Float32Array> = batch
                        .column_by_name("_distance")
                        .and_then(|c| c.as_any().downcast_ref());
                    if let (Some(ids), Some(distances)) = (id_col, dist_col) {
                        for rank in 0..ids.len().min(count - found) {
                            out_keys[offset + found] = ids.value(rank) as Key;
                            out_distances[offset + found] = distances.value(rank);
                            found += 1;
                        }
                    }
                }
                for rank in found..count {
                    out_keys[offset + rank] = Key::MAX;
                    out_distances[offset + rank] = Distance::INFINITY;
                }
                out_counts[query_index] = found;
            }
            Ok::<(), String>(())
        })
    }

    fn memory_bytes(&self) -> usize {
        0
    }
}

// #region main

fn main() {
    let cli = Cli::parse();

    // Validate the metric string early.
    parse_lancedb_metric(&cli.metric).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio");
    let db = runtime.block_on(async { lancedb::connect(&cli.db_path).execute().await.expect("lancedb connect") });
    let _ = runtime.block_on(db.drop_table(TABLE_NAME, &[]));

    let mut state = BenchState::load(&cli.common).unwrap_or_else(|e| {
        eprintln!("Failed to load benchmark state: {e}");
        std::process::exit(1);
    });
    if cli.common.dimensions.len() > 1 {
        retrieval::bail("--dimensions sweep with >1 value isn't supported on LanceDB; rerun the binary per dimensions");
    }
    let dimensions = cli
        .common
        .dimensions
        .first()
        .copied()
        .unwrap_or_else(|| state.dimensions());
    state.check_dimensions(dimensions).unwrap_or_bail("invalid --dimensions");

    let mut backend = LanceDbBackend {
        db,
        table: None,
        dimensions,
        metric: cli.metric.clone(),
        runtime: runtime.handle().clone(),
        description: format!("lancedb · {} · {dimensions}d", cli.metric),
        metadata: {
            let mut metadata = std::collections::HashMap::new();
            metadata.insert("backend".into(), json!("lancedb"));
            metadata.insert("metric".into(), json!(&cli.metric));
            metadata.insert("dimensions".into(), json!(dimensions));
            metadata
        },
    };

    run(&mut backend, &mut state, dimensions).unwrap_or_else(|e| {
        eprintln!("Benchmark failed: {e}");
        std::process::exit(1);
    });
    eprintln!("Benchmark complete.");
}
