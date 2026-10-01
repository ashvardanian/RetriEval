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
//!     --base-vectors datasets/wiki_1M/base.1M.fbin \
//!     --query-vectors datasets/wiki_1M/query.public.100K.fbin \
//!     --query-neighbors datasets/wiki_1M/groundtruth.public.100K.ibin \
//!     --metric ip \
//!     --output results/
//! ```

use std::{fmt, sync::Arc};

use arrow_array::{Float32Array, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use clap::Parser;
use futures_util::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use serde_json::json;

use retrieval::{run, Backend, BenchState, CommonArgs, Distance, Key, UnwrapOrBail, Vectors};

const TABLE_NAME: &str = "bench";

// #region Local metric mapping

/// LanceDB distances, as `--metric` spells them.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum Metric {
    Ip,
    Cos,
    #[value(alias = "l2sq")]
    L2,
}

impl From<Metric> for lancedb::DistanceType {
    fn from(metric: Metric) -> Self {
        match metric {
            Metric::Ip => Self::Dot,
            Metric::Cos => Self::Cosine,
            Metric::L2 => Self::L2,
        }
    }
}

impl fmt::Display for Metric {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        retrieval::spell_value(self, formatter)
    }
}

// #region CLI

#[derive(Parser, Debug)]
#[command(name = "retri-eval-lancedb", about = "Benchmark LanceDB")]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,

    /// Distance metric
    #[arg(long, value_enum, default_value = "l2")]
    metric: Metric,

    /// Path for LanceDB storage
    #[arg(long, default_value = "/tmp/retrieval-lancedb", value_parser = clap::builder::NonEmptyStringValueParser::new())]
    db_path: String,
}

// #region Backend

struct LanceDbBackend {
    scratch: Vec<f32, std::alloc::System>,
    db: lancedb::Connection,
    table: Option<lancedb::Table>,
    dimensions: usize,
    metric: Metric,
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

    fn make_batch(schema: Arc<Schema>, dimensions: usize, keys: &[Key], data: &[f32]) -> RecordBatch {
        let ids = UInt64Array::from(keys.iter().map(|&k| k as u64).collect::<Vec<_>>());
        let values = Float32Array::from(data.to_vec());
        let list_field = Arc::new(Field::new("item", DataType::Float32, true));
        let vectors = arrow_array::FixedSizeListArray::try_new(list_field, dimensions as i32, Arc::new(values), None)
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
        let schema = self.schema();
        let data = vectors.data.to_f32_in(&mut self.scratch)?;
        let batch = Self::make_batch(schema, self.dimensions, keys, data);

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
        &mut self,
        queries: Vectors,
        count: usize,
        out_keys: &mut [Key],
        out_distances: &mut [Distance],
        out_counts: &mut [usize],
    ) -> Result<(), String> {
        let data = queries.data.to_f32_in(&mut self.scratch)?;
        let dimensions = queries.dimensions;
        let num_vectors = data.len() / dimensions;
        let table = self.table.as_ref().ok_or("no table created")?;

        let lance_metric = lancedb::DistanceType::from(self.metric);

        self.runtime.block_on(async {
            for query_index in 0..num_vectors {
                let query = data[query_index * dimensions..(query_index + 1) * dimensions].to_vec();
                let mut results = table
                    .vector_search(query)
                    .map_err(|e| format!("LanceDB query build: {e}"))?
                    .distance_type(lance_metric)
                    .limit(count)
                    .execute()
                    .await
                    .map_err(|e| format!("LanceDB search: {e}"))?;

                let offset = query_index * count;
                out_keys[offset..offset + count].fill(Key::MAX);
                out_distances[offset..offset + count].fill(Distance::INFINITY);
                let mut found = 0;
                while let Some(batch) = results.try_next().await.map_err(|e| format!("LanceDB stream: {e}"))? {
                    let ids = batch
                        .column_by_name("id")
                        .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
                        .ok_or("LanceDB returned no id column")?;
                    let distances = batch
                        .column_by_name("_distance")
                        .and_then(|c| c.as_any().downcast_ref::<Float32Array>())
                        .ok_or("LanceDB returned no distance column")?;
                    for rank in 0..ids.len().min(count - found) {
                        out_keys[offset + found] =
                            Key::try_from(ids.value(rank)).map_err(|_| "LanceDB key overflow")?;
                        out_distances[offset + found] = distances.value(rank);
                        found += 1;
                    }
                    if found == count {
                        break;
                    }
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
    let cli: Cli = retrieval::parse_cli();

    if cli.common.index.is_some() {
        retrieval::bail("--index is not supported for this backend");
    }

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
    eprintln!("- Metric: {}", cli.metric);
    eprintln!("- Database path: {}", cli.db_path);
    if cli.common.dims.len() > 1 {
        retrieval::bail("--dims sweep with >1 value isn't supported on LanceDB; rerun the binary per dimensions");
    }
    let dimensions = cli.common.dims.first().copied().unwrap_or_else(|| state.dimensions());
    state.check_dimensions(dimensions).unwrap_or_bail("invalid --dims");

    let mut backend = LanceDbBackend {
        scratch: Vec::new_in(std::alloc::System),
        db,
        table: None,
        dimensions,
        metric: cli.metric,
        runtime: runtime.handle().clone(),
        description: format!("lancedb · {} · {dimensions}d", cli.metric),
        metadata: {
            let mut metadata = std::collections::HashMap::new();
            metadata.insert("backend".into(), json!("lancedb"));
            metadata.insert("metric".into(), json!(cli.metric.to_string()));
            metadata.insert("data_type".into(), json!("f32"));
            // No `create_index` call: every LanceDB graph index (`IvfHnswFlat` /
            // `IvfHnswSq` / `IvfHnswPq`) sits behind a k-means-trained IVF layer,
            // which the no-learned-codebook rule excludes. Exhaustive scan is
            // the compliant fallback, and its recall of 1.0 means that rather
            // than a perfect graph.
            metadata.insert("index_type".into(), json!("flat"));
            metadata
        },
    };

    run(&mut backend, &mut state, dimensions).unwrap_or_else(|e| {
        eprintln!("Benchmark failed: {e}");
        std::process::exit(1);
    });
    eprintln!("Benchmark complete.");
}
