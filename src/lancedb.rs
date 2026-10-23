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

use arrow_array::{
    cast::AsArray,
    types::{UInt32Type, UInt64Type},
    Array, ArrayRef, FixedSizeBinaryArray, Float32Array, RecordBatch, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema};
use clap::Parser;
use futures_util::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use serde_json::json;

use retrieval::{
    run_config, Backend, BatchKeys, BenchState, CommonArgs, Hits, IndexFile, Key, SweepSummary, UnwrapOrBail, Vectors,
};

const TABLE_NAME: &str = "bench";

/// The `id` column for `keys`, at the narrowest Arrow type that holds them.
fn id_column(keys: BatchKeys<'_>) -> Result<(DataType, ArrayRef), String> {
    Ok(match keys.key_bytes() {
        4 => (
            DataType::UInt32,
            Arc::new(UInt32Array::from_iter_values(keys.keys().map(|key| key.0 as u32))),
        ),
        8 => (
            DataType::UInt64,
            Arc::new(UInt64Array::from_iter_values(keys.keys().map(|key| key.0 as u64))),
        ),
        _ => (
            DataType::FixedSizeBinary(16),
            Arc::new(
                FixedSizeBinaryArray::try_from_iter(keys.keys().map(|key| key.0.to_be_bytes()))
                    .map_err(|error| format!("LanceDB id array: {error}"))?,
            ),
        ),
    })
}

/// Appends the keys of an `id` column that [`id_column`] built.
fn read_keys(ids: &dyn Array, keys: &mut Vec<Key, std::alloc::System>) -> Result<(), String> {
    match ids.data_type() {
        DataType::UInt32 => keys.extend(
            ids.as_primitive::<UInt32Type>()
                .values()
                .iter()
                .map(|&id| Key::from(id)),
        ),
        DataType::UInt64 => keys.extend(
            ids.as_primitive::<UInt64Type>()
                .values()
                .iter()
                .map(|&id| Key::from(id)),
        ),
        DataType::FixedSizeBinary(16) => keys.extend(
            ids.as_fixed_size_binary()
                .iter()
                .flatten()
                .map(|bytes| Key(u128::from_be_bytes(bytes.try_into().unwrap()))),
        ),
        other => return Err(format!("LanceDB returned an id column of type {other}")),
    }
    Ok(())
}

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
    #[arg(long, env = "RETRIEVAL_METRIC", value_enum, default_value = "l2")]
    metric: Metric,

    /// Path for LanceDB storage
    #[arg(long, env = "RETRIEVAL_DB_PATH", default_value = "/tmp/retrieval-lancedb", value_parser = clap::builder::NonEmptyStringValueParser::new())]
    db_path: String,
}

// #region Backend

struct LanceDbBackend<'a> {
    scratch: Vec<f32, std::alloc::System>,
    database: &'a lancedb::Connection,
    table: Option<lancedb::Table>,
    dimensions: usize,
    metric: Metric,
    runtime: &'a tokio::runtime::Runtime,
    metadata: std::collections::HashMap<String, serde_json::Value>,
}

impl<'a> LanceDbBackend<'a> {
    /// Starts from an empty benchmark table, created on the first insert.
    fn new(
        database: &'a lancedb::Connection,
        runtime: &'a tokio::runtime::Runtime,
        dimensions: usize,
        metric: Metric,
    ) -> Result<Self, String> {
        let _ = runtime.block_on(database.drop_table(TABLE_NAME, &[]));
        Ok(Self {
            scratch: Vec::new_in(std::alloc::System),
            database,
            table: None,
            dimensions,
            metric,
            runtime,
            metadata: std::collections::HashMap::from([
                ("backend".into(), json!("lancedb")),
                ("metric".into(), json!(metric.to_string())),
                ("data_type".into(), json!("f32")),
                // No `create_index`: LanceDB graphs all sit behind a trained IVF layer, so search stays exhaustive.
                ("index_type".into(), json!("flat")),
            ]),
        })
    }

    fn batch(dimensions: usize, keys: BatchKeys<'_>, values: &[f32]) -> Result<RecordBatch, String> {
        let item = Arc::new(Field::new("item", DataType::Float32, true));
        let (id_type, ids) = id_column(keys)?;
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", id_type, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(item.clone(), dimensions as i32),
                false,
            ),
        ]));
        let vectors = arrow_array::FixedSizeListArray::try_new(
            item,
            dimensions as i32,
            Arc::new(Float32Array::from(values.to_vec())),
            None,
        )
        .map_err(|error| format!("LanceDB vector array: {error}"))?;
        RecordBatch::try_new(schema, vec![ids, Arc::new(vectors)]).map_err(|error| format!("LanceDB batch: {error}"))
    }
}

impl Backend for LanceDbBackend<'_> {
    fn metadata(&self) -> std::collections::HashMap<String, serde_json::Value> {
        self.metadata.clone()
    }

    fn add(&mut self, keys: BatchKeys<'_>, vectors: Vectors) -> Result<(), String> {
        let values = vectors.to_f32_in(&mut self.scratch)?;
        let batch = Self::batch(self.dimensions, keys, values)?;
        self.runtime.block_on(async {
            match &self.table {
                None => {
                    let table = self
                        .database
                        .create_table(TABLE_NAME, vec![batch])
                        .execute()
                        .await
                        .map_err(|error| format!("LanceDB create: {error}"))?;
                    self.table = Some(table);
                }
                Some(table) => {
                    table
                        .add(vec![batch])
                        .execute()
                        .await
                        .map_err(|error| format!("LanceDB add: {error}"))?;
                }
            }
            Ok(())
        })
    }

    fn search(&mut self, queries: Vectors, mut hits: Hits<'_>) -> Result<(), String> {
        let count = hits.count();
        let dimensions = queries.dimensions();
        let values = queries.to_f32_in(&mut self.scratch)?;
        let table = self.table.as_ref().ok_or("no table created")?;
        let metric = lancedb::DistanceType::from(self.metric);
        self.runtime.block_on(async {
            for (query, row) in values.chunks_exact(dimensions).zip(hits.rows()) {
                let mut results = table
                    .vector_search(query.to_vec())
                    .map_err(|error| format!("LanceDB query build: {error}"))?
                    .distance_type(metric)
                    .limit(count)
                    .execute()
                    .await
                    .map_err(|error| format!("LanceDB search: {error}"))?;
                let mut keys = Vec::with_capacity_in(count, std::alloc::System);
                let mut distances = Vec::with_capacity_in(count, std::alloc::System);
                while let Some(batch) = results
                    .try_next()
                    .await
                    .map_err(|error| format!("LanceDB stream: {error}"))?
                {
                    let ids = batch.column_by_name("id").ok_or("LanceDB returned no id column")?;
                    read_keys(ids.as_ref(), &mut keys)?;
                    let batch_distances = batch
                        .column_by_name("_distance")
                        .and_then(|column| column.as_any().downcast_ref::<Float32Array>())
                        .ok_or("LanceDB returned no distance column")?;
                    distances.extend_from_slice(batch_distances.values());
                }
                row.write(keys.into_iter().zip(distances).map(Ok))?;
            }
            Ok(())
        })
    }

    fn memory_bytes(&self) -> usize {
        0
    }
}

// #region main

fn main() {
    let cli: Cli = retrieval::parse_cli();
    let mut state = BenchState::load(&cli.common).unwrap_or_bail("benchmark state");
    eprintln!("- Metric: {}", cli.metric);
    eprintln!("- Database path: {}", cli.db_path);
    let dimensions_sweep = cli.common.dimensions_sweep(state.dimensions());

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap_or_bail("tokio runtime");
    let database = runtime
        .block_on(lancedb::connect(&cli.db_path).execute())
        .unwrap_or_bail("LanceDB connect");
    let mut summary = SweepSummary::default();
    for &dimensions in &dimensions_sweep {
        let description = format!("lancedb · {} · {dimensions}d", cli.metric);
        let backend = LanceDbBackend::new(&database, &runtime, dimensions, cli.metric);
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
