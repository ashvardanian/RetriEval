//! Qdrant benchmark binary.
//!
//! ```sh
//! cargo run --release --bin bench-qdrant --features qdrant-backend -- \
//!     --vectors datasets/wiki_1M/base.1M.fbin \
//!     --queries datasets/wiki_1M/query.public.100K.fbin \
//!     --neighbors datasets/wiki_1M/groundtruth.public.100K.ibin \
//!     --metric ip \
//!     --output wiki-1M-qdrant.jsonl
//! ```

use std::collections::HashMap;
use std::time::Duration;

use clap::Parser;
use qdrant_client::qdrant::{
    point_id, CreateCollectionBuilder, Distance as QdrantDistance, HnswConfigDiffBuilder,
    PointStruct, SearchPointsBuilder, UpsertPointsBuilder, VectorParamsBuilder,
};
use qdrant_client::Qdrant;
use usearch_bench::docker::ContainerHandle;
use usearch_bench::{run, Backend, BenchState, CommonArgs, Distance, Key, Vectors};

const COLLECTION: &str = "bench";
const BATCH_SIZE: usize = 10_000;

// #region Local metric mapping

fn parse_qdrant_distance(s: &str) -> Result<QdrantDistance, String> {
    match s {
        "ip" => Ok(QdrantDistance::Dot),
        "cos" => Ok(QdrantDistance::Cosine),
        "l2sq" | "l2" => Ok(QdrantDistance::Euclid),
        "hamming" => Ok(QdrantDistance::Manhattan),
        _ => Err(format!("unknown Qdrant distance metric: {s}")),
    }
}

// #region CLI

#[derive(Parser, Debug)]
#[command(name = "bench-qdrant", about = "Benchmark Qdrant")]
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

struct QdrantBackend {
    client: Qdrant,
    container: Option<ContainerHandle>,
    dimensions: usize,
    runtime: tokio::runtime::Handle,
    description: String,
}

impl Backend for QdrantBackend {
    fn description(&self) -> String {
        self.description.clone()
    }

    fn add(&mut self, _keys: &[Key], vectors: Vectors) -> Result<(), String> {
        let data = vectors.data.to_f32();
        let dimensions = vectors.dimensions;
        let num_vectors = data.len() / dimensions;

        self.runtime.block_on(async {
            for batch_start in (0..num_vectors).step_by(BATCH_SIZE) {
                let batch_end = (batch_start + BATCH_SIZE).min(num_vectors);
                let points: Vec<PointStruct> = (batch_start..batch_end)
                    .map(|i| {
                        let vec = data[i * dimensions..(i + 1) * dimensions].to_vec();
                        let empty: HashMap<String, qdrant_client::qdrant::Value> = HashMap::new();
                        PointStruct::new(i as u64, vec, empty)
                    })
                    .collect();
                self.client
                    .upsert_points(UpsertPointsBuilder::new(COLLECTION, points).wait(true))
                    .await
                    .map_err(|e| format!("Qdrant upsert failed: {e}"))?;
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

        self.runtime.block_on(async {
            for q in 0..num_vectors {
                let query = data[q * dimensions..(q + 1) * dimensions].to_vec();
                let response = self
                    .client
                    .search_points(SearchPointsBuilder::new(COLLECTION, query, count as u64))
                    .await
                    .map_err(|e| format!("Qdrant search failed: {e}"))?;

                let offset = q * count;
                let mut found = 0;
                for (j, point) in response.result.iter().enumerate().take(count) {
                    let id = match &point.id {
                        Some(id) => match &id.point_id_options {
                            Some(point_id::PointIdOptions::Num(n)) => *n as Key,
                            _ => Key::MAX,
                        },
                        None => Key::MAX,
                    };
                    out_keys[offset + j] = id;
                    out_distances[offset + j] = point.score;
                    if id != Key::MAX {
                        found += 1;
                    }
                }
                for j in response.result.len()..count {
                    out_keys[offset + j] = Key::MAX;
                    out_distances[offset + j] = Distance::INFINITY;
                }
                out_counts[q] = found;
            }
            Ok::<(), String>(())
        })
    }

    fn memory_bytes(&self) -> usize {
        0
    }
}

impl Drop for QdrantBackend {
    fn drop(&mut self) {
        if let Some(c) = self.container.take() {
            let _ = self.runtime.block_on(c.stop());
        }
    }
}

// #region main

fn main() {
    let cli = Cli::parse();
    let distance_metric = parse_qdrant_distance(&cli.metric).unwrap_or_else(|e| {
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
            "qdrant/qdrant:latest",
            "usearch-bench-qdrant",
            &vec![(6333, 6333), (6334, 6334)],
            &[],
            timeout,
        )
        .await
        .expect("docker start");
        handle
            .wait_for_http("http://localhost:6333/healthz", timeout)
            .await
            .expect("health");
        handle
    });

    let client = Qdrant::from_url("http://localhost:6334")
        .build()
        .expect("qdrant client");

    let mut state = BenchState::load(&cli.common).unwrap_or_else(|e| {
        eprintln!("Failed to load benchmark state: {e}");
        std::process::exit(1);
    });
    let dimensions = state.dimensions();

    runtime.block_on(async {
        let _ = client.delete_collection(COLLECTION).await;
        client
            .create_collection(
                CreateCollectionBuilder::new(COLLECTION)
                    .vectors_config(VectorParamsBuilder::new(dimensions as u64, distance_metric))
                    .hnsw_config(
                        HnswConfigDiffBuilder::default()
                            .m(cli.connectivity as u64)
                            .ef_construct(cli.expansion_add as u64),
                    ),
            )
            .await
            .expect("create collection");
    });

    let mut backend = QdrantBackend {
        client,
        container: Some(handle),
        dimensions,
        runtime: runtime.handle().clone(),
        description: format!(
            "qdrant · {} · M={} · ef={} · {dimensions}d",
            cli.metric, cli.connectivity, cli.expansion_add,
        ),
    };

    run(&mut backend, &mut state).unwrap_or_else(|e| {
        eprintln!("Benchmark failed: {e}");
        std::process::exit(1);
    });
    eprintln!("Benchmark complete.");
}
