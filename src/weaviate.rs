//! Weaviate benchmark binary.
//!
//! ```sh
//! cargo run --release --bin bench-weaviate --features weaviate-backend -- \
//!     --vectors datasets/wiki_1M/base.1M.fbin \
//!     --queries datasets/wiki_1M/query.public.100K.fbin \
//!     --neighbors datasets/wiki_1M/groundtruth.public.100K.ibin \
//!     --metric ip \
//!     --output wiki-1M-weaviate.jsonl
//! ```

use std::time::Duration;

use clap::Parser;
use usearch_bench::docker::ContainerHandle;
use usearch_bench::{run, Backend, BenchState, CommonArgs, Distance, Key, Vectors};
use weaviate_community::collections::objects::Object;
use weaviate_community::collections::query::RawQuery;
use weaviate_community::collections::schema::*;
use weaviate_community::WeaviateClient;

const CLASS_NAME: &str = "Bench";

// #region Local metric mapping

fn parse_weaviate_distance(s: &str) -> Result<DistanceMetric, String> {
    match s {
        "ip" => Ok(DistanceMetric::DOT),
        "cos" => Ok(DistanceMetric::COSINE),
        "l2sq" | "l2" => Ok(DistanceMetric::L2SQUARED),
        "hamming" => Ok(DistanceMetric::HAMMING),
        _ => Err(format!("unknown Weaviate distance metric: {s}")),
    }
}

// #region CLI

#[derive(Parser, Debug)]
#[command(name = "bench-weaviate", about = "Benchmark Weaviate")]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,

    #[arg(long, default_value = "l2")]
    metric: String,

    #[arg(long, default_value_t = 16)]
    connectivity: usize,

    #[arg(long, default_value_t = 128)]
    expansion_add: usize,

    #[arg(long, default_value_t = 64)]
    expansion_search: usize,

    #[arg(long, default_value_t = 120)]
    docker_timeout: u64,
}

// #region Backend

struct WeaviateBackend {
    client: WeaviateClient,
    container: Option<ContainerHandle>,
    runtime: tokio::runtime::Handle,
    description: String,
}

impl Backend for WeaviateBackend {
    fn description(&self) -> String {
        self.description.clone()
    }

    fn add(&mut self, _keys: &[Key], vectors: Vectors) -> Result<(), String> {
        let data = vectors.data.to_f32();
        let dimensions = vectors.dimensions;
        let num_vectors = data.len() / dimensions;

        self.runtime.block_on(async {
            for i in 0..num_vectors {
                let vec: Vec<f64> = data[i * dimensions..(i + 1) * dimensions]
                    .iter()
                    .map(|&x| x as f64)
                    .collect();
                let obj = Object::builder(CLASS_NAME, serde_json::json!({ "idx": i as i64 }))
                    .with_vector(vec)
                    .build();
                self.client
                    .objects
                    .create(&obj, None)
                    .await
                    .map_err(|e| format!("Weaviate insert failed: {e}"))?;
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
                let query: Vec<f64> = data[q * dimensions..(q + 1) * dimensions].iter().map(|&x| x as f64).collect();
                let gql = format!(
                    r#"{{ Get {{ {CLASS_NAME}(nearVector: {{ vector: {query:?} }} limit: {count}) {{ idx _additional {{ distance }} }} }} }}"#
                );
                let response = self.client.query.raw(RawQuery::new(&gql)).await
                    .map_err(|e| format!("Weaviate query failed: {e}"))?;

                let offset = q * count;
                let mut found = 0;
                if let Some(items) = response.pointer(&format!("/data/Get/{CLASS_NAME}"))
                    .and_then(|v| v.as_array()) {
                    for (j, item) in items.iter().enumerate().take(count) {
                        let idx = item.get("idx").and_then(|v| v.as_i64()).unwrap_or(-1);
                        let distance_value = item.pointer("/_additional/distance")
                            .and_then(|d| d.as_f64()).unwrap_or(f64::INFINITY) as f32;
                        out_keys[offset + j] = if idx >= 0 { idx as Key } else { Key::MAX };
                        out_distances[offset + j] = distance_value;
                        if idx >= 0 { found += 1; }
                    }
                }
                for j in found..count {
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

impl Drop for WeaviateBackend {
    fn drop(&mut self) {
        if let Some(c) = self.container.take() {
            let _ = self.runtime.block_on(c.stop());
        }
    }
}

// #region main

fn main() {
    let cli = Cli::parse();
    let distance_metric = parse_weaviate_distance(&cli.metric).unwrap_or_else(|e| {
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
            "semitechnologies/weaviate:latest",
            "usearch-bench-weaviate",
            &vec![(8080, 8080), (50051, 50051)],
            &[
                "QUERY_DEFAULTS_LIMIT=25".into(),
                "AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true".into(),
                "PERSISTENCE_DATA_PATH=/var/lib/weaviate".into(),
                "DEFAULT_VECTORIZER_MODULE=none".into(),
                "CLUSTER_HOSTNAME=node1".into(),
            ],
            timeout,
        )
        .await
        .expect("docker start");
        handle
            .wait_for_http("http://localhost:8080/v1/.well-known/ready", timeout)
            .await
            .expect("weaviate not ready");
        handle
    });

    let client = WeaviateClient::new("http://localhost:8080", None, None).expect("weaviate client");

    runtime.block_on(async {
        let _ = client.schema.delete(CLASS_NAME).await;
        let class = Class::builder(CLASS_NAME)
            .with_description("Benchmark vectors")
            .with_vectorizer("none")
            .with_vector_index_type(VectorIndexType::HNSW)
            .with_vector_index_config(
                VectorIndexConfig::builder()
                    .with_distance(distance_metric)
                    .with_ef(cli.expansion_search as i64)
                    .with_ef_construction(cli.expansion_add as u64)
                    .with_max_connections(cli.connectivity as u64)
                    .build(),
            )
            .with_properties(Properties::new(vec![
                Property::builder("idx", vec!["int"]).build()
            ]))
            .build();
        client
            .schema
            .create_class(&class)
            .await
            .expect("create class");
    });

    let mut state = BenchState::load(&cli.common).unwrap_or_else(|e| {
        eprintln!("Failed to load benchmark state: {e}");
        std::process::exit(1);
    });
    let dimensions = state.dimensions();

    let mut backend = WeaviateBackend {
        client,
        container: Some(handle),
        runtime: runtime.handle().clone(),
        description: format!(
            "weaviate · {} · M={} · ef={} · {dimensions}d",
            cli.metric, cli.connectivity, cli.expansion_add,
        ),
    };

    run(&mut backend, &mut state).unwrap_or_else(|e| {
        eprintln!("Benchmark failed: {e}");
        std::process::exit(1);
    });
    eprintln!("Benchmark complete.");
}
