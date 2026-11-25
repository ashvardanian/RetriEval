//! Elasticsearch BM25 benchmark binary over text entries.
//!
//! ## Prerequisites
//!
//! Requires Docker — the benchmark auto-manages an `elasticsearch` container on Lucene 9.12, with one
//! shard and no replicas.
//!
//! ## Build & Install
//!
//! ```sh
//! cargo install --path . --no-default-features --features elasticsearch-backend
//! ```
//!
//! ## Examples
//!
//! ```sh
//! retri-eval-elasticsearch \
//!     --base-texts data/en-wiked.txt \
//!     --query-texts tasks/wiked-similar/queries.txt \
//!     --query-targets tasks/wiked-similar/targets.txt \
//!     --fuzziness 0,AUTO \
//!     --output results/
//! ```

use std::{collections::HashMap, fmt::Write, time::Duration};

use clap::Parser;
use serde_json::{json, Value};

use retrieval::{
    docker::ContainerHandle, run_config, spell_duration, spell_list, Backend, BatchKeys, BenchState, Distance,
    Fuzziness, Hits, IndexFile, Port, StepRefineEntry, SweepSummary, TextArgs, TextCorpus, Texts, UnwrapOrBail,
};

const INDEX_NAME: &str = "bench";
const IMAGE: &str = "docker.elastic.co/elasticsearch/elasticsearch:8.17.0";

#[derive(Parser, Debug)]
#[command(
    name = "retri-eval-elasticsearch",
    about = "Benchmark Elasticsearch BM25 over text entries"
)]
struct Cli {
    #[command(flatten)]
    text: TextArgs,

    /// Edits a query term may differ by: 0, 1, 2 or AUTO (comma-separated for sweep)
    #[arg(long, env = "RETRIEVAL_FUZZINESS", value_delimiter = ',', default_value = "0")]
    fuzziness: Vec<Fuzziness>,

    /// Time limit for container start and readiness, like 120s
    #[arg(long, env = "RETRIEVAL_STARTUP_TIME_LIMIT", default_value = "120s", value_parser = retrieval::parse_duration_flag)]
    startup_time_limit: Duration,

    /// Elasticsearch HTTP port
    #[arg(long, env = "RETRIEVAL_PORT", default_value = "9200", value_parser = retrieval::parse_port)]
    port: Port,
}

/// One config's index on the shared server.
struct ElasticsearchBackend<'a> {
    fuzziness: Fuzziness,
    http: &'a reqwest::Client,
    http_base: &'a str,
    container: &'a ContainerHandle,
    runtime: &'a tokio::runtime::Runtime,
    metadata: HashMap<String, Value>,
}

impl<'a> ElasticsearchBackend<'a> {
    fn new(
        http: &'a reqwest::Client,
        http_base: &'a str,
        container: &'a ContainerHandle,
        runtime: &'a tokio::runtime::Runtime,
        fuzziness: Fuzziness,
    ) -> Result<Self, String> {
        runtime.block_on(create_index(http, http_base))?;
        let metadata = HashMap::from([
            ("backend".into(), json!("elasticsearch")),
            ("image".into(), json!(IMAGE)),
            ("metric".into(), json!("bm25")),
            ("fuzziness".into(), json!(fuzziness.to_string())),
            ("threads".into(), json!(1)),
        ]);
        Ok(Self {
            fuzziness,
            http,
            http_base,
            container,
            runtime,
            metadata,
        })
    }

    /// POSTs `body` to `path` under the index.
    fn post(&self, path: &str, body: String) -> Result<Value, String> {
        let url = format!("{}/{INDEX_NAME}/{path}", self.http_base);
        self.runtime.block_on(post_ndjson(self.http, &url, body))
    }
}

/// Drops and recreates the benchmark index: one shard, no replicas, refreshed only when the harness flushes.
async fn create_index(http: &reqwest::Client, http_base: &str) -> Result<(), String> {
    let url = format!("{http_base}/{INDEX_NAME}");
    let _ = http.delete(&url).send().await;
    let settings = json!({
        "settings": { "number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1" },
        "mappings": { "properties": { "text": { "type": "text" } } }
    });
    let response = http
        .put(&url)
        .json(&settings)
        .send()
        .await
        .map_err(|error| format!("PUT {url} failed: {error}"))?;
    let status = response.status();
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        return Err(format!("PUT {url} -> {status}: {text}"));
    }
    Ok(())
}

/// POSTs `body` as newline-delimited JSON and returns the parsed response, turning a non-2xx status into an error.
async fn post_ndjson(http: &reqwest::Client, url: &str, body: String) -> Result<Value, String> {
    let response = http
        .post(url)
        .header(reqwest::header::CONTENT_TYPE, "application/x-ndjson")
        .body(body)
        .send()
        .await
        .map_err(|error| format!("POST {url} failed: {error}"))?;
    let status = response.status();
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        return Err(format!("POST {url} -> {status}: {text}"));
    }
    response
        .json()
        .await
        .map_err(|error| format!("POST {url} returned invalid JSON: {error}"))
}

impl Backend for ElasticsearchBackend<'_> {
    type Corpus = TextCorpus;

    fn metadata(&self) -> HashMap<String, Value> {
        let mut metadata = self.metadata.clone();
        let limits = self.runtime.block_on(self.container.limits());
        let limits = limits.unwrap_or_else(|error| json!(format!("unread: {error}")));
        metadata.insert("container_limits".into(), limits);
        metadata
    }

    fn add(&mut self, keys: BatchKeys<'_>, texts: Texts<'_>) -> Result<(), String> {
        let mut body = String::new();
        for (row, text) in keys.rows().zip(texts.iter()) {
            writeln!(body, r#"{{"index":{{"_id":"{row}"}}}}"#).expect("a String takes any write");
            writeln!(body, "{}", json!({ "text": text })).expect("a String takes any write");
        }
        let response = self.post("_bulk", body)?;
        match response.get("errors").and_then(Value::as_bool) {
            Some(false) => Ok(()),
            _ => Err(format!("Elasticsearch bulk insert failed: {response}")),
        }
    }

    fn search(&mut self, queries: Texts<'_>, mut hits: Hits<'_>) -> Result<(), String> {
        const MAX_RESULT_WINDOW: usize = 10_000;
        let count = hits.count();
        if count > MAX_RESULT_WINDOW {
            return Err(format!(
                "Elasticsearch caps results at {MAX_RESULT_WINDOW} per query, below top-k {count}"
            ));
        }
        let mut body = String::new();
        for query in queries.iter() {
            let search = json!({
                "size": count,
                "_source": false,
                "query": { "match": { "text": { "query": query, "fuzziness": self.fuzziness.to_string() } } }
            });
            writeln!(body, "{{}}\n{search}").expect("a String takes any write");
        }
        let response = self.post("_msearch", body)?;
        let answers = response
            .get("responses")
            .and_then(Value::as_array)
            .ok_or("Elasticsearch msearch response has no responses")?;
        if answers.len() != queries.len() {
            return Err("Elasticsearch msearch response length mismatch".into());
        }
        for (row, answer) in hits.rows().zip(answers) {
            if let Some(error) = answer.get("error") {
                return Err(format!("Elasticsearch query failed: {error}"));
            }
            let found = answer
                .pointer("/hits/hits")
                .and_then(Value::as_array)
                .ok_or("Elasticsearch query response has no hits")?;
            row.write_rows(found.iter().map(|hit| {
                let row = hit
                    .get("_id")
                    .and_then(Value::as_str)
                    .and_then(|id| id.parse::<usize>().ok())
                    .ok_or("Elasticsearch returned a hit without a row id")?;
                let score = hit
                    .get("_score")
                    .and_then(Value::as_f64)
                    .ok_or("Elasticsearch returned a hit without a score")?;
                Ok((row, -score as Distance))
            }))?;
        }
        Ok(())
    }

    fn memory_bytes(&self) -> usize {
        self.runtime.block_on(self.container.memory_usage_bytes()) as usize
    }

    /// Refreshes the index, so everything added is searchable.
    fn flush(&mut self) -> Result<(), String> {
        self.post("_refresh", String::new()).map(drop)
    }

    /// Merges the index into one segment, as `forceMerge(1)` does, and refreshes it.
    fn finish(&mut self) -> Result<Option<StepRefineEntry>, String> {
        self.post("_forcemerge?max_num_segments=1", String::new())?;
        self.flush()?;
        Ok(None)
    }
}

fn main() {
    let cli: Cli = retrieval::parse_cli();
    let mut state = BenchState::load(&cli.text).unwrap_or_bail("benchmark state");
    eprintln!("- Fuzziness: {}", spell_list(&cli.fuzziness));
    eprintln!("- Startup time limit: {}", spell_duration(cli.startup_time_limit));
    eprintln!("- Port: {}", cli.port);

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap_or_bail("tokio runtime");
    let container = runtime
        .block_on(ContainerHandle::start(
            IMAGE,
            "retrieval-elasticsearch",
            &[(cli.port.into(), 9200)],
            &[
                "discovery.type=single-node".into(),
                "xpack.security.enabled=false".into(),
            ],
            &[],
        ))
        .unwrap_or_bail("Elasticsearch container");
    let http_base = format!("http://localhost:{}", cli.port);
    let http = reqwest::Client::new();
    let ready =
        runtime.block_on(container.wait_for_http(&format!("{http_base}/_cluster/health"), cli.startup_time_limit));

    let mut summary = SweepSummary::default();
    match ready {
        Ok(()) => {
            for &fuzziness in &cli.fuzziness {
                let description = format!("elasticsearch · fuzziness={fuzziness}");
                let backend = ElasticsearchBackend::new(&http, &http_base, &container, &runtime, fuzziness);
                summary.record(run_config(&description, IndexFile::None, backend, &mut state, ()));
            }
        }
        Err(error) => eprintln!("Elasticsearch did not come up: {error}"),
    }
    runtime.block_on(container.stop()).ok();
    summary.finish();
}
