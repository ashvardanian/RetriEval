//! Tantivy BM25 benchmark binary over text entries.
//!
//! The index lives in RAM and scores with Tantivy's BM25, whose k1 1.2 and b 0.75 are fixed and
//! match Lucene's defaults. Each query is split by the field's own analyzer into `Should` terms,
//! since the query parser would read a sentence's punctuation as operators. Under `--fuzziness`,
//! each term also matches its edits through `FuzzyTermQuery`, which Tantivy scores as a constant
//! rather than blending BM25 over the expanded terms as Lucene does.
//!
//! ## Build & Install
//!
//! ```sh
//! cargo install --path . --no-default-features --features tantivy-backend
//! ```
//!
//! ## Examples
//!
//! ```sh
//! retri-eval-tantivy \
//!     --base-texts data/en-wiked.txt \
//!     --query-texts tasks/wiked-similar/queries.txt \
//!     --query-targets tasks/wiked-similar/targets.txt \
//!     --fuzziness 0,AUTO \
//!     --output results/
//! ```

use std::collections::HashMap;

use clap::Parser;
use forkunion as fu;
use serde_json::{json, Value};
use tantivy::{
    collector::TopDocs,
    columnar::Column,
    doc,
    merge_policy::NoMergePolicy,
    query::{BooleanQuery, FuzzyTermQuery, Occur, Query, TermQuery},
    schema::{Field, IndexRecordOption, Schema, FAST, TEXT},
    tokenizer::{TextAnalyzer, TokenStream},
    Index, IndexReader, IndexWriter, ReloadPolicy, Searcher, Term,
};

use retrieval::{
    run_config, spawn_pool, spell_list, spell_size, Backend, BatchKeys, BenchState, Distance, Fuzziness, Hits,
    IndexFile, StepRefineEntry, SweepSummary, TextArgs, TextCorpus, Texts, Threads, UnwrapOrBail,
};

#[derive(Parser, Debug)]
#[command(name = "retri-eval-tantivy", about = "Benchmark Tantivy BM25 over text entries")]
struct Cli {
    #[command(flatten)]
    text: TextArgs,

    /// Edits a query term may differ by: 0, 1, 2 or AUTO, as Elasticsearch spells them (comma-separated for sweep)
    #[arg(long, env = "RETRIEVAL_FUZZINESS", value_delimiter = ',', default_value = "0")]
    fuzziness: Vec<Fuzziness>,

    /// Search threads, 0 for all cores (comma-separated for sweep)
    #[arg(long, env = "RETRIEVAL_THREADS", value_delimiter = ',', default_value = "0", value_parser = retrieval::parse_threads_flag)]
    threads: Vec<Threads>,

    /// Memory the index writer buffers before it writes a segment, a budget `Index::writer` requires
    #[arg(long, env = "RETRIEVAL_WRITER_MEMORY", default_value = "1GB", value_parser = retrieval::parse_size_flag)]
    writer_memory: u64,
}

struct TantivyBackend {
    index: Index,
    writer: IndexWriter,
    reader: IndexReader,
    searcher: Searcher,
    /// Each segment's `row` column, in the searcher's segment order.
    rows: Vec<Column<u64>>,
    analyzer: TextAnalyzer,
    text: Field,
    row: Field,
    fuzziness: Fuzziness,
    pool: fu::ThreadPool,
    /// Each query's hits of the current batch, reused across batches.
    answers: Vec<Vec<Result<(usize, Distance), String>>>,
    metadata: HashMap<String, Value>,
}

fn failed(verb: &'static str) -> impl Fn(tantivy::TantivyError) -> String {
    move |error| format!("Tantivy {verb} failed: {error}")
}

impl TantivyBackend {
    fn new(fuzziness: Fuzziness, threads: Threads, writer_memory: u64) -> Result<Self, String> {
        let mut schema = Schema::builder();
        let text = schema.add_text_field("text", TEXT);
        let row = schema.add_u64_field("row", FAST);
        let index = Index::create_in_ram(schema.build());
        // Segments merge once, in `finish`, as Lucene's `forceMerge(1)` does after the inserts.
        let writer: IndexWriter = index.writer(writer_memory as usize).map_err(failed("writer"))?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .map_err(failed("reader"))?;
        let analyzer = index.tokenizer_for_field(text).map_err(failed("tokenizer"))?;
        let pool = spawn_pool(threads).map_err(|error| format!("ForkUnion pool of {threads} threads: {error}"))?;
        let metadata = HashMap::from([
            ("backend".into(), json!("tantivy")),
            ("library_version".into(), json!(tantivy::version_string())),
            ("metric".into(), json!("bm25")),
            ("fuzziness".into(), json!(fuzziness.to_string())),
            ("threads".into(), json!(threads.0.get())),
            ("writer_memory".into(), json!(writer_memory)),
        ]);
        Ok(Self {
            searcher: reader.searcher(),
            index,
            writer,
            reader,
            rows: Vec::new(),
            analyzer,
            text,
            row,
            fuzziness,
            pool,
            answers: Vec::new(),
            metadata,
        })
    }

    /// Opens a searcher over everything committed, with its segments' row columns.
    fn reload(&mut self) -> Result<(), String> {
        self.reader.reload().map_err(failed("reload"))?;
        self.searcher = self.reader.searcher();
        self.rows.clear();
        for segment in self.searcher.segment_readers() {
            self.rows
                .push(segment.fast_fields().u64("row").map_err(failed("row column"))?);
        }
        Ok(())
    }
}

/// One `Should` clause per analyzed term of `text`, each also matching its edits under `fuzziness`.
fn query_of(text: &str, field: Field, analyzer: &TextAnalyzer, fuzziness: Fuzziness) -> BooleanQuery {
    let mut analyzer = analyzer.clone();
    let mut tokens = analyzer.token_stream(text);
    let mut clauses: Vec<(Occur, Box<dyn Query>)> = Vec::new();
    while let Some(token) = tokens.next() {
        let term = Term::from_field_text(field, &token.text);
        let edits = fuzziness.edits(token.text.chars().count());
        if edits > 0 {
            clauses.push((Occur::Should, Box::new(FuzzyTermQuery::new(term.clone(), edits, true))));
        }
        clauses.push((
            Occur::Should,
            Box::new(TermQuery::new(term, IndexRecordOption::WithFreqs)),
        ));
    }
    BooleanQuery::new(clauses)
}

impl Backend for TantivyBackend {
    type Corpus = TextCorpus;

    fn metadata(&self) -> HashMap<String, Value> {
        self.metadata.clone()
    }

    fn add(&mut self, keys: BatchKeys<'_>, texts: Texts<'_>) -> Result<(), String> {
        for (row, text) in keys.rows().zip(texts.iter()) {
            self.writer
                .add_document(doc!(self.text => text, self.row => row as u64))
                .map_err(failed("add"))?;
        }
        Ok(())
    }

    fn search(&mut self, queries: Texts<'_>, mut hits: Hits<'_>) -> Result<(), String> {
        let count = hits.count();
        if self.answers.len() < queries.len() {
            self.answers.resize_with(queries.len(), Vec::new);
        }
        let Self {
            searcher,
            rows,
            analyzer,
            text,
            fuzziness,
            pool,
            answers,
            ..
        } = self;
        let (searcher, rows, analyzer, text, fuzziness) = (&*searcher, &*rows, &*analyzer, *text, *fuzziness);
        fu::for_each_task_mut_dynamic(pool, &mut answers[..queries.len()], |answer, query, _| {
            answer.clear();
            let query = query_of(queries.get(query), text, analyzer, fuzziness);
            match searcher.search(&query, &TopDocs::with_limit(count)) {
                Ok(found) => answer.extend(found.into_iter().map(|(score, address)| -> Result<_, String> {
                    let row = rows[address.segment_ord as usize].first(address.doc_id);
                    let row = row.ok_or_else(|| format!("Tantivy returned document {address:?} without a row"))?;
                    Ok((row as usize, -score))
                })),
                Err(error) => answer.push(Err(failed("search")(error))),
            }
        })
        .map_err(|error| format!("ForkUnion search fan-out failed: {error}"))?;
        for (row, answer) in hits.rows().zip(answers.iter_mut()) {
            row.write_rows(answer.drain(..))?;
        }
        Ok(())
    }

    fn memory_bytes(&self) -> usize {
        self.searcher
            .space_usage()
            .map_or(0, |usage| usage.total().get_bytes() as usize)
    }

    /// Commits the inserts and opens a searcher over them.
    fn flush(&mut self) -> Result<(), String> {
        self.writer.commit().map_err(failed("commit"))?;
        self.reload()
    }

    /// Merges every segment into one, as Lucene's `forceMerge(1)` does.
    fn finish(&mut self) -> Result<Option<StepRefineEntry>, String> {
        let segments = self.index.searchable_segment_ids().map_err(failed("segments"))?;
        if segments.len() > 1 {
            self.writer.merge(&segments).wait().map_err(failed("merge"))?;
            self.reload()?;
        }
        Ok(None)
    }
}

fn main() {
    let cli: Cli = retrieval::parse_cli();
    eprintln!("Tantivy {}", tantivy::version_string());

    let mut state = BenchState::load(&cli.text).unwrap_or_bail("benchmark state");
    eprintln!("- Fuzziness: {}", spell_list(&cli.fuzziness));
    eprintln!("- Threads: {}", spell_list(&cli.threads));
    eprintln!("- Writer memory: {}", spell_size(cli.writer_memory));

    let mut summary = SweepSummary::default();
    for &fuzziness in &cli.fuzziness {
        for &threads in &cli.threads {
            let description = format!("tantivy · fuzziness={fuzziness} · {threads} threads");
            let backend = TantivyBackend::new(fuzziness, threads, cli.writer_memory);
            summary.record(run_config(&description, IndexFile::None, backend, &mut state, ()));
        }
    }
    summary.finish();
}
