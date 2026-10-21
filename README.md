![RetriEval benchmarks thumbnail](https://github.com/ashvardanian/ashvardanian/raw/master/repositories/RetriEval.jpg?raw=true) 

__RetriEval__ is a benchmarking suite designed for Billion-scale Vector Search workloads.
It's primarily used to benchmark in-process Search Engines on CPUs and GPUs, like [USearch](https://github.com/unum-cloud/usearch), [FAISS](https://github.com/facebookresearch/faiss), and [cuVS](https://github.com/NVIDIA/cuvs), but it also reuses similar profiling logic for standalone databases like [Qdrant](https://github.com/qdrant/qdrant), [Weaviate](https://github.com/weaviate/weaviate), and [Redis](https://github.com/redis/redis).
It works with the same plain input format standardized by the [BigANN benchmark](https://big-ann-benchmarks.com/), aiming for reproducible measurements – with shuffled parallel construction, incremental recall curves, normalized metrics, and machine-readable reports, capturing everything from machine topology to indexing hyper-parameters.

<table>
  <thead>
    <tr>
      <th align="left">Engine</th>
      <th align="left">Config</th>
      <th align="right">N</th>
      <th align="right">Recall @ 10</th>
      <th align="right">Add/s</th>
      <th align="right">Search/s</th>
      <th align="right">Memory</th>
      <th align="right">Duration</th>
    </tr>
  </thead>
  <tbody>
    <tr><th colspan="8" align="left">PubChem MACCS — 168-bit binary, Hamming · calibrated at 10M, same config at 100M</th></tr>
    <tr>
      <td rowspan="2">USearch</td>
      <td rowspan="2">M=32, ef=128/64</td>
      <td align="right">10M</td>
      <td align="right">0.9696</td><td align="right">36,347</td><td align="right">35,767</td>
      <td align="right">4.7 GB</td><td align="right">5.0m</td>
    </tr>
    <tr>
      <td align="right">100M</td>
      <td align="right">0.8438</td><td align="right">35,080</td><td align="right">38,432</td>
      <td align="right">40.8 GB</td><td align="right">54.6m</td>
    </tr>
    <tr>
      <td rowspan="2">FAISS</td>
      <td rowspan="2">M=64, ef=40/16</td>
      <td align="right">10M</td>
      <td align="right">0.9661</td><td align="right">95,230</td><td align="right">293,795</td>
      <td align="right">7.6 GB</td><td align="right">1.5m</td>
    </tr>
    <tr>
      <td align="right">100M</td>
      <td align="right">—</td><td align="right">—</td><td align="right">—</td>
      <td align="right">≥ 63 GB</td><td align="right">killed at 9h</td>
    </tr>
    <tr><th colspan="8" align="left">SIFT — 128D <code>u8</code>, L2 · iso-recall baseline at ≥ 99 % recall@10</th></tr>
    <tr>
      <td rowspan="2">USearch</td>
      <td rowspan="2">M=16, ef=128/256</td>
      <td align="right">10M</td>
      <td align="right">0.9938</td><td align="right">35,405</td><td align="right">80,729</td>
      <td align="right">4.4 GB</td><td align="right">4.8m</td>
    </tr>
    <tr>
      <td align="right">100M</td>
      <td align="right">0.9833</td><td align="right">39,831</td><td align="right">75,808</td>
      <td align="right">53.7 GB</td><td align="right">48.7m</td>
    </tr>
    <tr>
      <td rowspan="2">FAISS</td>
      <td rowspan="2">M=16, ef=128/256</td>
      <td align="right">10M</td>
      <td align="right">0.9952</td><td align="right">26,374</td><td align="right">38,278</td>
      <td align="right">5.9 GB</td><td align="right">5.6m</td>
    </tr>
    <tr>
      <td align="right">100M</td>
      <td align="right">—</td><td align="right">—</td><td align="right">—</td>
      <td align="right">≥ 46 GB</td><td align="right">killed at 9h</td>
    </tr>
    <tr>
      <td rowspan="2">cuVS ¹</td>
      <td>CAGRA 64/128, itopk=128</td>
      <td align="right">10M</td>
      <td align="right">0.9966 ²</td><td align="right">204,838</td><td align="right">1,204,000</td>
      <td align="right">—</td><td align="right">0.8m</td>
    </tr>
    <tr>
      <td>CAGRA 64/128, itopk=256</td>
      <td align="right">100M</td>
      <td align="right">0.9963 ²</td><td align="right">154,799</td><td align="right">582,000</td>
      <td align="right">—</td><td align="right">10.8m</td>
    </tr>
    <tr><th colspan="8" align="left">Microsoft Turing-ANNS — 100D <code>f32</code>, L2 · iso-recall baseline at ≥ 99 % recall@10</th></tr>
    <tr>
      <td rowspan="11">USearch</td>
      <td rowspan="2">M=48, ef=768/384, <code>f32</code></td>
      <td align="right">10M</td>
      <td align="right">0.9929</td><td align="right">8,532</td><td align="right">12,331</td>
      <td align="right">13.0 GB</td><td align="right">18.3m</td>
    </tr>
    <tr>
      <td align="right">100M</td>
      <td align="right">0.9929</td><td align="right">6,646</td><td align="right">10,398</td>
      <td align="right">139.6 GB</td><td align="right">4h 1m</td>
    </tr>
    <tr>
      <td rowspan="2">M=48, ef=768/384, <code>bf16</code></td>
      <td align="right">10M</td>
      <td align="right">0.9929</td><td align="right">10,496</td><td align="right">16,940</td>
      <td align="right">10.9 GB</td><td align="right">14.1m</td>
    </tr>
    <tr>
      <td align="right">100M</td>
      <td align="right">0.9931</td><td align="right">8,564</td><td align="right">14,772</td>
      <td align="right">105.2 GB</td><td align="right">3h 1m</td>
    </tr>
    <tr>
      <td rowspan="2">M=48, ef=768/384, <code>f16</code></td>
      <td align="right">10M</td>
      <td align="right">0.9929</td><td align="right">10,969</td><td align="right">20,246</td>
      <td align="right">10.9 GB</td><td align="right">13.5m</td>
    </tr>
    <tr>
      <td align="right">100M</td>
      <td align="right">0.9930</td><td align="right">8,807</td><td align="right">15,412</td>
      <td align="right">105.2 GB</td><td align="right">2h 54m</td>
    </tr>
    <tr>
      <td rowspan="2">M=48, ef=768/384, <code>e5m2</code></td>
      <td align="right">10M</td>
      <td align="right">0.9919</td><td align="right">10,526</td><td align="right">20,534</td>
      <td align="right">9.8 GB</td><td align="right">13.5m</td>
    </tr>
    <tr>
      <td align="right">100M</td>
      <td align="right">0.9924</td><td align="right">7,368</td><td align="right">13,227</td>
      <td align="right">88.0 GB</td><td align="right">3h 15m</td>
    </tr>
    <tr>
      <td>M=48, ef=768/384, <code>e4m3</code></td>
      <td align="right">10M</td>
      <td align="right">0.9930</td><td align="right">7,353</td><td align="right">12,106</td>
      <td align="right">9.8 GB</td><td align="right">19.4m</td>
    </tr>
    <tr>
      <td>M=48, ef=768/384, <code>e3m2</code></td>
      <td align="right">10M</td>
      <td align="right">0.9728</td><td align="right">10,398</td><td align="right">18,022</td>
      <td align="right">9.8 GB</td><td align="right">13.3m</td>
    </tr>
    <tr>
      <td>M=48, ef=768/384, <code>e2m3</code></td>
      <td align="right">10M</td>
      <td align="right">0.7941</td><td align="right">10,935</td><td align="right">21,313</td>
      <td align="right">9.8 GB</td><td align="right">13.2m</td>
    </tr>
    <tr>
      <td rowspan="3">FAISS</td>
      <td>M=48, ef=768/384, <code>f32</code></td>
      <td align="right">10M</td>
      <td align="right">0.9944</td><td align="right">7,491</td><td align="right">16,486</td>
      <td align="right">14.1 GB</td><td align="right">20.6m</td>
    </tr>
    <tr>
      <td>M=48, ef=768/384, <code>bf16</code></td>
      <td align="right">10M</td>
      <td align="right">0.9944</td><td align="right">3,800</td><td align="right">10,391</td>
      <td align="right">12.1 GB</td><td align="right">39.4m</td>
    </tr>
    <tr>
      <td>M=48, ef=768/384, <code>f16</code></td>
      <td align="right">10M</td>
      <td align="right">0.9944</td><td align="right">2,545</td><td align="right">10,032</td>
      <td align="right">12.1 GB</td><td align="right">1h 1m</td>
    </tr>
  </tbody>
</table>

> Benchmarks were conducted on dual socket Intel Xeon6 with 192 logical threads.
> USearch v2.25 was compared to FAISS v1.12.0 (static, via faiss-sys 0.7.0).
> Both engines used the native input quantization type — no rescaling in either.
>
> ¹ cuVS 26.10 ran on one NVIDIA B300 SXM6 through NVIDIA's `cuvs-bench` harness, not `retri-eval-cuvs`.
> CAGRA used graph degree 64 and intermediate degree 128 with NN-descent, `search_width` 4, and one thread issuing 10K-query batches with the queries already in device memory.
> Its build also kept 30-85 of the host's 256 logical threads busy.
> ² K-recall@10, `intersection_at_k`, which is stricter than the 1-recall@10 of the other rows; the best of the sweep was 0.9999 at 10M and 0.9993 at 100M.

The recommended methodology is to parameter-sweep different configuration options to achieve comparable recall between search backends on a given dataset.
Once the behavior is confirmed on a small 1M–10M subset, 100M–1B and larger benchmarks can be run to validate scaling curves.

Each search reports two recall conventions, because they answer different questions and one of them saturates.
`recall_at_k` is __1-recall@K__ (hit-rate@K): the fraction of queries whose single true nearest neighbor lands within the top-K — FAISS's `OneRecallAtRCriterion`, and the measure USearch reports.
`intersection_at_k` is __K-recall@K__ (`|top-K ∩ ground-truth-K| / K`), order-independent, and what ann-benchmarks and cuVS's own harness publish as recall.
1-recall@K ≥ K-recall@K for the same run, so the two are never interchangeable.

K is `--top-k`, which defaults to the ground-truth file's width — 10 on the Wiki sets, 100 on the BigANN ones.
That matters because 1-recall@K stops discriminating as K grows: at K=100 nearly any sane configuration finds the single true neighbor somewhere in the list.
`intersection_at_k` is what stays informative there, and `ndcg_at_k` likewise normalizes over the full truth prefix, so both are stricter at wide K than the same-named numbers at K=10.
Pass `--top-k 10` for figures comparable across datasets of differing ground-truth width.

## Quick Start

Install the default `retri-eval-usearch` binary:

```sh
cargo install --path .
```

Fetch the Unum Wiki 1M dataset — ~1.1 GB of vectors, queries, and ground truth — into `data/wiki-1m/`:

```sh
scripts/download.sh wiki-1m
```

Run a sweep over three quantizations and write JSON reports under `results/`:

```sh
retri-eval-usearch \
    --base-vectors data/wiki-1m/base.1M.fbin \
    --query-vectors data/wiki-1m/query.public.100K.fbin \
    --query-neighbors data/wiki-1m/groundtruth.public.100K.ibin \
    --data-type f32,f16,i8 \
    --metric ip \
    --output results/
```

Generate plots from the results:

```sh
uv run scripts/plot.py results/ --output-dir plots/
```

## Backends

### Search Engines

| Backend     | Parallelism | Quantization                                            | Metrics                   |
| ----------- | ----------- | ------------------------------------------------------- | ------------------------- |
| __USearch__ | ForkUnion   | f64, f32, bf16, f16, e5m2, e4m3, e3m2, e2m3, i8, u8, b1 | ip, l2, cos, hamming, ... |
| __FAISS__   | OpenMP      | f32, f16, bf16, u8, i8, b1                              | ip, l2                    |
| __cuVS__    | CUDA        | f32, f16, i8, u8                                        | l2, ip, cos               |

- __USearch__: Input is passed directly in the specified type.
  `--data-type` selects both the input interpretation and the internal quantization.
- __FAISS__: Input is always f32.
  `--data-type` selects the internal scalar quantizer (SQfp16, SQbf16, SQ8_direct, etc.).
- __cuVS__: Input is converted on the host to the `--data-type`, one of f32, f16, i8 or u8, before CAGRA builds on it.
  It reuses the shared `--connectivity`, `--expansion-add` and `--expansion-search` names and JSON keys, so its runs plot on the same axes as the HNSW backends.
  The mapping is an analogy, not an identity: `--expansion-add` is CAGRA's pre-prune candidate width rather than `ef_construction`, and `--expansion-search` its internal top-i list rather than `ef_search`.

```sh
retri-eval-usearch --data-type bf16 --metric l2 ...
retri-eval-faiss --data-type f16 --metric l2 ...
retri-eval-cuvs --metric l2 ...
```

### Vector Databases

Server-side quantization is managed by the database engine, not the benchmark.
Binary quantization is deterministic `sign(x)` per dim, and scalar quantization is per-dim min/max — neither trains a codebook, so both stay inside the "no learned logic" constraint the rest of the benchmark holds for the native backends.
Product quantization is deliberately excluded everywhere.

| Backend       | Client                       | Docker Image                       | Metrics                | Wire dtype sweep                        | Server-side quantization   |
| ------------- | ---------------------------- | ---------------------------------- | ---------------------- | --------------------------------------- | -------------------------- |
| __Qdrant__    | `qdrant-client`, gRPC        | `qdrant/qdrant:v1.19.1`            | ip, l2, cos, manhattan | `f32`, `f16`, `u8`                      | `none`, `binary`, `scalar` |
| __Redis__     | `redis`, RESP                | `redis:8.10`                       | ip, l2, cos            | `f32`, `f64`, `f16`, `bf16`, `u8`, `i8` | —                          |
| __Weaviate__  | `reqwest`, REST              | `semitechnologies/weaviate:1.39.7` | ip, l2, cos            | `f32` only                              | `none`, `binary`           |
| __LanceDB__   | `lancedb`, in-process, Arrow | —                                  | ip, l2, cos            | `f32` only                              | — (exact scan) ¹           |
| __Turso__     | `turso = 0.8.1`, native Rust | —                                  | l2, cos                | `f32` only                              | — (exact scan)             |
| __SurrealDB__ | `reqwest`, HTTP SQL          | `surrealdb/surrealdb:v3.3.0`       | l2, cos                | `f32` only                              | — (HNSW)                   |

¹ LanceDB's Rust client — `lancedb 0.37` — exposes graph-based search only via `IvfHnswFlat` / `IvfHnswSq` / `IvfHnswPq`, all behind a k-means-trained IVF layer that cannot be disabled.
No pure-HNSW variant is offered, so this benchmark leaves LanceDB on plain `f32` + L2/IP/Cos until upstream adds one.
Hamming is only available on `IvfFlat`, outside our graph path.

Turso runs in-process with exact `f32` scans over an in-memory database, or a `--db-path` file whose benchmark table alone is recreated; it measures exact search, not a libSQL ANN index.
The server backends run in Docker containers the binaries manage themselves; [`docker/README.md`](docker/README.md) covers their lifecycle and per-server caveats.

---

Each backend is behind its own feature flag.
Build only what you need:

```sh
cargo build --release --features usearch-backend    # USearch
cargo build --release --features faiss-backend      # FAISS
cargo build --release --features qdrant-backend     # Qdrant
cargo build --release --features redis-backend      # Redis
cargo build --release --features lancedb-backend    # LanceDB
cargo build --release --features weaviate-backend   # Weaviate
cargo build --release --features cuvs-backend       # cuVS
cargo build --release --no-default-features --features turso-backend # Turso
cargo build --release --no-default-features --features surrealdb-backend # SurrealDB
```

Or combine multiple:

```sh
cargo build --release --features usearch-backend,faiss-backend,qdrant-backend
```

## CLI Reference

Each backend is a separate binary.
The table lists every flag of every binary, with its default.
Flags marked "sweep" take comma-separated values and run one configuration per combination.
A bad value prints `--flag="value" does not parse, expected …` and exits with status 1.
Every `retri-eval-*` flag also reads `RETRIEVAL_<FLAG>` as its default, upper-cased with dashes as underscores, so `--top-k` reads `RETRIEVAL_TOP_K`.
The flag on the command line overrides the variable, and switches like `--self-search` read `1`, `0`, `true` or `false`.

| Flag                     | Binaries                                    | Default                  | Meaning                                                                                              |
| ------------------------ | ------------------------------------------- | ------------------------ | ---------------------------------------------------------------------------------------------------- |
| `--base-vectors`         | all `retri-eval-*`                          | required                 | Base vectors to index (`.fbin`, `.u8bin`, `.i8bin`, `.b1bin`), a path or glob                        |
| `--base-keys`            | all `retri-eval-*`                          | sequential               | Keys for the base vectors (`.i32bin`), one per vector                                                |
| `--query-vectors`        | all `retri-eval-*`                          | required                 | Query vectors to search with                                                                         |
| `--query-neighbors`      | all `retri-eval-*`                          | required                 | Ground-truth neighbors for those queries (`.ibin`)                                                   |
| `--top-k`                | all `retri-eval-*`                          | ground-truth width       | Neighbors per query, the k every metric is taken at; may not exceed the ground-truth width           |
| `--insertion-order`      | all `retri-eval-*`                          | `shuffled`               | `shuffled` by `--seed`, or the file's `original` order                                               |
| `--seed`                 | all `retri-eval-*`                          | `42`                     | 32-bit seed of the shuffled insertion order, or `random`; rejected with `--insertion-order original` |
| `--steps`                | all `retri-eval-*`                          | `10`                     | Measurement steps; the base is inserted in this many equal parts                                     |
| `--inserts-per-core`     | all `retri-eval-*`                          | `2048`                   | Vectors per `add` call for each backend thread: `--threads` on usearch and faiss, one elsewhere      |
| `--queries-per-core`     | all `retri-eval-*`                          | `2048`                   | Queries per `search` call for each backend thread: `--threads` on usearch and faiss, one elsewhere   |
| `--output`               | all `retri-eval-*`                          | none                     | Directory for JSON result files; progress only when unset                                            |
| `--max-base-vectors`     | all `retri-eval-*`                          | all                      | Cap on the base vectors used, for calibration on a slice                                             |
| `--index`                | usearch, faiss, cuvs                        | none                     | Persisted index: loaded and searched if the path exists, else built and saved; needs a single config |
| `--dims`                 | all `retri-eval-*`                          | native                   | Matryoshka truncations, sweep; each at most the native width, and a multiple of 8 for `.b1bin`       |
| `--self-search`          | all `retri-eval-*`                          | off                      | After the last insertion, replay indexed vectors as their own queries                                |
| `--self-search-top-k`    | all `retri-eval-*`                          | `10`                     | Neighbors per self-search query                                                                      |
| `--self-search-sample`   | all `retri-eval-*`                          | all                      | Leading base rows to replay: a count like `50000` or a fraction like `0.1`; needs `--self-search`    |
| `--data-type`            | usearch                                     | `bf16`                   | Sweep of `f64`, `f32`, `bf16`, `f16`, `e5m2`, `e4m3`, `e3m2`, `e2m3`, `i8`, `u8`, `b1`               |
| `--data-type`            | faiss                                       | `bf16`                   | Sweep of `f32`, `f16`, `bf16`, `u8`, `i8`, `b1`                                                      |
| `--data-type`            | cuvs                                        | `f32`                    | Sweep of `f32`, `f16`, `i8`, `u8`                                                                    |
| `--data-type`            | qdrant                                      | `f32`                    | Sweep of storage types `f32`, `f16`, `u8`                                                            |
| `--data-type`            | redis                                       | `f32`                    | Sweep of `f32`, `f64`, `f16`, `bf16`, `u8`, `i8`; `u8` and `i8` need Redis 8                         |
| `--metric`               | usearch                                     | `l2`                     | Sweep of `ip`, `l2`, `cos`, `hamming`, `jaccard`, `sorensen`, `pearson`, `haversine`, `divergence`   |
| `--metric`               | faiss                                       | `l2`                     | Sweep of `ip`, `l2`; `b1` data always uses Hamming                                                   |
| `--metric`               | cuvs, redis, weaviate                       | `l2`                     | Sweep of `l2`, `ip`, `cos`                                                                           |
| `--metric`               | qdrant                                      | `l2`                     | Sweep of `ip`, `cos`, `l2`, `manhattan`                                                              |
| `--metric`               | lancedb                                     | `l2`                     | One of `ip`, `cos`, `l2`                                                                             |
| `--metric`               | turso                                       | `l2`                     | One of `l2`, `cos`                                                                                   |
| `--metric`               | surrealdb                                   | `l2`                     | Sweep of `l2`, `cos`                                                                                 |
| `--quantization`         | qdrant                                      | `none`                   | Sweep of server-side `none`, `binary`, `scalar`                                                      |
| `--quantization`         | weaviate                                    | `none`                   | Sweep of server-side `none`, `binary`                                                                |
| `--connectivity`         | usearch, qdrant, redis, weaviate, surrealdb | `16`                     | HNSW M, sweep                                                                                        |
| `--connectivity`         | faiss, cuvs                                 | `32`                     | HNSW M, or CAGRA's output graph degree, sweep                                                        |
| `--expansion-add`        | all but cuvs, lancedb, turso                | `128`                    | HNSW construction width, sweep                                                                       |
| `--expansion-add`        | cuvs                                        | `64`                     | CAGRA's intermediate graph degree before pruning, sweep                                              |
| `--expansion-search`     | all but lancedb, turso                      | `64`                     | HNSW search width, or CAGRA's internal top-i list, sweep                                             |
| `--shards`               | usearch                                     | `1`                      | Index shards, sweep                                                                                  |
| `--threads`              | usearch                                     | `0`                      | Threads, `0` for all cores, sweep                                                                    |
| `--threads`              | faiss                                       | `0`                      | OpenMP threads, `0` for all cores                                                                    |
| `--search-width`         | cuvs                                        | `auto`                   | Graph nodes that start each search iteration, or `auto` for cuVS's choice, sweep                     |
| `--min-iterations`       | cuvs                                        | cuVS's choice            | Minimum search iterations                                                                            |
| `--max-iterations`       | cuvs                                        | cuVS's choice            | Maximum search iterations                                                                            |
| `--num-random-samplings` | cuvs                                        | cuVS's choice            | Random sampling rounds for the initial search points                                                 |
| `--build-algo`           | cuvs                                        | `auto`                   | Graph build algorithm, `auto` or `nn_descent`                                                        |
| `--startup-time-limit`   | qdrant, redis, weaviate, surrealdb          | `120s`                   | Time limit for container start and readiness, like `120s`                                            |
| `--grpc-port`            | qdrant                                      | `6334`                   | Host gRPC port                                                                                       |
| `--http-port`            | qdrant                                      | `6333`                   | Host HTTP port                                                                                       |
| `--port`                 | redis                                       | `6379`                   | Host port                                                                                            |
| `--port`                 | weaviate                                    | `8080`                   | Host HTTP port                                                                                       |
| `--port`                 | surrealdb                                   | `8000`                   | Host HTTP port                                                                                       |
| `--vectors-per-upsert`   | qdrant                                      | `10000`                  | Vectors per upsert request                                                                           |
| `--vectors-per-upsert`   | redis                                       | `1000`                   | Vectors per pipeline flush                                                                           |
| `--db-path`              | lancedb                                     | `/tmp/retrieval-lancedb` | LanceDB storage directory                                                                            |
| `--db-path`              | turso                                       | `:memory:`               | Turso database file, or `:memory:`; only its `bench` table is recreated                              |
| `--format`               | retri-generate                              | required                 | `b1bin` for clustered binary with Hamming ground truth, or `fbin` for Gaussian `f32` with L2         |
| `--base-count`           | retri-generate                              | required                 | Base vectors to generate                                                                             |
| `--query-count`          | retri-generate                              | required                 | Query vectors to generate                                                                            |
| `--query-count`          | retri-download-*                            | `10000`                  | Queries sampled from the base                                                                        |
| `--dims`                 | retri-generate                              | required                 | Bits for `b1bin`, a multiple of 8, or scalars for `fbin`                                             |
| `--clusters`             | retri-generate                              | `256`                    | Cluster centers                                                                                      |
| `--noise`                | retri-generate                              | `0.1`                    | Bit-flip probability from 0 to 1                                                                     |
| `--top-k`                | retri-generate, retri-download-*            | `10`                     | Neighbors per query in the ground-truth file                                                         |
| `--ground-truth-batch`   | retri-generate, retri-download-*            | from free RAM            | Queries per ground-truth batch                                                                       |
| `--threads`              | retri-generate, retri-download-*            | `0`                      | Ground-truth threads, `0` for all cores                                                              |
| `--seed`                 | retri-generate, retri-download-*            | `42`                     | 32-bit seed of generation or query sampling, or `random`                                             |
| `--output`               | retri-generate, retri-download-*            | required                 | Output directory                                                                                     |
| `--source`               | retri-download-molecules                    | required                 | `pubchem` (115M), `gdb13` (977M) or `enamine` (6.04B)                                                |
| `--fingerprint`          | retri-download-molecules                    | required                 | `maccs` (166 bits), `pubchem` (881), `ecfp4` (2048) or `fcfp4` (2048)                                |
| `--url-prefix`           | retri-download-molecules                    | the public bucket        | S3 or HTTPS prefix of the shards                                                                     |
| `--language`             | retri-download-cohere                       | `en`                     | Language config of the Hugging Face dataset                                                          |
| `--no-text`              | retri-download-cohere                       | off                      | Skip the aligned `titles.txt`, `texts.txt` and `urls.txt`                                            |
| `--limit`                | retri-download-*                            | all                      | Rows to extract                                                                                      |
| `--download-concurrency` | retri-download-*                            | `4`                      | Concurrent shard downloads                                                                           |
| `--keep-parquet`         | retri-download-*                            | off                      | Keep the Parquet shards under `<output>/parquet/`                                                    |

### Self-Recall

`--self-search` answers a different question from `--query-neighbors`, and closes two gaps the ground-truth path leaves open.

Shipped query files are small — SIFT has 10K queries, a single batch finishing in under a tenth of a second on a modern GPU.
A QPS figure measured over that burst is real, but says nothing about sustained throughput.
Self-recall replays indexed vectors as queries, so an unsampled 100M-vector index issues 100M queries, making it the only phase in a report measured under load.
Use `--self-search-sample` to cap the sweep (e.g. `0.1` for a tenth, or `50000` for an absolute count) when a full pass is too slow, as it is on the server backends.

It is also the metric that survives `--max-base-vectors` most cleanly.
Ground-truth files name neighbors drawn from the whole published base, so capping a run to a slice drags recall down by roughly the cap ratio no matter how good the index is — a quarter-size slice of the SIFT base reports about 0.25, not 0.99.
The harness does not divide that back out: the correction would assume the indexed slice is a uniform sample of the base, and a leading-prefix cap is not one.
It reports the raw figure and, in `dataset.vectors_count`, the divisor — coverage is `steps[].vectors_indexed / dataset.vectors_count`, plotted as `coverage.png`.
Self-recall needs no such correction at all: identity truth is computed against exactly the vectors that were inserted.

Three caveats are worth knowing before quoting the numbers.
Exact-duplicate vectors in the base can cost a hit, since either twin may take rank 1, so a self-recall below 1.0 bounds the index's error rate rather than measuring it exactly.
A self-query is a graph node rather than an out-of-sample point, so traversal converges faster than it would on real queries — treat the throughput as an optimistic bound, not as a substitute for a query-set measurement.
And `--self-search-sample` replays the _leading_ N rows rather than a random draw, because contiguous slices are what keep the queries zero-copy and the throughput figure clean.
On a base whose row order carries structure, only the default full sweep is unbiased.

Note also that a self-search's `recall_at_k` is very nearly redundant with its `recall_at_1`: a vector's own distance to itself is zero, which is minimal, so if the index retrieves it at all it lands at rank 1.
Raising `--self-search-top-k` therefore changes how much search work each query does — which is the point when you are measuring throughput — but not the recall value.

## Observability

Building with `--features perf-counters` adds per-phase hardware counters on Linux, and an external `perf stat` sidecar works on any build.
[`src/README.md`](src/README.md) covers both, along with how each backend measures `memory_bytes`, which differs from one backend to the next.

## Output Format

One JSON file per backend configuration, written to `--output <dir>`.
Files are auto-named `<backend>-<hash>.json`.

```json
{
  "machine": { "cpu_model": "Intel Xeon 6776P", "physical_cores": 96, ... },
  "dataset": { "base_vectors_path": "...", "vectors_count": 10000000, "dimensions": 100, ... },
  "config": { "backend": "usearch", "data_type": "f32", "metric": "l2", "connectivity": 16,
              "dimensions": 100, "vectors_count": 10000000, "top_k": 10, ... },
  "steps": [
    {
      "vectors_indexed": 1000000,
      "memory_bytes": 412000000,
      "add": { "elapsed": 12.3, "throughput": 81300 },
      "ground_truth_search": {
        "queries": 100000, "top_k": 10,
        "elapsed": 0.45, "throughput": 222000,
        "recall_at_1": 0.0942, "recall_at_k": 0.2815,
        "intersection_at_k": 0.2604, "ndcg_at_k": 0.1847
      },
      "self_search": {
        "queries": 1000000, "top_k": 10,
        "elapsed": 1.6, "throughput": 627541,
        "recall_at_1": 0.9958, "recall_at_k": 0.9961
      }
    }
  ]
}
```

A step pairs the index state (`vectors_indexed`, `memory_bytes`) with the phases that acted on it.
`add` is `null` on the `--index` load path, where nothing was inserted; `self_search` appears on at most one step — the last — because it runs once against the finished index, and that step is the index state it ran against.
Its metrics drop the `self_` prefix because the container already says it, and it carries no `intersection_at_k` or `ndcg_at_k`: identity truth is a single key per query, so neither a set overlap nor a ranked gain says anything beyond `recall_at_1`.

`dataset` describes the input files and `config` describes the run, so `dataset.vectors_count` is the base file's row count while `config.vectors_count` is what `--max-base-vectors` actually indexed; divide `steps[].vectors_indexed` by the former for the share of the ground truth a step could possibly have found.
`dataset.dimensions` is likewise the file's width, and `config.dimensions` the width a `--dims` sweep truncated to.

## Datasets

The BigANN collections are a good starting point, as most ship pre-computed ground-truth neighbors for recall evaluation.
The tables below group the datasets by scale, and each name links to its download and run recipe in [`data/README.md`](data/README.md), which also covers the file format and sharded inputs.

### ~1M Scale — Development & Testing

| Dataset                         | Scalar Type | Dimensions | Metric | Base Size | Ground Truth      |
| :------------------------------ | ----------: | ---------: | -----: | --------: | :---------------- |
| [Unum UForm Wiki][unum-wiki-1m] |       `f32` |        256 |     IP |      1 GB | 100K queries, yes |
| [Arxiv with E5][unum-arxiv-2m]  |       `f32` |        768 |     IP |      7 GB | 2.3M queries, yes |

### ~10M Scale

| Dataset                              | Scalar Type | Dimensions |  Metric | Base Size | Ground Truth      |
| :----------------------------------- | ----------: | ---------: | ------: | --------: | :---------------- |
| [Meta BIGANN — SIFT][meta-bigann]    |        `u8` |        128 |      L2 |    1.2 GB | 10K queries, yes  |
| [Microsoft Turing-ANNS][msft-turing] |       `f32` |        100 |      L2 |    3.7 GB | 100K queries, yes |
| [Cohere Wiki EN][cohere-wiki]        |        `b1` |       1024 | Hamming |    5.3 GB | self-sampled ¹    |

> ¹ These sources ship vectors without ground truth, so their `retri-download-*` binaries sample queries with a fixed seed and compute exact Hamming top-K.

### ~100M Scale

| Dataset                               | Scalar Type | Dimensions |  Metric | Base Size | Ground Truth      |
| :------------------------------------ | ----------: | ---------: | ------: | --------: | :---------------- |
| [Meta BIGANN — SIFT][meta-bigann]     |        `u8` |        128 |      L2 |     12 GB | 10K queries, yes  |
| [Microsoft Turing-ANNS][msft-turing]  |       `f32` |        100 |      L2 |     37 GB | 100K queries, yes |
| [Microsoft SpaceV][msft-spacev]       |        `i8` |        100 |      L2 |    9.3 GB | 30K queries, yes  |
| [Unum WikiVerse][wikiverse] ²         |       `f16` |  128–4096³ |  Cos/IP |  95-505GB | pipeline pending  |
| [USearchMolecules PubChem][usm] MACCS |        `b1` |        168 | Hamming |    2.4 GB | self-sampled ¹    |
| [USearchMolecules PubChem][usm] ECFP4 |        `b1` |       2048 | Hamming |     29 GB | self-sampled ¹    |

> ² WikiVerse ships `.f16bin`, which RetriEval does not read yet.
> ³ nomic-embed at 768, arctic-embed and Qwen3 at 1024, e5-mistral at 4096; the ColBERT model's 128d per token needs multi-vector support.

### ~1B Scale

| Dataset                                    | Scalar Type | Dimensions |  Metric | Base Size | Ground Truth      |
| :----------------------------------------- | ----------: | ---------: | ------: | --------: | :---------------- |
| [Meta BIGANN — SIFT][meta-bigann]          |        `u8` |        128 |      L2 |    119 GB | 10K queries, yes  |
| [Microsoft Turing-ANNS][msft-turing]       |       `f32` |        100 |      L2 |    373 GB | 100K queries, yes |
| [Microsoft SpaceV][msft-spacev]            |        `i8` |        100 |      L2 |     93 GB | 30K queries, yes  |
| [Yandex Text-to-Image][yandex-t2i]         |       `f32` |        200 |     Cos |    750 GB | 100K queries, yes |
| [Yandex Deep][yandex-deep]                 |       `f32` |         96 |      L2 |    358 GB | 10K queries, yes  |
| [USearchMolecules GDB-13][usm] MACCS       |        `b1` |        168 | Hamming |     21 GB | self-sampled ¹    |
| [USearchMolecules GDB-13][usm] ECFP4       |        `b1` |       2048 | Hamming |    250 GB | self-sampled ¹    |
| [USearchMolecules Enamine REAL][usm] MACCS |        `b1` |        168 | Hamming |    127 GB | self-sampled ¹    |
| [USearchMolecules Enamine REAL][usm] ECFP4 |        `b1` |       2048 | Hamming |   1.55 TB | self-sampled ¹    |

[unum-wiki-1m]: data/README.md#unum-uform-wiki
[unum-arxiv-2m]: data/README.md#arxiv-with-e5
[meta-bigann]: data/README.md#meta-bigann-sift
[msft-turing]: data/README.md#microsoft-turing-anns
[msft-spacev]: data/README.md#microsoft-spacev
[yandex-deep]: data/README.md#yandex-deep
[yandex-t2i]: data/README.md#yandex-text-to-image
[usm]: data/README.md#usearchmolecules
[cohere-wiki]: data/README.md#cohere-wikipedia-multilingual
[wikiverse]: data/README.md#unum-wikiverse

## Development

The repository pins `nightly-2026-09-24`.
Run `scripts/check.sh` for formatting, Clippy, and CPU tests; pass an explicit feature list for optional native engines.
`src/` holds the shared library in `bench.rs`, one binary per backend and `retri-generate`, `scripts/` the dataset downloaders, plotting and Git hooks, and `docker/` the compose files for the server backends.
