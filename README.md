![RetriEval benchmarks thumbnail](https://github.com/ashvardanian/ashvardanian/raw/master/repositories/RetriEval.jpg?raw=true)

__RetriEval__ is a benchmarking suite designed for Billion-scale Vector Search workloads.
It's primarily used to benchmark in-process Search Engines on CPUs and GPUs, like [USearch](https://github.com/unum-cloud/usearch), [FAISS](https://github.com/facebookresearch/faiss), and [cuVS](https://github.com/NVIDIA/cuvs), but it also reuses similar profiling logic for standalone databases like [Qdrant](https://github.com/qdrant/qdrant), [Weaviate](https://github.com/weaviate/weaviate), and [Redis](https://github.com/redis/redis).
It works with the same plain input format standardized by the [BigANN benchmark](https://big-ann-benchmarks.com/), aiming for reproducible measurements – with shuffled parallel construction, incremental recall curves, normalized metrics, and machine-readable reports, capturing everything from machine topology to indexing hyper-parameters.

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

| Backend     | Parallelism | Quantization                                  | Metrics                       |
| :---------- | :---------- | :-------------------------------------------- | :---------------------------- |
| __USearch__ | ForkUnion   | bf16, f16, e5m2, e4m3, e3m2, e2m3, i8, u8, b1 | ip, l2, cos, hamming, jaccard |
| __FAISS__   | OpenMP      | f32, f16, bf16, u8, i8, b1                    | ip, l2                        |
| __cuVS__    | CUDA        | f32, f16, i8, u8                              | l2, ip, cos                   |

- __USearch__: The input is stored in the `--data-type` quantization, converted on the fly.
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

## Results

Each table holds one row per backend, device and data type, so a new run is a new row and nothing else moves.
`1-r@10` is 1-recall@10, the share of queries whose true nearest neighbor lands in the top 10, and `10-r@10` is K-recall@10, the share of the true top 10 found.
`Connectivity` is the graph degree, and `Expansions` the construction and search expansion, split by `/`.
`Add/s` and `Search/s` count vectors per second, `Memory` is the backend's own peak figure, and `Build` is the wall time to index the whole base.
`Device` is `CPU` for a dual-socket server and `GPU` for a single accelerator, and every backend took its native input type, with no rescaling.

The recommended methodology is to sweep each backend's settings until they reach 99% recall@10 or more on a given dataset.
A row never builds faster than it searches: when `Add/s` exceeds `Search/s`, raise the construction breadth and lower the search breadth until search is at least as fast at that recall.
Once the behavior is confirmed on a 1M–10M subset, 100M–1B runs validate the scaling curves.

Each search reports both recall conventions, because they answer different questions and one of them saturates.
`recall_at_k` is 1-recall@K, FAISS's `OneRecallAtRCriterion` and the measure USearch reports.
`intersection_at_k` is K-recall@K, `|top-K ∩ ground-truth-K| / K`, order-independent, and what ann-benchmarks and cuVS's own harness publish as recall.
1-recall@K ≥ K-recall@K for the same run, so the two are never interchangeable.

K is `--top-k`, which defaults to the ground-truth file's width, 10 on the Wiki sets and 100 on the BigANN ones.
1-recall@K stops discriminating as K grows: at K=100 nearly any sane configuration finds the single true neighbor somewhere in the list.
`intersection_at_k` stays informative there, and `ndcg_at_k` likewise normalizes over the full truth prefix, so both are stricter at wide K than the same-named numbers at K=10.
Pass `--top-k 10` for figures comparable across datasets of differing ground-truth width.

### PubChem MACCS

168-bit binary fingerprints of PubChem molecules under Hamming, with settings calibrated at 10M and reused at 100M; recipes in [`data/README.md`](data/README.md#usearchmolecules).

#### PubChem MACCS 10M

| Backend      | Device | Type | Connectivity | Expansions | 1-r@10 | 10-r@10 |  Add/s | Search/s | Memory | Build |
| :----------- | :----- | :--- | -----------: | ---------: | -----: | ------: | -----: | -------: | -----: | ----: |
| USearch 2.25 | CPU    | `b1` |           32 |     128/64 | 0.9696 |       … | 36,347 |   35,767 | 4.7 GB |  5.0m |
| FAISS 1.12   | CPU    | `b1` |           64 |      40/16 | 0.9661 |       … | 95,230 |  293,795 | 7.6 GB |  1.5m |

#### PubChem MACCS 100M

| Backend      | Device | Type | Connectivity | Expansions | 1-r@10 | 10-r@10 |   Add/s |  Search/s |  Memory |        Build |
| :----------- | :----- | :--- | -----------: | ---------: | -----: | ------: | ------: | --------: | ------: | -----------: |
| USearch 2.25 | CPU    | `b1` |           32 |     128/64 | 0.8438 |       … |  35,080 |    38,432 | 40.8 GB |        54.6m |
| FAISS 1.12   | CPU    | `b1` |           64 |      40/16 |      … |       … |       … |         … | ≥ 63 GB | killed at 9h |
| USearch 3.0  | GPU    | `b1` |           32 |     256/64 | 0.9907 |  0.9906 | 559,140 | 1,460,000 | 18.3 GB |         3.4m |

### SIFT

Meta's 128-dimensional `u8` descriptors under L2, compared at the setting that first reaches 99% recall@10; recipes in [`data/README.md`](data/README.md#meta-bigann-sift).

#### SIFT 10M

| Backend      | Device | Type  | Connectivity | Expansions | 1-r@10 | 10-r@10 |   Add/s |  Search/s |  Memory | Build |
| :----------- | :----- | :---- | -----------: | ---------: | -----: | ------: | ------: | --------: | ------: | ----: |
| USearch 2.25 | CPU    | `u8`  |           16 |    128/256 | 0.9938 |       … |  35,405 |    80,729 |  4.4 GB |  4.8m |
| FAISS 1.12   | CPU    | `u8`  |           16 |    128/256 | 0.9952 |       … |  26,374 |    38,278 |  5.9 GB |  5.6m |
| cuVS 26.10   | GPU    | `u8`  |           64 |    128/128 |      … |  0.9966 | 204,838 | 1,204,000 |       … |  0.8m |
| USearch 3.0  | GPU    | `u8`  |           32 |    256/128 | 0.9911 |  0.9780 | 558,510 | 1,040,000 |  2.6 GB |  0.3m |
| FAISS 1.15.1 | CPU    | `u8`  |           32 |     128/96 | 0.9900 |  0.9765 | 115,643 |   133,919 |  8.6 GB |  1.4m |
| FAISS 1.15.1 | CPU    | `f32` |           32 |     128/96 | 0.9900 |  0.9765 |  59,688 |    61,027 | 14.4 GB |  2.8m |

#### SIFT 100M

| Backend      | Device | Type | Connectivity | Expansions | 1-r@10 | 10-r@10 |   Add/s | Search/s |  Memory |        Build |
| :----------- | :----- | :--- | -----------: | ---------: | -----: | ------: | ------: | -------: | ------: | -----------: |
| USearch 2.25 | CPU    | `u8` |           16 |    128/256 | 0.9833 |       … |  39,831 |   75,808 | 53.7 GB |        48.7m |
| FAISS 1.12   | CPU    | `u8` |           16 |    128/256 |      … |       … |       … |        … | ≥ 46 GB | killed at 9h |
| cuVS 26.10   | GPU    | `u8` |           64 |    128/256 |      … |  0.9963 | 154,799 |  582,000 |       … |        10.8m |
| USearch 3.0  | GPU    | `u8` |           32 |    256/256 | 0.9913 |  0.9810 | 431,520 |  604,490 | 24.8 GB |         3.9m |

### Turing-ANNS

Microsoft's 100-dimensional `f32` web-search embeddings under L2, compared at the setting that first reaches 99% recall@10 across storage types; recipes in [`data/README.md`](data/README.md#microsoft-turing-anns).

#### Turing-ANNS 10M

| Backend      | Device | Type   | Connectivity | Expansions | 1-r@10 | 10-r@10 |  Add/s | Search/s |  Memory | Build |
| :----------- | :----- | :----- | -----------: | ---------: | -----: | ------: | -----: | -------: | ------: | ----: |
| USearch 2.25 | CPU    | `f32`  |           48 |    768/384 | 0.9929 |       … |  8,532 |   12,331 | 13.0 GB | 18.3m |
| USearch 2.25 | CPU    | `bf16` |           48 |    768/384 | 0.9929 |       … | 10,496 |   16,940 | 10.9 GB | 14.1m |
| USearch 2.25 | CPU    | `f16`  |           48 |    768/384 | 0.9929 |       … | 10,969 |   20,246 | 10.9 GB | 13.5m |
| USearch 2.25 | CPU    | `e5m2` |           48 |    768/384 | 0.9919 |       … | 10,526 |   20,534 |  9.8 GB | 13.5m |
| USearch 2.25 | CPU    | `e4m3` |           48 |    768/384 | 0.9930 |       … |  7,353 |   12,106 |  9.8 GB | 19.4m |
| USearch 2.25 | CPU    | `e3m2` |           48 |    768/384 | 0.9728 |       … | 10,398 |   18,022 |  9.8 GB | 13.3m |
| USearch 2.25 | CPU    | `e2m3` |           48 |    768/384 | 0.7941 |       … | 10,935 |   21,313 |  9.8 GB | 13.2m |
| FAISS 1.12   | CPU    | `f32`  |           48 |    768/384 | 0.9944 |       … |  7,491 |   16,486 | 14.1 GB | 20.6m |
| FAISS 1.12   | CPU    | `bf16` |           48 |    768/384 | 0.9944 |       … |  3,800 |   10,391 | 12.1 GB | 39.4m |
| FAISS 1.12   | CPU    | `f16`  |           48 |    768/384 | 0.9944 |       … |  2,545 |   10,032 | 12.1 GB | 1h 1m |
| USearch 3.0  | GPU    | `bf16` |           32 |  1024/1024 | 0.9908 |  0.9831 | 95,870 |  155,810 |  3.4 GB |  1.7m |

#### Turing-ANNS 100M

| Backend      | Device | Type   | Connectivity | Expansions | 1-r@10 | 10-r@10 |  Add/s | Search/s |   Memory |  Build |
| :----------- | :----- | :----- | -----------: | ---------: | -----: | ------: | -----: | -------: | -------: | -----: |
| USearch 2.25 | CPU    | `f32`  |           48 |    768/384 | 0.9929 |       … |  6,646 |   10,398 | 139.6 GB |  4h 1m |
| USearch 2.25 | CPU    | `bf16` |           48 |    768/384 | 0.9931 |       … |  8,564 |   14,772 | 105.2 GB |  3h 1m |
| USearch 2.25 | CPU    | `f16`  |           48 |    768/384 | 0.9930 |       … |  8,807 |   15,412 | 105.2 GB | 2h 54m |
| USearch 2.25 | CPU    | `e5m2` |           48 |    768/384 | 0.9924 |       … |  7,368 |   13,227 |  88.0 GB | 3h 15m |
| USearch 3.0  | GPU    | `bf16` |           32 |  1024/1024 | 0.9933 |  0.9868 | 88,660 |  156,790 |  32.1 GB |  18.8m |

## CLI Reference

Each backend is a separate binary, and the manuals below list every flag of every binary with its default.
A value spelled `<…,...>` takes comma-separated values and runs one configuration per combination.
A bad value prints `--flag="value" does not parse, expected …` and exits with status 1.
Every `retri-eval-*` flag also reads `RETRIEVAL_<FLAG>` as its default, upper-cased with dashes as underscores, so `--top-k` reads `RETRIEVAL_TOP_K`.
The flag on the command line overrides the variable, and switches like `--self-search` read `1`, `0`, `true` or `false`.

<details>
<summary>Shared by every <code>retri-eval-*</code> binary</summary>

```text
retri-eval-*
  --base-vectors <PATH|GLOB>            required
      Base vectors to index: .fbin, .u8bin, .i8bin or .b1bin.
  --base-keys <PATH>                    sequential
      Keys for the base vectors, .i32bin, one per vector.
  --query-vectors <PATH>                required
      Query vectors to search with.
  --query-neighbors <PATH>              required
      Ground-truth neighbors for those queries, .ibin.
  --top-k <K>                           ground-truth width
      Neighbors per query, the K every metric is taken at; at most the ground-truth width.
  --insertion-order <ORDER>             shuffled
      shuffled by --seed, or the file's original order.
  --seed <SEED>                         42
      32-bit seed of the shuffled order, or random; rejected with --insertion-order original.
  --steps <N>                           10
      Measurement steps; the base is inserted in this many equal parts.
  --inserts-per-core <N>                2048
      Vectors per add call for each backend thread: --threads on usearch and faiss, one elsewhere.
  --queries-per-core <N>                2048
      Queries per search call for each backend thread: --threads on usearch and faiss, one elsewhere.
  --output <DIR>                        none
      Directory for JSON result files; progress only when unset.
  --max-base-vectors <N>                all
      Cap on the base vectors used, for calibration on a slice.
  --dims <N,...>                        native
      Matryoshka truncations, each at most the native width, and a multiple of 8 for .b1bin.
  --self-search                         off
      After the last insertion, replay indexed vectors as their own queries.
  --self-search-top-k <K>               10
      Neighbors per self-search query.
  --self-search-sample <N|FRACTION>     all
      Leading base rows to replay, a count like 50000 or a fraction like 0.1; needs --self-search.
```

</details>

<details>
<summary>Search engines: <code>retri-eval-usearch</code>, <code>retri-eval-faiss</code> & <code>retri-eval-cuvs</code></summary>

```text
retri-eval-usearch, retri-eval-faiss, retri-eval-cuvs
  --index <PATH>                        none
      Persisted index: searched if the path exists, else built and saved; needs a single config.

retri-eval-usearch
  --data-type <TYPE,...>                bf16
      Stored type: bf16, f16, e5m2, e4m3, e3m2, e2m3, i8, u8 or b1.
  --metric <METRIC,...>                 l2
      ip, l2, cos, hamming or jaccard.
  --connectivity <N,...>                16
      Graph connectivity.
  --expansion-add <N,...>               128
      Construction expansion.
  --expansion-search <N,...>            64
      Search expansion.
  --memory-domains <NODES>              every node the process may use
      Comma-separated NUMA nodes, one shard on each.
  --threads <N,...>                     0
      Threads, 0 for all cores.

retri-eval-faiss
  --data-type <TYPE,...>                bf16
      Scalar quantizer over f32 input: f32, f16, bf16, u8, i8 or b1.
  --metric <METRIC,...>                 l2
      ip or l2; b1 data always uses Hamming.
  --connectivity <N,...>                32
      HNSW M.
  --expansion-add <N,...>               128
      HNSW construction width.
  --expansion-search <N,...>            64
      HNSW search width.
  --threads <N>                         0
      OpenMP threads, 0 for all cores.

retri-eval-cuvs
  --data-type <TYPE,...>                f32
      f32, f16, i8 or u8, converted on the host before CAGRA builds on it.
  --metric <METRIC,...>                 l2
      l2, ip or cos.
  --connectivity <N,...>                32
      CAGRA's output graph degree.
  --expansion-add <N,...>               64
      CAGRA's intermediate graph degree before pruning.
  --expansion-search <N,...>            64
      CAGRA's internal top-i list.
  --search-width <N|auto,...>           auto
      Graph nodes that start each search iteration, or auto for cuVS's choice.
  --min-iterations <N>                  cuVS's choice
      Minimum search iterations.
  --max-iterations <N>                  cuVS's choice
      Maximum search iterations.
  --num-random-samplings <N>            cuVS's choice
      Random sampling rounds for the initial search points.
  --build-algo <ALGORITHM>              auto
      Graph build algorithm, auto or nn_descent.
```

</details>

<details>
<summary>Vector databases: <code>retri-eval-qdrant</code>, <code>-redis</code>, <code>-weaviate</code>, <code>-surrealdb</code>, <code>-lancedb</code> & <code>-turso</code></summary>

```text
retri-eval-qdrant, retri-eval-redis, retri-eval-weaviate, retri-eval-surrealdb
  --startup-time-limit <DURATION>       120s
      Time limit for container start and readiness.
  --connectivity <N,...>                16
      HNSW M.
  --expansion-add <N,...>               128
      HNSW construction width.
  --expansion-search <N,...>            64
      HNSW search width.

retri-eval-qdrant
  --data-type <TYPE,...>                f32
      Storage type: f32, f16 or u8.
  --metric <METRIC,...>                 l2
      ip, cos, l2 or manhattan.
  --quantization <MODE,...>             none
      Server-side none, binary or scalar.
  --grpc-port <PORT>                    6334
      Host gRPC port.
  --http-port <PORT>                    6333
      Host HTTP port.
  --vectors-per-upsert <N>              10000
      Vectors per upsert request.

retri-eval-redis
  --data-type <TYPE,...>                f32
      f32, f64, f16, bf16, u8 or i8; u8 and i8 need Redis 8.
  --metric <METRIC,...>                 l2
      l2, ip or cos.
  --port <PORT>                         6379
      Host port.
  --vectors-per-upsert <N>              1000
      Vectors per pipeline flush.

retri-eval-weaviate
  --metric <METRIC,...>                 l2
      l2, ip or cos.
  --quantization <MODE,...>             none
      Server-side none or binary.
  --port <PORT>                         8080
      Host HTTP port.

retri-eval-surrealdb
  --metric <METRIC,...>                 l2
      l2 or cos.
  --port <PORT>                         8000
      Host HTTP port.

retri-eval-lancedb
  --metric <METRIC>                     l2
      One of ip, cos or l2.
  --db-path <DIR>                       /tmp/retrieval-lancedb
      LanceDB storage directory.

retri-eval-turso
  --metric <METRIC>                     l2
      One of l2 or cos.
  --db-path <PATH>                      :memory:
      Turso database file, or :memory:; only its bench table is recreated.
```

</details>

<details>
<summary>Dataset tools: <code>retri-generate</code>, <code>retri-download-molecules</code> & <code>retri-download-cohere</code></summary>

```text
retri-generate, retri-download-molecules, retri-download-cohere
  --threads <N>                         0
      Ground-truth threads, 0 for all cores.
  --seed <SEED>                         42
      32-bit seed of generation or query sampling, or random.
  --output <DIR>                        required
      Output directory.

retri-generate
  --format <FORMAT>                     required
      b1bin for clustered binary with Hamming ground truth, or fbin for Gaussian f32 with L2.
  --base-count <N>                      required
      Base vectors to generate.
  --query-count <N>                     required
      Query vectors to generate.
  --dims <N>                            required
      Bits for b1bin, a multiple of 8, or scalars for fbin.
  --clusters <N>                        256
      Cluster centers.
  --noise <PROBABILITY>                 0.1
      Bit-flip probability from 0 to 1.
  --top-k <K>                           10
      Neighbors per query in the ground-truth file.

retri-download-molecules, retri-download-cohere
  --query-count <N>                     10000
      Queries sampled from the base, along with the ground truth.
  --limit <N>                           all
      Rows to extract.
  --download-concurrency <N>            4
      Concurrent shard downloads.
  --keep-parquet                        off
      Keep the Parquet shards under <output>/parquet/.

retri-download-molecules
  --source <SOURCE>                     required
      pubchem (115M), gdb13 (977M) or enamine (6.04B).
  --fingerprint <COLUMN>                required unless --conformers
      maccs (166 bits), pubchem (881), ecfp4 (2048) or fcfp4 (2048).
  --conformers                          off
      Each embedded molecule's first conformer, as base.part_N.f16bin xyz points beside
      counts.part_N.ibin points per cloud; excludes --fingerprint and --top-k.
  --top-k <K>                           none
      Neighbors per query in the ground-truth file, which only this flag writes.
  --part-rows <N>                       512000000
      Rows per base part, a multiple of the bucket's 1M-row shards.
  --url-prefix <URL>                    the public bucket
      S3 or HTTPS prefix of the shards.

retri-download-cohere
  --top-k <K>                           10
      Neighbors per query in the ground-truth file.
  --language <CODE>                     en
      Language config of the Hugging Face dataset.
  --no-text                             off
      Skip the aligned titles.txt, texts.txt and urls.txt.
```

</details>

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

## Development

The repository pins `nightly-2026-09-24`.
Run `scripts/check.sh` for formatting, Clippy, and CPU tests; pass an explicit feature list for optional native engines.
`src/` holds the shared library in `bench.rs`, one binary per backend and `retri-generate`, `scripts/` the dataset downloaders, plotting and Git hooks, and `docker/` the compose files for the server backends.
