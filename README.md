![RetriEval benchmarks thumbnail](https://github.com/ashvardanian/ashvardanian/raw/master/repositories/RetriEval.jpg?raw=true) 

__RetriEval__ is a benchmarking suite designed for Billion-scale Vector Search workloads.
It's primarily used to benchmark in-process Search Engines on CPUs and GPUs, like [USearch](https://github.com/unum-cloud/usearch), [FAISS](https://github.com/facebookresearch/faiss), and [cuVS](https://github.com/rapidsai/cuvs), but it also reuses similar profiling logic for standalone databases like [Qdrant](https://github.com/qdrant/qdrant), [Weaviate](https://github.com/weaviate/weaviate), and [Redis](https://github.com/redis/redis).
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

The recommended methodology is to parameter-sweep different configuration options to achieve comparable recall between search backends on a given dataset.
Once the behavior is confirmed on a small 1M–10M subset, 100M–1B and larger benchmarks can be run to validate scaling curves.

## Quick Start

Install the default `retri-eval-usearch` binary:

```sh
cargo install --path .
```

Fetch the Unum Wiki 1M dataset — ~400 MB of vectors, queries, and ground truth:

```sh
mkdir -p datasets/wiki_1M && \
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-wiki-1m/resolve/main/base.1M.fbin -P datasets/wiki_1M/ && \
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-wiki-1m/resolve/main/query.public.100K.fbin -P datasets/wiki_1M/ && \
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-wiki-1m/resolve/main/groundtruth.public.100K.ibin -P datasets/wiki_1M/
```

Run a sweep over three quantizations and write JSON reports under `results/`:

```sh
retri-eval-usearch \
    --vectors datasets/wiki_1M/base.1M.fbin \
    --queries datasets/wiki_1M/query.public.100K.fbin \
    --neighbors datasets/wiki_1M/groundtruth.public.100K.ibin \
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
- __cuVS__: Currently benchmarks with f32.
  CAGRA natively supports f32, f16, i8, u8 for build.

```sh
retri-eval-usearch --data-type bf16 --metric l2 ...
retri-eval-faiss --data-type f16 --metric l2 ...
retri-eval-cuvs --metric l2 ...
```

### Vector Databases

Server-side quantization is managed by the database engine, not the benchmark.
Binary quantization is deterministic `sign(x)` per dim, and scalar quantization is per-dim min/max — neither trains a codebook, so both stay inside the "no learned logic" constraint the rest of the benchmark holds for the native backends.
Product quantization is deliberately excluded everywhere.

| Backend      | Client                       | Docker Image                        | Metrics                | Wire dtype sweep                        | Server-side quantization   |
| ------------ | ---------------------------- | ----------------------------------- | ---------------------- | --------------------------------------- | -------------------------- |
| __Qdrant__   | `qdrant-client`, gRPC        | `qdrant/qdrant:v1.17.1`             | ip, l2, cos, manhattan | `f32`, `f16`, `u8`                      | `none`, `binary`, `scalar` |
| __Redis__    | `redis`, RESP                | `redis:8.6`                         | ip, l2, cos            | `f32`, `f64`, `f16`, `bf16`, `u8`, `i8` | —                          |
| __Weaviate__ | `weaviate-community`, REST   | `semitechnologies/weaviate:1.36.10` | ip, l2, cos            | `f32` only                              | `none`, `binary`           |
| __LanceDB__  | `lancedb`, in-process, Arrow | —                                   | ip, l2, cos            | `f32` only                              | — (IVF-bucketed only) ¹    |

¹ LanceDB's Rust client — `lancedb 0.27` — exposes graph-based search only via `IvfHnswPq` / `IvfHnswSq`, both IVF-bucketed and PQ k-means-trained. No pure-HNSW variant is offered, so this benchmark leaves LanceDB on plain `f32` + L2/IP/Cos until upstream adds one. Hamming is only available on `IvfFlat`, outside our graph path.

Redis 8.x is required for `i8`, `u8`, `f16`, and `bf16` — the older `redis/redis-stack` images on Redis 7.4 reject those four types at `FT.CREATE`.
Qdrant server-side `Float16` and `Uint8` accept f32 upserts and convert on ingest, so the wire payload we send is unchanged.
Weaviate stores only f32 internally; the wire dtype sweep there is intentionally a single-option list.

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
```

Or combine multiple:

```sh
cargo build --release --features usearch-backend,faiss-backend,qdrant-backend
```

## CLI Reference

Each backend is a separate binary. Common flags shared by all:

```
--vectors <PATH|GLOB>      # Base vectors (.fbin, .u8bin, .i8bin, .b1bin)
--queries <PATH|GLOB>      # Query vectors
--neighbors <PATH|GLOB>    # Ground-truth neighbors (.ibin)
--keys <PATH|GLOB>         # Optional keys file (.i32bin)
--epochs <N>               # Measurement steps (dataset split into N parts, default: 10)
--no-shuffle               # Disable random insertion order (shuffle is on by default)
--output <DIR>             # Output directory for JSON result files (omit for progress-only)
--index <PATH>             # Persisted index handle. If the path exists, the run skips the
                           # add phase, loads, and search-only-runs; otherwise the run
                           # builds, then saves to that path. Requires a single-config sweep.
                           # USearch / FAISS / cuVS only.
--dimensions <LIST>        # Matryoshka truncations to evaluate (e.g. 128,256,512,1024).
                           # Empty → use the file's native dim. Each value must be ≤ native;
                           # for `.b1bin` files each must be a multiple of 8.
```

`--vectors` / `--queries` / `--neighbors` / `--keys` accept shell glob patterns
(`*`, `?`, `[…]`). Matched shards are natural-sorted (`shard_2.fbin` before
`shard_10.fbin`) and validated for matching dim and scalar format — useful for
multi-shard datasets like USearchWiki.

__retri-eval-usearch__ additionally supports comma-separated sweeps:

```
--data-type <LIST>         # f32, f16, bf16, e5m2, e4m3, e3m2, e2m3, i8, u8, b1
--metric <LIST>            # ip, l2, cos, hamming, jaccard, sorensen, pearson, haversine, divergence
--connectivity <LIST>      # HNSW M parameter (default: 0 = auto)
--expansion-add <LIST>     # expansion factor during indexing (default: 0 = auto)
--expansion-search <LIST>  # expansion factor during search (default: 0 = auto)
--shards <LIST>            # Index shards (default: 2)
--threads <LIST>           # Thread count (default: available cores)
```

__retri-eval-cuvs__ — requires `--features cuvs-backend` and an NVIDIA GPU:

```
--data-type <LIST>                 # f32, f16, u8                  (default: f32)
--metric <LIST>                    # l2, ip, cos (default: l2)
--graph-degree <LIST>              # CAGRA output graph degree (default: 32)
--intermediate-graph-degree <LIST> # CAGRA intermediate graph degree (default: 64)
--itopk-size <LIST>                # Search-time intermediate results (default: 64)
```

__retri-eval-qdrant__ extends the common flags with:

```
--data-type <LIST>      # f32, f16, u8                  (default: f32)
--quantization <LIST>   # none, binary, scalar          (default: none)
--metric <LIST>         # ip, l2, cos, manhattan        (default: l2)
```

__retri-eval-redis__ extends the common flags with:

```
--data-type <LIST>      # f32, f64, f16, bf16, u8, i8   (default: f32)
--metric <LIST>         # ip, l2, cos                   (default: l2)
```

__retri-eval-weaviate__ extends the common flags with:

```
--quantization <LIST>   # none, binary                  (default: none)
--metric <LIST>         # ip, l2, cos                   (default: l2)
```

## Observability

### Hardware Counters on Linux

Wall-clock throughput and peak RSS are always recorded in the JSON report.
For deeper attribution — "how many cycles did construction spend in cache misses vs searching?" — build with `--features perf-counters`.
On Linux this pulls [`perf-event2`] and wraps the `index.add` and `index.search` loops inside `src/bench.rs::run` with system-wide hardware counters, populating eight new optional fields on each `StepEntry`:

```
cycles_add / instructions_add / cache_misses_add / branch_misses_add
cycles_search / instructions_search / cache_misses_search / branch_misses_search
```

Fields are `Option<u64>` with `skip_serializing_if = "Option::is_none"`, so reports from runs without the feature are byte-identical to the pre-feature schema.

```sh
sudo sysctl -w kernel.perf_event_paranoid=-1   # once per host
ulimit -n 65536                                 # see RLIMIT note below
cargo build --release --features usearch-backend,perf-counters

retri-eval-usearch \
    --vectors datasets/pubchem_maccs/base.115627267.b1bin \
    --queries datasets/pubchem_maccs/query.10000.b1bin \
    --neighbors datasets/pubchem_maccs/groundtruth.10000.ibin \
    --data-type b1 --metric hamming --output results/pubchem_maccs
```

__Scope__ is system-wide per-CPU — `pid == -1`, `cpu == i`, one counter group per online CPU, summed at read.
This is the only way to cover every ForkUnion pool thread, because per-process `inherit(true)` would miss workers spawned before the counter was enabled.
Trade-off: on shared hosts the numbers include other tenants' activity; on a dedicated box this is exactly what you want.

__Permissions__ require `CAP_PERFMON` or `CAP_SYS_ADMIN`, or relaxed paranoia via `kernel.perf_event_paranoid ≤ 0`.
Without either, `PerfCounters::new` returns `EACCES`, the bench prints `perf counters: unavailable …; running without` and completes normally with the counter fields absent.

__RLIMIT_NOFILE__ matters: each CPU opens six file descriptors — a no-op leader fd plus five hardware counters.
At 192 CPUs that's 1,152 fds, above the default `ulimit -n 1024` on most distros.
Bump it per shell with `ulimit -n 65536` or system-wide via `/etc/security/limits.conf` before running.
Without the bump you'll get `EMFILE` around the 170th CPU's group.

__Cross-platform__ is Linux-only.
On macOS, Windows, or BSD, Cargo simply does not pull `perf-event2` into the dep graph — the dep line is gated behind `[target.'cfg(target_os = "linux")']`.
The module falls back to a stub whose `PerfCounters::new` returns `Unsupported`.
Enabling the feature on a non-Linux target compiles cleanly and runs as if it were disabled — you still get the JSON, just without counter fields.

### External `perf stat` Sidecar

When profiling an already-compiled binary, or when you want OS-level metrics alongside hardware counters, run `perf stat` and `mpstat` directly alongside the bench instead of rebuilding with `--features perf-counters`.

```sh
sudo apt install linux-tools-common linux-tools-generic sysstat
sudo sysctl -w kernel.perf_event_paranoid=-1

mpstat 1 > results/cohere_en/mpstat.txt &        # 1 Hz all-core utilization
perf stat -a -e cycles,instructions,cache-references,cache-misses,\
LLC-load-misses,branch-misses,context-switches,cpu-migrations,page-faults \
    --output results/cohere_en/perf.txt -- \
    retri-eval-usearch \
        --vectors datasets/cohere_en/base.41488110.b1bin \
        --queries datasets/cohere_en/query.10000.b1bin \
        --neighbors datasets/cohere_en/groundtruth.10000.ibin \
        --data-type b1 --metric hamming \
        --output results/cohere_en
kill %1
```

This covers the whole process lifetime including dataset-load and ground-truth I/O rather than just the add/search loops, useful for spotting cost outside the measured regions.

### Memory Consumption Tracking

`StepEntry.memory_bytes` is populated per step by asking the backend what it's currently using.
The mechanism depends on the backend:

| Backend                                 | How `memory_bytes` is measured                                                                                                                                                                                 |
| :-------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| In-process — USearch, FAISS, cuVS       | The engine exposes its internal allocator or `index.size()` API, giving exact index footprint excluding dataset mmap. USearch: `index.memory_usage()`. FAISS: `index.stats().indexed_vectors * sizeof`.        |
| Tier 2 Docker — Qdrant, Redis, Weaviate | `docker stats --no-stream --format '{{.MemUsage}}'` is sampled per step against the running container and parsed into bytes. This includes the whole engine process, not just the index, so it's an overcount. |
| LanceDB — in-process, Arrow IPC         | Filesystem-backed; `memory_bytes` reports the table's on-disk size from `fs::metadata`, not RSS.                                                                                                               |

The `peak memory` line printed at the end of a run is `steps.iter().map(|s| s.memory_bytes).max()`.
Process-wide peak RSS — the kernel's accounting of everything including mmapped datasets — is available via `getrusage(RUSAGE_SELF)` but is not currently reported in the JSON.
On the wishlist if you want mmap cost separated out.

### Tier 2 Backend Docker Lifecycle

Tier 2 backends — Qdrant, Redis, Weaviate — don't run in-process.
They run as Docker containers the benchmark spawns and tears down automatically.
`src/docker.rs` wraps `bollard`, the async Docker API client, and does:

1. __Pull__ — runs `docker pull qdrant/qdrant:vX.Y.Z` or equivalent if the image isn't cached locally.
2. __Run__ — creates the container with port bindings and environment variables from the compose file at `docker/<backend>.yml`, then starts it.
3. __Wait for ready__ — polls an HTTP health endpoint such as `/healthz` or `/health` with 500 ms intervals until the backend accepts connections, or a configurable timeout fires.
4. __Run the benchmark__ against the container.
5. __Stop and remove__ the container regardless of success or failure — RAII-style via `ContainerHandle::Drop`.

Per-step memory for these backends comes from the Docker stats API.
`memory_bytes` reflects the container's resident set including the engine process, its heap, page cache attributed to it, and so on.
Overcount compared to just-the-index, but it's the honest picture of what the engine costs to run.

Requirements: Docker daemon accessible over Unix socket or TCP, images pullable by the current user.
On systems where the Docker daemon runs as root, either add your user to the `docker` group or run the benchmark with `sudo`.

## Output Format

One JSON file per backend configuration, written to `--output <dir>`.
Files are auto-named `<backend>-<hash>.json`.

```json
{
  "machine": { "cpu_model": "Intel Xeon 6776P", "physical_cores": 96, ... },
  "dataset": { "vectors_path": "...", "vectors_count": 10000000, "dimensions": 100, ... },
  "config": { "backend": "usearch", "data_type": "f32", "metric": "l2", "connectivity": 16, ... },
  "steps": [
    {
      "vectors_indexed": 1000000,
      "add_elapsed": 12.3,
      "add_throughput": 81300,
      "memory_bytes": 412000000,
      "search_elapsed": 0.45,
      "search_throughput": 222000,
      "recall_at_1": 0.0942,
      "recall_at_10": 0.2815,
      "ndcg_at_10": 0.1847,
      "recall_at_1_normalized": 0.9420,
      "recall_at_10_normalized": 0.9512,
      "ndcg_at_10_normalized": 0.8470
    }
  ]
}
```

## Project Structure

```
Cargo.toml
src/
    bench.rs                # Library root: Backend trait, types, BenchState, benchmark loop
    dataset.rs              # Memory-mapped .fbin/.ibin loading (zero-copy)
    eval.rs                 # Recall@K, NDCG@K
    output.rs               # Report types, JSON writer, machine info
    docker.rs               # Docker container lifecycle (Tier 2 backends)
    usearch.rs              # retri-eval-usearch binary
    faiss.rs                # retri-eval-faiss binary
    cuvs.rs                 # retri-eval-cuvs binary
    qdrant.rs               # retri-eval-qdrant binary
    redis.rs                # retri-eval-redis binary
    lancedb.rs              # retri-eval-lancedb binary
    weaviate.rs             # retri-eval-weaviate binary
    generate.rs             # retri-generate — synthetic dataset generator with GT
    perf_counters.rs        # Linux perf_event_open wrapper for hardware counters
docker/
    qdrant.yml              # Docker compose for Qdrant
    redis.yml               # Docker compose for Redis
    weaviate.yml            # Docker compose for Weaviate
scripts/
    plot.py                 # JSON results → PNG plots (Plotly, runnable via uv)
    download_molecules.rs   # retri-download-molecules binary (--features download)
    download_cohere.rs      # retri-download-cohere binary (--features download)
```

## Datasets

BigANN benchmark is a good starting point, if you are searching for large collections of high-dimensional vectors.
Those often come with precomputed ground-truth neighbors, which is handy for recall evaluation.
Datasets below are grouped by scale; only configurations with matching ground truth support recall evaluation.

Most datasets ship as one file per role (base / queries / ground-truth), but larger ones — like [USearchWiki][usearch-wiki] — are split across many `.fbin` shards.
RetriEval accepts shell glob patterns on `--vectors` / `--queries` / `--neighbors` / `--keys`, so a sharded dataset reads exactly like a single-file one: pass `--vectors 'base.shard_*.fbin'`, quoted so the shell doesn't expand it.
Matched shards are natural-sorted (`shard_2.fbin` before `shard_10.fbin`) and validated for consistent dimensionality and scalar format; per-row stride and recall metrics are unchanged versus the single-file path.

### ~1M Scale — Development & Testing

| Dataset                                    | Scalar Type | Dimensions | Metric | Base Size | Ground Truth      |
| :----------------------------------------- | ----------: | ---------: | -----: | --------: | :---------------- |
| [Unum UForm Wiki][unum-wiki-1m]            |       `f32` |        256 |     IP |      1 GB | 100K queries, yes |
| [Unum UForm Creative Captions][unum-cc-3m] |       `f32` |        256 |     IP |      3 GB | 3M queries, yes   |
| [Arxiv with E5][unum-arxiv-2m]             |       `f32` |        768 |     IP |      6 GB | 2M queries, yes   |

### ~10M Scale

| Dataset                              | Scalar Type | Dimensions |  Metric | Base Size | Ground Truth      |
| :----------------------------------- | ----------: | ---------: | ------: | --------: | :---------------- |
| [Meta BIGANN — SIFT][meta-bigann]    |        `u8` |        128 |      L2 |    1.2 GB | 10K queries, yes  |
| [Microsoft Turing-ANNS][msft-turing] |       `f32` |        100 |      L2 |    3.7 GB | 100K queries, yes |
| [Cohere Wiki EN][cohere-wiki]        |        `b1` |       1024 | Hamming |    5.3 GB | self-sampled ¹    |

> ¹ Binary fingerprint and embedding sources ship vectors but no ground truth.
> The `retri-download-molecules` and `retri-download-cohere` binaries behind `--features download` fetch the Parquet shards from S3 and Hugging Face, extract the bit-packed column straight into `.b1bin`, sample queries with a fixed seed, and compute exact brute-force Hamming top-K using NumKong's SIMD kernels.

### ~100M Scale

| Dataset                               | Scalar Type | Dimensions |  Metric | Base Size | Ground Truth      |
| :------------------------------------ | ----------: | ---------: | ------: | --------: | :---------------- |
| [Meta BIGANN — SIFT][meta-bigann]     |        `u8` |        128 |      L2 |     12 GB | 10K queries, yes  |
| [Microsoft Turing-ANNS][msft-turing]  |       `f32` |        100 |      L2 |     37 GB | 100K queries, yes |
| [Microsoft SpaceV][msft-spacev]       |        `i8` |        100 |      L2 |    9.3 GB | 30K queries, yes  |
| [Unum WikiVerse][wikiverse] ²         |       `f16` |  128–4096³ |  Cos/IP |  95-505GB | pipeline pending  |
| [USearchMolecules PubChem][usm] MACCS |        `b1` |        168 | Hamming |    2.4 GB | self-sampled ¹    |
| [USearchMolecules PubChem][usm] ECFP4 |        `b1` |       2048 | Hamming |     29 GB | self-sampled ¹    |

> ² WikiVerse uses `.f16bin` (`u32` rows + `u32` cols + `f16` values), which RetriEval does not yet read — adding `f16` to `Dataset::load`'s extension match is a small follow-up.
> ³ Per-model: nomic-embed 768, arctic-embed/Qwen3 1024, e5-mistral 4096; ColBERT-style multi-vector (128d/token) needs the deferred multi-vector plan.

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

[unum-cc-3m]: https://huggingface.co/datasets/unum-cloud/ann-cc-3m
[unum-wiki-1m]: https://huggingface.co/datasets/unum-cloud/ann-wiki-1m
[unum-arxiv-2m]: https://huggingface.co/datasets/unum-cloud/ann-arxiv-2m
[msft-spacev]: https://github.com/ashvardanian/SpaceV
[msft-turing]: https://learning2hash.github.io/publications/microsoftturinganns1B/
[yandex-t2i]: https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search
[yandex-deep]: https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search
[meta-bigann]: https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/
[usm]: https://github.com/ashvardanian/USearchMolecules
[cohere-wiki]: https://huggingface.co/datasets/CohereLabs/wikipedia-2023-11-embed-multilingual-v3-int8-binary
[wikiverse]: https://huggingface.co/datasets/unum-cloud/WikiVerse
[usearch-wiki]: https://github.com/unum-cloud/USearchWiki

### Unum UForm Wiki

Image-and-text embeddings from the UForm small multimodal model, projected to a shared 256d space.
Bench against IP since UForm is L2-normalised at training time.

<details>
<summary>1M — f32, 256d, IP, ~1 GB</summary>

```sh
mkdir -p datasets/wiki_1M/ && \
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-wiki-1m/resolve/main/base.1M.fbin -P datasets/wiki_1M/ && \
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-wiki-1m/resolve/main/query.public.100K.fbin -P datasets/wiki_1M/ && \
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-wiki-1m/resolve/main/groundtruth.public.100K.ibin -P datasets/wiki_1M/
```

```sh
retri-eval-usearch \
    --vectors datasets/wiki_1M/base.1M.fbin \
    --queries datasets/wiki_1M/query.public.100K.fbin \
    --neighbors datasets/wiki_1M/groundtruth.public.100K.ibin \
    --data-type f32,f16,i8 --metric ip \
    --output results/wiki_1M
```

</details>

### Unum UForm Creative Captions

Conceptual Captions image embeddings from the same UForm model as Wiki.
Ground truth was computed offline by shuffling the base set as queries and recording the top-100 IP neighbors per row — see `scripts/compute_unum_orphan_gt.py`.

<details>
<summary>3M — f32, 256d, IP, ~3 GB</summary>

```sh
mkdir -p datasets/cc_3M/ && \
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-cc-3m/resolve/main/base.fbin -P datasets/cc_3M/ && \
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-cc-3m/resolve/main/query.fbin -P datasets/cc_3M/ && \
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-cc-3m/resolve/main/groundtruth.ibin -P datasets/cc_3M/
```

```sh
retri-eval-usearch \
    --vectors datasets/cc_3M/base.fbin \
    --queries datasets/cc_3M/query.fbin \
    --neighbors datasets/cc_3M/groundtruth.ibin \
    --data-type f32,bf16,f16,i8 --metric ip \
    --output results/cc_3M
```

</details>

### Arxiv with E5

Arxiv abstracts embedded with the `intfloat/e5-base` model.
Same offline GT recipe as Creative Captions: shuffled base as queries, top-100 IP neighbors.

<details>
<summary>2M — f32, 768d, IP, ~6 GB</summary>

```sh
mkdir -p datasets/arxiv_2M/ && \
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-arxiv-2m/resolve/main/base.fbin -P datasets/arxiv_2M/ && \
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-arxiv-2m/resolve/main/query.fbin -P datasets/arxiv_2M/ && \
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-arxiv-2m/resolve/main/groundtruth.ibin -P datasets/arxiv_2M/
```

```sh
retri-eval-usearch \
    --vectors datasets/arxiv_2M/base.fbin \
    --queries datasets/arxiv_2M/query.fbin \
    --neighbors datasets/arxiv_2M/groundtruth.ibin \
    --data-type f32,bf16,f16,i8 --metric ip \
    --output results/arxiv_2M
```

</details>

### Meta BIGANN — SIFT

Billion-scale SIFT descriptors from Meta.
No pre-sliced subset base files exist, so the recipes use range requests against the single 1B file followed by an in-place header patch to update the vector count.
Pre-computed ground truth is available for 10M and 100M subsets.

<details>
<summary>10M — u8, 128d, L2, ~1.2 GB</summary>

```sh
mkdir -p datasets/sift_10M/ && \
    wget -nc https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/query.public.10K.u8bin -P datasets/sift_10M/ && \
    wget -nc https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_10M/bigann-10M -O datasets/sift_10M/groundtruth.public.10K.ibin && \
    wget --header="Range: bytes=0-1280000007" \
        https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin \
        -O datasets/sift_10M/base.10M.u8bin && \
    python3 -c "
import struct
with open('datasets/sift_10M/base.10M.u8bin', 'r+b') as f:
    f.write(struct.pack('I', 10_000_000))
"
```

```sh
retri-eval-usearch \
    --vectors datasets/sift_10M/base.10M.u8bin \
    --queries datasets/sift_10M/query.public.10K.u8bin \
    --neighbors datasets/sift_10M/groundtruth.public.10K.ibin \
    --data-type f32,f16,i8 --metric l2 \
    --output results/sift_10M
```

</details>

<details>
<summary>100M — u8, 128d, L2, ~12 GB</summary>

```sh
mkdir -p datasets/sift_100M/ && \
    wget -nc https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/query.public.10K.u8bin -P datasets/sift_100M/ && \
    wget -nc https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_100M/bigann-100M -O datasets/sift_100M/groundtruth.public.10K.ibin && \
    wget --header="Range: bytes=0-12800000007" \
        https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin \
        -O datasets/sift_100M/base.100M.u8bin && \
    python3 -c "
import struct
with open('datasets/sift_100M/base.100M.u8bin', 'r+b') as f:
    f.write(struct.pack('I', 100_000_000))
"
```

```sh
retri-eval-usearch \
    --vectors datasets/sift_100M/base.100M.u8bin \
    --queries datasets/sift_100M/query.public.10K.u8bin \
    --neighbors datasets/sift_100M/groundtruth.public.10K.ibin \
    --data-type f32,f16,i8 --metric l2 \
    --epochs 20 --output results/sift_100M
```

</details>

### Microsoft Turing-ANNS

373 GB of f32 vectors with 100 dimensions at full 1B scale.
Subsets follow the same range-request + header-patch recipe as BIGANN.
Pre-computed ground truth is available for 1M, 10M, and 100M.

<details>
<summary>1M — f32, 100d, L2, ~400 MB</summary>

```sh
mkdir -p datasets/turing_1M/ && \
    wget -nc https://comp21storage.z5.web.core.windows.net/comp21/MSFT-TURING-ANNS/query100K.fbin \
        -O datasets/turing_1M/query.public.100K.fbin && \
    wget -nc https://comp21storage.z5.web.core.windows.net/comp21/MSFT-TURING-ANNS/msturing-gt-1M \
        -O datasets/turing_1M/groundtruth.public.100K.ibin && \
    wget --header="Range: bytes=0-400000007" \
        https://comp21storage.z5.web.core.windows.net/comp21/MSFT-TURING-ANNS/base1b.fbin \
        -O datasets/turing_1M/base.1M.fbin && \
    python3 -c "
import struct
with open('datasets/turing_1M/base.1M.fbin', 'r+b') as f:
    f.write(struct.pack('I', 1_000_000))
"
```

```sh
retri-eval-usearch \
    --vectors datasets/turing_1M/base.1M.fbin \
    --queries datasets/turing_1M/query.public.100K.fbin \
    --neighbors datasets/turing_1M/groundtruth.public.100K.ibin \
    --data-type f32,bf16,f16,i8 --metric l2 \
    --output results/turing_1M
```

</details>

<details>
<summary>10M — f32, 100d, L2, ~3.7 GB</summary>

```sh
mkdir -p datasets/turing_10M/ && \
    wget -nc https://comp21storage.z5.web.core.windows.net/comp21/MSFT-TURING-ANNS/query100K.fbin \
        -O datasets/turing_10M/query.public.100K.fbin && \
    wget -nc https://comp21storage.z5.web.core.windows.net/comp21/MSFT-TURING-ANNS/msturing-gt-10M \
        -O datasets/turing_10M/groundtruth.public.100K.ibin && \
    wget --header="Range: bytes=0-4000000007" \
        https://comp21storage.z5.web.core.windows.net/comp21/MSFT-TURING-ANNS/base1b.fbin \
        -O datasets/turing_10M/base.10M.fbin && \
    python3 -c "
import struct
with open('datasets/turing_10M/base.10M.fbin', 'r+b') as f:
    f.write(struct.pack('I', 10_000_000))
"
```

```sh
retri-eval-usearch \
    --vectors datasets/turing_10M/base.10M.fbin \
    --queries datasets/turing_10M/query.public.100K.fbin \
    --neighbors datasets/turing_10M/groundtruth.public.100K.ibin \
    --data-type f32,bf16,f16,i8 --metric l2 \
    --output results/turing_10M
```

</details>

<details>
<summary>100M — f32, 100d, L2, ~37 GB</summary>

```sh
mkdir -p datasets/turing_100M/ && \
    wget -nc https://comp21storage.z5.web.core.windows.net/comp21/MSFT-TURING-ANNS/query100K.fbin \
        -O datasets/turing_100M/query.public.100K.fbin && \
    wget -nc https://comp21storage.z5.web.core.windows.net/comp21/MSFT-TURING-ANNS/msturing-gt-100M \
        -O datasets/turing_100M/groundtruth.public.100K.ibin && \
    wget --header="Range: bytes=0-40000000007" \
        https://comp21storage.z5.web.core.windows.net/comp21/MSFT-TURING-ANNS/base1b.fbin \
        -O datasets/turing_100M/base.100M.fbin && \
    python3 -c "
import struct
with open('datasets/turing_100M/base.100M.fbin', 'r+b') as f:
    f.write(struct.pack('I', 100_000_000))
"
```

```sh
retri-eval-usearch \
    --vectors datasets/turing_100M/base.100M.fbin \
    --queries datasets/turing_100M/query.public.100K.fbin \
    --neighbors datasets/turing_100M/groundtruth.public.100K.ibin \
    --data-type f32,bf16,f16,i8 --metric l2 \
    --epochs 20 --output results/turing_100M
```

</details>

### Microsoft SpaceV

Web-search embeddings already quantised to int8 at the source.
A 100M subset is mirrored on Hugging Face; the original 1B lives on AWS S3.

<details>
<summary>100M — i8, 100d, L2, ~9.3 GB</summary>

```sh
mkdir -p datasets/spacev_100M/ && \
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-spacev-100m/resolve/main/base.100M.i8bin -P datasets/spacev_100M/ && \
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-spacev-100m/resolve/main/query.30K.i8bin -P datasets/spacev_100M/ && \
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-spacev-100m/resolve/main/groundtruth.30K.i32bin -P datasets/spacev_100M/
```

```sh
retri-eval-usearch \
    --vectors datasets/spacev_100M/base.100M.i8bin \
    --queries datasets/spacev_100M/query.30K.i8bin \
    --neighbors datasets/spacev_100M/groundtruth.30K.i32bin \
    --data-type f32,f16,i8 --metric l2 \
    --epochs 20 --output results/spacev_100M
```

</details>

### Yandex Deep

Image embeddings extracted from the GoogLeNet penultimate layer.
Only the full 1B is included here — the smaller subsets duplicate the same distribution at scales already covered by other datasets.

<details>
<summary>1B — f32, 96d, L2, ~358 GB</summary>

```sh
mkdir -p datasets/deep_1B/ && \
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin -P datasets/deep_1B/ && \
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin -P datasets/deep_1B/ && \
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/groundtruth.public.10K.ibin -P datasets/deep_1B/
```

</details>

### Yandex Text-to-Image

Cross-modal text-and-image embeddings benchmarked under cosine similarity.

<details>
<summary>1M — f32, 200d, Cos, ~750 MB</summary>

```sh
mkdir -p datasets/t2i/ && \
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.1M.fbin -P datasets/t2i/ && \
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.public.100K.fbin -P datasets/t2i/ && \
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/groundtruth.public.100K.ibin -P datasets/t2i/
```

```sh
retri-eval-usearch \
    --vectors datasets/t2i/base.1M.fbin \
    --queries datasets/t2i/query.public.100K.fbin \
    --neighbors datasets/t2i/groundtruth.public.100K.ibin \
    --data-type f32,bf16,f16,i8 --metric cos \
    --output results/t2i_1M
```

</details>

<details>
<summary>1B — f32, 200d, Cos, ~750 GB</summary>

```sh
mkdir -p datasets/t2i_1B/ && \
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.1B.fbin -P datasets/t2i_1B/ && \
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.public.100K.fbin -P datasets/t2i_1B/ && \
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/groundtruth.public.100K.ibin -P datasets/t2i_1B/
```

</details>

### USearchMolecules

A corpus of small molecules with pre-computed binary fingerprints at four widths: MACCS 166 bits, PubChem 881 bits, ECFP4 2048 bits, FCFP4 2048 bits.
Three subsets are hosted on AWS Open Data as Parquet shards: PubChem at 115M molecules, GDB-13 at 977M, and Enamine REAL at 6.04B.
Natural fit for Hamming and Jaccard benchmarks since the vectors are genuinely binary rather than quantised floats.

The `retri-download-molecules` binary fetches the requested fingerprint column directly into `.b1bin`, samples queries with a fixed seed, and computes brute-force Hamming top-K ground truth.
Use `--limit N` to take a subset and `--source {pubchem,gdb13,enamine}` to pick the scale.

<details>
<summary>PubChem 115M MACCS — b1, 168 bits, Hamming, ~2.4 GB</summary>

```sh
cargo install --path . --features download
retri-download-molecules \
    --source pubchem --fingerprint maccs \
    --query-count 10000 --neighbors 10 \
    --output datasets/pubchem_maccs/
```

```sh
retri-eval-usearch \
    --vectors datasets/pubchem_maccs/base.115627267.b1bin \
    --queries datasets/pubchem_maccs/query.10000.b1bin \
    --neighbors datasets/pubchem_maccs/groundtruth.10000.ibin \
    --data-type b1 --metric hamming,jaccard \
    --output results/pubchem_maccs
```

</details>

<details>
<summary>PubChem 115M ECFP4 — b1, 2048 bits, Hamming, ~29 GB</summary>

```sh
retri-download-molecules \
    --source pubchem --fingerprint ecfp4 \
    --query-count 10000 --neighbors 10 \
    --output datasets/pubchem_ecfp4/
```

```sh
retri-eval-usearch \
    --vectors datasets/pubchem_ecfp4/base.115627267.b1bin \
    --queries datasets/pubchem_ecfp4/query.10000.b1bin \
    --neighbors datasets/pubchem_ecfp4/groundtruth.10000.ibin \
    --data-type b1 --metric hamming \
    --output results/pubchem_ecfp4
```

</details>

<details>
<summary>GDB-13 977M MACCS — b1, 168 bits, Hamming, ~21 GB</summary>

```sh
retri-download-molecules \
    --source gdb13 --fingerprint maccs \
    --query-count 10000 --neighbors 10 \
    --output datasets/gdb13_maccs/
```

</details>

<details>
<summary>Enamine REAL 6.04B MACCS — b1, 168 bits, Hamming, ~127 GB</summary>

```sh
retri-download-molecules \
    --source enamine --fingerprint maccs \
    --query-count 10000 --neighbors 10 \
    --output datasets/enamine_maccs/
```

</details>

Substitute `--fingerprint ecfp4` for the 2048-bit variant, which multiplies the base-file size by roughly 12× at each scale.
Ground-truth time dominates at billion scale; set `--batch-size` explicitly if you have a lot of RAM and want larger query batches.

### Cohere Wikipedia Multilingual

247M Wikipedia paragraphs embedded with Cohere Embed v3 and bit-packed into 1024-bit `emb_ubinary` columns at 128 bytes per vector.
The dataset also ships text metadata — title, paragraph body, URL — alongside the vectors.
`--with-text` extracts them into aligned newline-delimited files for downstream semantic-search demos.

<details>
<summary>English subset 41.5M — b1, 1024 bits, Hamming, ~5.3 GB</summary>

```sh
retri-download-cohere \
    --language en \
    --query-count 10000 --neighbors 10 \
    --output datasets/cohere_en/
```

```sh
retri-eval-usearch \
    --vectors datasets/cohere_en/base.41488110.b1bin \
    --queries datasets/cohere_en/query.10000.b1bin \
    --neighbors datasets/cohere_en/groundtruth.10000.ibin \
    --data-type b1 --metric hamming \
    --output results/cohere_en
```

FAISS binary indexes via `IndexBinaryHNSW` also work — pass `--data-type b1`, and the metric is Hamming by construction.

```sh
retri-eval-faiss \
    --vectors datasets/cohere_en/base.41488110.b1bin \
    --queries datasets/cohere_en/query.10000.b1bin \
    --neighbors datasets/cohere_en/groundtruth.10000.ibin \
    --data-type b1 --metric hamming \
    --output results/cohere_en_faiss
```

</details>

### Unum WikiVerse

Multi-model embedding dataset built on [HuggingFaceFW/finewiki][finewiki] — 61.5M articles across 325 languages, embedded by five models (Qwen3-Embedding-0.6B 1024d, GTE-ModernColBERT-v1 128d/token, Snowflake arctic-embed-l-v2.0 1024d, nomic-embed-text-v1.5 768d, e5-mistral-7b-instruct 4096d).
Each `.f16bin` shard is row-aligned with the source FineWiki parquet — directory layout is `<model>/<lang>wiki/<group>_<shard>.{body,title}.f16bin`, mirroring FineWiki 1:1.
The full corpus is 95-505 GB depending on the model; ColBERT-style embeddings reach 6.2 TB at one vector per token.

Two prerequisites are still pending on the RetriEval side: `Dataset::load` doesn't yet recognize the `.f16bin` extension (a small follow-up — same header layout as `.fbin`, swap `f32` for `f16` in `ScalarFormat`), and the ColBERT model needs the deferred multi-vector plan.
The dense models will work as soon as `f16` lands; the example below assumes that, plus the existing GLOB support for sharded inputs.

<details>
<summary>English subset, Qwen3-Embedding-0.6B — f16, 1024d, ~13 GB</summary>

```sh
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/unum-cloud/WikiVerse datasets/wikiverse/
cd datasets/wikiverse
hf download unum-cloud/WikiVerse \
    --repo-type dataset \
    --include "qwen3-embedding-0.6b/enwiki/*.body.f16bin"
cd ../..
```

```sh
retri-eval-usearch \
    --vectors 'datasets/wikiverse/qwen3-embedding-0.6b/enwiki/*.body.f16bin' \
    --queries datasets/wikiverse/qwen3-embedding-0.6b/enwiki/000_00000.body.f16bin \
    --data-type f16 --metric cos \
    --output results/wikiverse_en_qwen3
```

The `--vectors` glob picks up every English shard in natural-sort order; queries reuse one shard until the official query/GT split lands upstream.

</details>

[finewiki]: https://huggingface.co/datasets/HuggingFaceFW/finewiki
