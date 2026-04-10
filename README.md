# RetriEval

Benchmark suite for vector search engines, written to be as fast as the underlying engines.
Compares in-memory ANN libraries (USearch, FAISS) and vector databases (Qdrant, Redis, Weaviate, LanceDB) on [BigANN](https://big-ann-benchmarks.com/) and similar datasets.

## Quick Start

```sh
cargo build --release
```

Each backend is a separate binary. Run USearch against a dataset:

```sh
./target/release/retri-eval-usearch \
    --vectors datasets/wiki_1M/base.1M.fbin \
    --queries datasets/wiki_1M/query.public.100K.fbin \
    --neighbors datasets/wiki_1M/groundtruth.public.100K.ibin \
    --dtype f32,f16,i8 \
    --metric ip \
    --output results/
```

Generate plots from the results:

```sh
uv run scripts/plot.py results/ --output-dir plots/
```

## Datasets

BigANN benchmark datasets with precomputed ground-truth neighbors.
Only configurations with matching ground truth support recall evaluation.

### Unum UForm Wiki — 1M, f32, 256d, IP

```sh
mkdir -p datasets/wiki_1M/ && \
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-wiki-1m/resolve/main/base.1M.fbin -P datasets/wiki_1M/ && \
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-wiki-1m/resolve/main/query.public.100K.fbin -P datasets/wiki_1M/ && \
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-wiki-1m/resolve/main/groundtruth.public.100K.ibin -P datasets/wiki_1M/
```

```sh
./target/release/retri-eval-usearch \
    --vectors datasets/wiki_1M/base.1M.fbin \
    --queries datasets/wiki_1M/query.public.100K.fbin \
    --neighbors datasets/wiki_1M/groundtruth.public.100K.ibin \
    --dtype f32,f16,i8 --metric ip --threads 16 \
    --output wiki-1M.jsonl
```

### Meta BIGANN — SIFT

The full 1B dataset is available from Meta. No pre-sliced subset base files exist, so range requests are
used to download only the first N vectors, followed by a header patch to update the vector count.
Pre-computed ground truth is available for 10M and 100M subsets.

#### 10M subset, u8, 128d, L2, ~1.2 GB

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
./target/release/retri-eval-usearch \
    --vectors datasets/sift_10M/base.10M.u8bin \
    --queries datasets/sift_10M/query.public.10K.u8bin \
    --neighbors datasets/sift_10M/groundtruth.public.10K.ibin \
    --dtype f32,f16,i8 --metric l2 --threads 16 \
    --output sift-10M.jsonl
```

#### 100M subset, u8, 128d, L2, ~12 GB

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
./target/release/retri-eval-usearch \
    --vectors datasets/sift_100M/base.100M.u8bin \
    --queries datasets/sift_100M/query.public.10K.u8bin \
    --neighbors datasets/sift_100M/groundtruth.public.10K.ibin \
    --dtype f32,f16,i8 --metric l2 --threads 96 \
    --step-size 5000000 --output sift-100M.jsonl
```

### Microsoft Turing-ANNS

The full 1B dataset is ~373 GB of f32 vectors with 100 dimensions.
Subsets can be obtained via range requests, followed by a header patch to update the vector count.
Pre-computed ground truth is available for 1M, 10M, and 100M subsets.

#### 1M subset, f32, 100d, L2, ~400 MB

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
./target/release/retri-eval-usearch \
    --vectors datasets/turing_1M/base.1M.fbin \
    --queries datasets/turing_1M/query.public.100K.fbin \
    --neighbors datasets/turing_1M/groundtruth.public.100K.ibin \
    --dtype f32,bf16,f16,i8 --metric l2 --threads 16 \
    --output turing-1M.jsonl
```

#### 10M subset, f32, 100d, L2, ~3.7 GB

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
./target/release/retri-eval-usearch \
    --vectors datasets/turing_10M/base.10M.fbin \
    --queries datasets/turing_10M/query.public.100K.fbin \
    --neighbors datasets/turing_10M/groundtruth.public.100K.ibin \
    --dtype f32,bf16,f16,i8 --metric l2 --threads 16 \
    --output turing-10M.jsonl
```

#### 100M subset, f32, 100d, L2, ~37 GB

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
./target/release/retri-eval-usearch \
    --vectors datasets/turing_100M/base.100M.fbin \
    --queries datasets/turing_100M/query.public.100K.fbin \
    --neighbors datasets/turing_100M/groundtruth.public.100K.ibin \
    --dtype f32,bf16,f16,i8 --metric l2 --threads 96 \
    --step-size 5000000 --output turing-100M.jsonl
```

### Microsoft SpaceV

A 100M subset is available from Hugging Face. The original 1B dataset can be pulled from AWS S3.

#### 100M subset, i8, 100d, L2, ~9.3 GB

```sh
mkdir -p datasets/spacev_100M/ && \
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-spacev-100m/resolve/main/base.100M.i8bin -P datasets/spacev_100M/ && \
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-spacev-100m/resolve/main/query.30K.i8bin -P datasets/spacev_100M/ && \
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-spacev-100m/resolve/main/groundtruth.30K.i32bin -P datasets/spacev_100M/
```

```sh
./target/release/retri-eval-usearch \
    --vectors datasets/spacev_100M/base.100M.i8bin \
    --queries datasets/spacev_100M/query.30K.i8bin \
    --neighbors datasets/spacev_100M/groundtruth.30K.i32bin \
    --dtype f32,f16,i8 --metric l2 --threads 96 \
    --step-size 5000000 --output spacev-100M.jsonl
```

### Yandex Deep

Pre-built 10M subset and full 1B available from Yandex.

#### 10M subset, f32, 96d, L2, ~3.6 GB

```sh
mkdir -p datasets/deep_10M/ && \
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.10M.fbin -P datasets/deep_10M/ && \
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin -P datasets/deep_10M/ && \
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/groundtruth.public.10K.ibin -P datasets/deep_10M/
```

```sh
./target/release/retri-eval-usearch \
    --vectors datasets/deep_10M/base.10M.fbin \
    --queries datasets/deep_10M/query.public.10K.fbin \
    --neighbors datasets/deep_10M/groundtruth.public.10K.ibin \
    --dtype f32,bf16,f16,i8 --metric l2 --threads 16 \
    --output deep-10M.jsonl
```

#### 1B, f32, 96d, L2, ~358 GB

```sh
mkdir -p datasets/deep_1B/ && \
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin -P datasets/deep_1B/ && \
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin -P datasets/deep_1B/ && \
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/groundtruth.public.10K.ibin -P datasets/deep_1B/
```

### Yandex Text-to-Image

1M subset and full 1B available from Yandex.

#### 1M subset, f32, 200d, Cos, ~750 MB

```sh
mkdir -p datasets/t2i/ && \
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.1M.fbin -P datasets/t2i/ && \
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.public.100K.fbin -P datasets/t2i/ && \
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/groundtruth.public.100K.ibin -P datasets/t2i/
```

```sh
./target/release/retri-eval-usearch \
    --vectors datasets/t2i/base.1M.fbin \
    --queries datasets/t2i/query.public.100K.fbin \
    --neighbors datasets/t2i/groundtruth.public.100K.ibin \
    --dtype f32,bf16,f16,i8 --metric cos --threads 16 \
    --output t2i-1M.jsonl
```

#### 1B, f32, 200d, Cos, ~750 GB

```sh
mkdir -p datasets/t2i_1B/ && \
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.1B.fbin -P datasets/t2i_1B/ && \
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.public.100K.fbin -P datasets/t2i_1B/ && \
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/groundtruth.public.100K.ibin -P datasets/t2i_1B/
```

## Backends

### Tier 1 — In-Memory ANN Libraries

| Backend     | Quantization                                       | Metrics                | Threading                    |
| ----------- | -------------------------------------------------- | ---------------------- | ---------------------------- |
| __USearch__ | f32, bf16, f16, e5m2, e4m3, e3m2, e2m3, i8, u8, b1 | ip, l2sq, cos, hamming | fork_union pool per instance |
| __FAISS__   | f32, f16, bf16, u8, i8, b1                         | ip, l2sq               | OpenMP (omp_set_num_threads) |

### Tier 2 — Vector Databases

| Backend      | Client               | Docker Image              |
| ------------ | -------------------- | ------------------------- |
| __Qdrant__   | qdrant-client (gRPC) | qdrant/qdrant             |
| __Redis__    | redis (RediSearch)   | redis/redis-stack         |
| __Weaviate__ | weaviate-community   | semitechnologies/weaviate |
| __LanceDB__  | lancedb (in-process) | —                         |

Tier 2 backends are behind feature flags:

```sh
cargo build --release --features qdrant-backend,redis-backend,lancedb-backend,weaviate-backend
```

## CLI Reference

Each backend is a separate binary. Common flags shared by all:

```
--vectors <PATH>           # Base vectors (.fbin, .u8bin, .i8bin)
--queries <PATH>           # Query vectors
--neighbors <PATH>         # Ground-truth neighbors (.ibin)
--keys <PATH>              # Optional keys file (.i32bin)
--step-size <N>            # Vectors per measurement step (default: 1000000)
--no-shuffle               # Disable random insertion order (shuffle is on by default)
--output <DIR>             # Output directory for JSON result files (omit for progress-only)
```

__retri-eval-usearch__ additionally supports comma-separated sweeps:

```
--dtype <LIST>             # f32,f16,bf16,e5m2,e4m3,e3m2,e2m3,i8,u8,b1
--metric <LIST>            # ip, l2, cos, hamming, jaccard, sorensen, pearson, haversine, divergence
--connectivity <LIST>      # HNSW M parameter (default: 16)
--expansion-add <LIST>     # efConstruction (default: 128)
--expansion-search <LIST>  # efSearch (default: 64)
--shards <LIST>            # Index shards (default: 1)
--threads <LIST>           # Thread count (default: available cores)
```

## Output Format

One JSON file per backend configuration, written to `--output <dir>`.
Files are auto-named `<backend>-<hash>.json`.

```json
{
  "machine": { "cpu_model": "Intel Xeon 6776P", "physical_cores": 96, ... },
  "dataset": { "vectors_path": "...", "vectors_count": 10000000, "dimensions": 100, ... },
  "config": { "backend": "usearch", "dtype": "f32", "metric": "l2", "connectivity": 16, ... },
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
    bench.rs            # Library root: Backend trait, types, BenchState, benchmark loop
    dataset.rs          # Memory-mapped .fbin/.ibin loading (zero-copy)
    eval.rs             # Recall@K, NDCG@K
    output.rs           # Report types, JSON writer, machine info
    docker.rs           # Docker container lifecycle (Tier 2 backends)
    usearch.rs          # retri-eval-usearch binary
    faiss.rs            # retri-eval-faiss binary
    qdrant.rs           # retri-eval-qdrant binary
    redis.rs            # retri-eval-redis binary
    lancedb.rs          # retri-eval-lancedb binary
    weaviate.rs         # retri-eval-weaviate binary
docker/
    qdrant.yml          # Docker compose for Qdrant
    redis.yml           # Docker compose for Redis Stack
    weaviate.yml        # Docker compose for Weaviate
scripts/
    plot.py             # JSON results → PNG plots (Plotly, runnable via uv)
```
