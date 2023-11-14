# USearch Benchmarks

This set of benchmarks is meant to test USearch capabilities for Billion-scale vector search.
It provides an alternative to the `ann-benchmarks` and the `big-ann-benchmarks` which generally operate on much smaller collections.

The main objective is to understand the scaling laws of the USearch compared to [FAISS](https://github.com/facebookresearch/faiss).
Supplementary adapters for other popular systems is also available under `index/` directory:

- Alternative HNSW implementations, like HNSWlib,
- Alternative CPU-based libraries, like SCANN,
- Vector Databases, like Qdrant, and Wevaite.

The primary dataset used for benchmarks is the [Deep1B](https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search) dataset of 1 Billion 96-dimensional vectors, totalling at __384 GB__.
Ground-truth nearest neighbors are provided to calculate the recall metrics.

## Setup

First of all, we recommend creating a `conda` environment to isolate the dependencies:

```sh
conda create -n usearch-benchmarks python=3.10
conda activate usearch-benchmarks
```

Then install dependencies, getting an MKL-accelerated version of FAISS library.

```sh
pip install usearch hnswlib scann lancedb qdrant-client weaviate-client psutil plotly kaleido
conda install -c pytorch faiss-cpu=1.7.4 mkl=2021 blas=1.0=mkl
```

To benchmark Qdrant, you need to run their Docker container:

```sh
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

Finally, download the [Deep1B](https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search) dataset:

```sh
wget https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin -P data
```

To run the ANN benchmarks pass a configuration file:

```sh
python run.py configs/usearch_1B.json 1B # Outputs stats/*.npz file
python utils/draw_plots.py # Exports tp plots/*.png
```
