# USearch Benchmarks

## Setup

First of all, create conda environment:

```sh
conda create -n usearch-benchmarks python=3.10
conda activate usearch-benchmarks
```

Then install dependencies:
```sh
sh prepare_env.sh
```

You need to launch server inside docker container for Qdrant benchmarking:

```sh
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

Finally, download [DEEP1B](https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search) vectors (warning: the size of file is __384GB__):

```sh
wget https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin -P data
```

## Benchmark & Plotting

To run ANN benchmarks, run:

```sh
sh main.sh
```

You will afterwards find the charts in the `plots/` subdirectory.

If you want to test USearch and FAISS on the whole dataset, you need to run these commands:

```sh
python run.py configs/usearch_faiss_1B.json 1B
python utils/draw_plots.py
```