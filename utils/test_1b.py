from time import perf_counter

import numpy as np
from usearch.index import Index
from usearch.io import load_matrix
from tqdm import tqdm
import plotly.graph_objects as go

from utils.metrics import recall

INDEX_PATHS = (
    "../tmp/USearch-HNSW-f32.usearch",
    "../tmp/USearch-HNSW-i8.usearch",
    "../tmp/USearch-HNSW-f16",
)

INDEX_NAMES = (
    "USearch(HNSW,f32)",
    "USearch(HNSW,f16)",
    "USearch(HNSW,i8)",
)

DATA_PATH = "../data/base.1B.fbin"

QUERY_SIZES = (10_000, 50_000, 100_000, 500_000, 1000_000, 5000_000, 10_000_000)


def measure(index_path, seed=42, k=10):
    index = Index.restore(index_path, view=True)
    print(index)

    index.search(np.random.random((5000_000, 96)), 10)

    np.random.seed(seed)
    search_speed = []
    recalls = []

    for size in tqdm(QUERY_SIZES):
        start_idx = np.random.randint(0, len(index) - size)
        groundtruth = np.arange(start_idx, start_idx + size)
        query = load_matrix(DATA_PATH, start_idx, size, view=True)
        query = query / np.linalg.norm(query, axis=1, keepdims=True)

        start_time = perf_counter()
        nn_ids = index.search(query, k).keys
        spent_time = perf_counter() - start_time
        search_speed.append(size / spent_time)

        recall_at_k = float(recall(nn_ids, groundtruth, ats=(k,))[0])
        recalls.append(recall_at_k)

    return search_speed, recalls


def draw_plot(xs, ys, methods, x_title, y_title, title):
    graphs = []

    for x, y, method in zip(xs, ys, methods):
        if "USearch" not in method:
            graphs.append(
                go.Scatter(
                    x=x,
                    y=y,
                    name=method,
                    mode="lines",
                    line=dict(dash="dash", width=3),
                    marker=dict(size=5),
                )
            )
        else:
            graphs.append(go.Scatter(x=x, y=y, name=method, line=dict(width=3)))

    fig = go.Figure(
        data=graphs,
        layout={
            "xaxis": {"title": x_title},
            "yaxis": {"title": y_title},
            "title": {"text": title},
            "width": 1200,
            "legend": {
                "xanchor": "center",
                "x": 0.5,
                "y": 1.1,
                "orientation": "h",
            },
        },
    )
    fig.update_layout(title_x=0.5)

    return fig


if __name__ == "__main__":
    results = {"search_speed": [], "recall": []}

    for index_name, index_path in zip(INDEX_NAMES, INDEX_PATHS):
        search_speed, recalls = measure(index_path)
        results["search_speed"].append(search_speed)
        results["recall"].append(recalls)

    np.savez(f"stats/{index_name}_1B.npz", **results)

    fig = draw_plot(
        [QUERY_SIZES] * len(INDEX_PATHS),
        results["search_speed"],
        INDEX_NAMES,
        "num of queries",
        "vecs/s",
        "DEEP 1B: Search Speed",
    )
    fig.write_image("../plots/search_speed_1B.png", scale=2)

    fig = draw_plot(
        [QUERY_SIZES] * len(INDEX_PATHS),
        results["recall"],
        INDEX_NAMES,
        "num of queries",
        "recall",
        "DEEP 1B: Recall@10",
    )
    fig.write_image("../plots/recall_1B.png", scale=2)
