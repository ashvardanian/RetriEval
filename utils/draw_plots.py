from os.path import exists

import plotly.graph_objects as go
import numpy as np


STEP_SIZE = 100_000
SEARCH_SIZE = 10_000

INDEX_NAMES = (
    "USearch-HNSW-f32",
    "USearch-HNSW-f16",
    "USearch-HNSW-i8",
    "FAISS-HNSW-f32",
    "FAISS-HNSW-f16",
    "FAISS-HNSW-i8",
    "HNSWLIB",
    "SCANN",
    "Weaviate",
    "Qdrant",
)

EXACT_INDEX_NAMES = (
    "USearch-Exact-f32",
    "USearch-Exact-f16",
    "USearch-Exact-i8",
    "LanceDB",
    "FAISS-Exact",
)

INDEX_NAMES_1B96D = (
    "USearch-HNSW-f32-1B",
    "USearch-HNSW-f16-1B",
    "USearch-HNSW-i8-1B",
    "FAISS-HNSW-f32-1B",
    "FAISS-HNSW-f16-1B",
    "FAISS-HNSW-i8-1B",
)

INDEX_NAMES_100M96D = (
    "USearch-HNSW-f32-100M",
    "USearch-HNSW-f16-100M",
    "USearch-HNSW-i8-100M",
    "FAISS-HNSW-f32-100M",
    "FAISS-HNSW-f16-100M",
    "FAISS-HNSW-i8-100M",
)

INDEX_NAMES_100M1536D = (
    "USearch-HNSW-f32-1536d",
    "USearch-HNSW-f16-1536d",
    "USearch-HNSW-i8-1536d",
    "FAISS-HNSW-f32-1536d",
    "FAISS-HNSW-f16-1536d",
    "FAISS-HNSW-i8-1536d",
)

SLOW_INDEX_NAMES = ("Qdrant", "Weaviate")

NAME_MAPPING = {
    "USearch-HNSW-f32": "USearch, 32-bit float",
    "USearch-HNSW-f16": "USearch, 16-bit float",
    "USearch-HNSW-i8": "USearch, 8-bit int",
    "USearch-HNSW-f32-1B": "USearch, 32-bit float",
    "USearch-HNSW-f16-1B": "USearch, 16-bit float",
    "USearch-HNSW-i8-1B": "USearch, 8-bit int",
    "FAISS-HNSW-f32": "FAISS, 32-bit float",
    "FAISS-HNSW-f16": "FAISS, 16-bit float",
    "FAISS-HNSW-i8": "FAISS, 8-bit int",
    "FAISS-HNSW-f32-1B": "FAISS, 32-bit float",
    "FAISS-HNSW-f16-1B": "FAISS, 16-bit float",
    "FAISS-HNSW-i8-1B": "FAISS, 8-bit int",
    "USearch-HNSW-f32-100M": "USearch, 32-bit float",
    "USearch-HNSW-f16-100M": "USearch, 16-bit float",
    "USearch-HNSW-i8-100M": "USearch, 8-bit int",
    "FAISS-HNSW-f32-100M": "FAISS, 32-bit float",
    "FAISS-HNSW-f16-100M": "FAISS, 16-bit float",
    "FAISS-HNSW-i8-100M": "FAISS, 8-bit int",
    "USearch-HNSW-f32-1536d": "USearch, 32-bit float",
    "USearch-HNSW-f16-1536d": "USearch, 16-bit float",
    "USearch-HNSW-i8-1536d": "USearch, 8-bit int",
    "FAISS-HNSW-f32-1536d": "FAISS, 32-bit float",
    "FAISS-HNSW-f16-1536d": "FAISS, 16-bit float",
    "FAISS-HNSW-i8-1536d": "FAISS, 8-bit int",
    "HNSWLIB": "HNSWLIB",
    "SCANN": "SCANN",
    "Weaviate": "Weaviate",
    "LanceDB": "LanceDB",
    "Qdrant": "Qdrant",
}

STATS_DIR = "stats"
PLOTS_DIR = "plots"


def smooth(x, window_size):
    window = np.ones(window_size) / window_size
    x = np.convolve(x, window, mode="same")
    return x


def draw_plot(
    xs,
    ys,
    x_title,
    y_title,
    methods,
    title,
    log_scale=False,
    smoothing=False,
    display_slow_index=True,
):
    graphs = []

    for x, y, method in zip(xs, ys, methods):
        if not display_slow_index and method in SLOW_INDEX_NAMES:
            continue

        if smoothing:
            y = smooth(y, 10)

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
            "yaxis": {"title": y_title, "type": "log" if log_scale else "linear"},
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


def draw_plots(index_names, prefix="", draw_log_plots=True, ndims=96):
    xs = []
    constuction_speed = []
    constuction_memory = []
    search_speed = []
    search_recall = []
    method = []

    for index_name in index_names:
        path = f"{STATS_DIR}/{index_name}.npz"
        if not exists(path):
            continue

        data = np.load(f"{STATS_DIR}/{index_name}.npz")

        xs.append(
            np.arange(STEP_SIZE, (data["recalls"].shape[0] + 1) * STEP_SIZE, STEP_SIZE)
        )

        constuction_speed.append(STEP_SIZE / data["construction_time"])
        constuction_memory.append(data["memory_consumption"] / 1024 / 1024)
        search_speed.append(SEARCH_SIZE / data["search_time"])
        search_recall.append(data["recalls"])
        method.append(NAME_MAPPING[index_name])

    if len(xs) > 0:
        if draw_log_plots:
            fig = draw_plot(
                xs,
                search_speed,
                "index size, number of vectors",
                "speed, vectors per second",
                method,
                f"Search Speed, {ndims}d Vectors (log scale)",
                log_scale=True,
                smoothing=True,
            )
            fig.write_image(f"{PLOTS_DIR}/{prefix}search_speed_log.png", scale=2)

        fig = draw_plot(
            xs,
            search_speed,
            "index size, number of vectors",
            "speed, vectors per second",
            method,
            f"Search Speed, {ndims}d Vectors",
            log_scale=False,
            smoothing=True,
            display_slow_index=False,
        )
        fig.write_image(f"{PLOTS_DIR}/{prefix}search_speed_linear.png", scale=2)

        fig = draw_plot(
            xs,
            search_recall,
            "index size, number of vectors",
            "accuracy, fraction",
            method,
            f"Top-1 Search Accuracy, {ndims}d Vectors",
        )
        fig.write_image(f"{PLOTS_DIR}/{prefix}search_recall.png", scale=2)

        if draw_log_plots:
            fig = draw_plot(
                xs,
                constuction_speed,
                "index size, number of vectors",
                "speed, vectors per second",
                method,
                f"Construction Speed, {ndims}d Vectors (log scale)",
                log_scale=True,
                smoothing=True,
            )
            fig.write_image(f"{PLOTS_DIR}/{prefix}construction_speed_log.png", scale=2)

        fig = draw_plot(
            xs,
            constuction_speed,
            "index size, number of vectors",
            "speed, vectors per second",
            method,
            f"Construction Speed, {ndims}d Vectors",
            smoothing=True,
            display_slow_index=False,
        )
        fig.write_image(f"{PLOTS_DIR}/{prefix}construction_speed_linear.png", scale=2)

        fig = draw_plot(
            xs,
            constuction_memory,
            "index size, number of vectors",
            "memory consumption, gigabytes",
            method,
            f"Memory Consumption, {ndims}d Vectors",
            display_slow_index=False,
        )
        fig.write_image(f"{PLOTS_DIR}/{prefix}construction_memory.png", scale=2)


if __name__ == "__main__":
    draw_plots(INDEX_NAMES_1B96D, prefix="1b-96d-", draw_log_plots=False)
    draw_plots(INDEX_NAMES_100M96D, prefix="100m-96d-", draw_log_plots=False)
    draw_plots(
        INDEX_NAMES_100M1536D, prefix="100m-1536d-", draw_log_plots=False, ndims=1536
    )
