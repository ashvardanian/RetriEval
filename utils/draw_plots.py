import plotly.express as px
import plotly.graph_objects as go
import numpy as np


STEP_SIZE = 100_000
SEARCH_SIZE = 10_000

INDEX_NAMES = (
    "USearch-HNSW-f32",
    "USearch-HNSW-f16",
    "USearch-HNSW-i8",
    "FAISS-HNSW",
    "SCANN",
    "Weaviate",
    # 'Qdrant'
)

EXACT_INDEX_NAMES = (
    "USearch-Exact-f32",
    "USearch-Exact-f16",
    "USearch-Exact-i8",
    "LanceDB",
    "FAISS-Exact",
)

SLOW_INDEX_NAMES = ("Qdrant", "Weaviate")

NAME_MAPPING = {
    "USearch-HNSW-f32": "USearch(HNSW,f32)",
    "USearch-HNSW-f16": "USearch(HNSW,f16)",
    "USearch-HNSW-i8": "USearch(HNSW,i8)",
    "USearch-Exact-f32": "USearch(Exact,f32)",
    "USearch-Exact-f16": "USearch(Exact,f16)",
    "USearch-Exact-i8": "USearch(Exact,i8)",
    "FAISS-HNSW": "FAISS(HNSW)",
    "FAISS-Exact": "FAISS(Exact)",
    "SCANN": "SCANN",
    "Weaviate": "Weaviate",
    "LanceDB": "LanceDB",
    "Qdrant": "Qdrant",
}

STATS_DIR = "stats"


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


if __name__ == "__main__":
    xs = []
    constuction_speed = []
    constuction_memory = []
    search_speed = []
    search_recall = []
    method = []

    for index_name in INDEX_NAMES:
        data = np.load(f"{STATS_DIR}/{index_name}.npz")

        xs.append(
            np.arange(
                STEP_SIZE, STEP_SIZE + data["recalls"].shape[0] * STEP_SIZE, STEP_SIZE
            )
        )

        constuction_speed.append(STEP_SIZE / data["construction_time"])
        constuction_memory.append(data["memory_consumption"] / 1024 / 1024)
        search_speed.append(SEARCH_SIZE / data["search_time"])
        search_recall.append(data["recalls"])
        method.append(NAME_MAPPING[index_name])

    exact_xs = []
    exact_search_speed = []
    exact_method = []

    for index_name in EXACT_INDEX_NAMES:
        data = np.load(f"{STATS_DIR}/{index_name}.npz")
        exact_xs.append(
            np.arange(
                STEP_SIZE,
                STEP_SIZE + data["search_time"].shape[0] * STEP_SIZE,
                STEP_SIZE,
            )
        )
        exact_search_speed.append(SEARCH_SIZE / data["search_time"])

    fig = draw_plot(
        xs,
        search_speed,
        "index size",
        "vecs/s",
        method,
        "DEEP1B: Search Speed (log scale)",
        log_scale=True,
        smoothing=True,
    )
    fig.write_image("plots/search_speed_log.png", scale=2)

    fig = draw_plot(
        xs,
        search_speed,
        "index size",
        "vecs/s",
        method,
        "DEEP1B: Search Speed",
        log_scale=False,
        smoothing=True,
        display_slow_index=False,
    )
    fig.write_image("plots/search_speed_linear.png", scale=2)

    fig = draw_plot(
        xs, search_recall, "index size", "recall", method, "DEEP1B: Search Recall@1"
    )
    fig.write_image("plots/search_recall.png", scale=2)

    fig = draw_plot(
        xs,
        constuction_speed,
        "index size",
        "vecs/s",
        method,
        "DEEP1B: Construction Speed (log scale)",
        log_scale=True,
        smoothing=True,
    )
    fig.write_image("plots/construction_speed_log.png", scale=2)

    fig = draw_plot(
        xs,
        constuction_speed,
        "index size",
        "vecs/s",
        method,
        "DEEP1B: Construction Speed",
        smoothing=True,
        display_slow_index=False,
    )
    fig.write_image("plots/construction_speed_linear.png", scale=2)

    fig = draw_plot(
        xs,
        constuction_memory,
        "index size",
        "memory (gb)",
        method,
        "DEEP1B: Construction Memory Consumption",
        display_slow_index=False,
    )
    fig.write_image("plots/construction_memory.png", scale=2)

    fig = draw_plot(
        exact_xs,
        exact_search_speed,
        "index size",
        "vecs/s",
        [NAME_MAPPING[name] for name in EXACT_INDEX_NAMES],
        "DEEP1B: Search Speed (log scale)",
        log_scale=True,
    )
    fig.write_image("plots/exact_search_speed.png", scale=2)
