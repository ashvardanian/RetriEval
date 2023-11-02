from time import perf_counter
from json import load
from sys import argv

import psutil
import numpy as np
from tqdm import tqdm

from index import INDEXES
from usearch.io import load_matrix
from utils.metrics import recall


STATS_DIR = "stats"
SAVE_DIRS = {
    "USearch-HNSW-f32": "/mnt/nvme5n1",
    "USearch-HNSW-i8": "/mnt/disk2",
    "USearch-HNSW-f16": "/mnt/disk14",
    "FAISS-HNSW": "/mnt/disk15",
}


def get_memory_usage():
    return psutil.Process().memory_info().rss / 1024


def get_train_dataset(dataset_path, dataset_size, frac=0.01):
    size = int(dataset_size * frac)
    start_idx = np.random.randint(0, dataset_size - size)

    return load_matrix(dataset_path, start_idx, size, view=True)


def measure(
    index,
    dataset_path,
    query_path,
    groundtruth_path,
    dataset_size,
    step_size,
    save_every_nth,
    suffix,
):
    queries = load_matrix(query_path)
    if index.dim != queries.shape[1]:
        queries = np.tile(queries, (1, index.dim // queries.shape[1]))

    groundtruth = load_matrix(groundtruth_path)[:, :1]

    construction_time = []
    memory_consumption = []
    search_time = []
    recalls = []
    chunk_idx = 1

    memory_usage_before = get_memory_usage()
    for start_idx in tqdm(range(0, dataset_size, step_size), desc=index.name):
        if index.name != "SCANN":
            chunk = load_matrix(dataset_path, start_idx, step_size, view=True)
        else:
            chunk = load_matrix(dataset_path, 0, chunk_idx * step_size, view=True)

        if index.dim != chunk.shape[1]:
            chunk = np.tile(chunk, (1, index.dim // chunk.shape[1]))

        start_time = perf_counter()

        index.add(chunk)

        consumed_time = perf_counter() - start_time
        consumed_memory = get_memory_usage() - memory_usage_before

        construction_time.append(consumed_time)
        memory_consumption.append(consumed_memory)

        start_time = perf_counter()
        nn_ids = index.search(queries, 1)
        consumed_time = perf_counter() - start_time
        search_time.append(consumed_time)

        recall_at_one = recall(nn_ids, groundtruth, ats=(1,))
        recalls.append(float(recall_at_one[0]))

        if chunk_idx % save_every_nth == 0:
            np.savez(
                f"{STATS_DIR}/{index.name}.chunk_{chunk_idx}{suffix}.npz",
                construction_time=construction_time,
                memory_consumption=memory_consumption,
                search_time=search_time,
                recalls=recalls,
            )

        chunk_idx += 1

    """if suffix == "-1B" and index.name in SAVE_DIRS:
        index.index.save(f"{SAVE_DIRS[index.name]}/{index.name}.usearch")"""

    return construction_time, memory_consumption, search_time, recalls


if __name__ == "__main__":
    config_path = argv[1]
    suffix = "" if len(argv) < 3 else f"-{argv[2]}"

    with open(config_path, "r") as f:
        config = load(f)

    for index_config in config["indexes"]:
        index = INDEXES[index_config["name"]](**index_config["params"])

        if index.is_training_needed:
            train_dataset = get_train_dataset(
                config["index_vectors_path"], config["dataset_size"]
            )
            if index.dim != train_dataset.shape[1]:
                train_dataset = np.tile(
                    train_dataset, (1, index.dim // train_dataset.shape[1])
                )
            print(f"Training: {index}")
            index.train(train_dataset)

        construction_time, memory_consumption, search_time, recalls = measure(
            index,
            config["index_vectors_path"],
            config["query_vectors_path"],
            config["groundtruth_path"],
            config["dataset_size"],
            config["step_size"],
            config["save_every_nth"],
            suffix,
        )

        np.savez(
            f"{STATS_DIR}/{index.name}{suffix}.npz",
            construction_time=construction_time,
            memory_consumption=memory_consumption,
            search_time=search_time,
            recalls=recalls,
        )
