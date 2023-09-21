from time import perf_counter
from json import load
from sys import argv

import psutil
import numpy as np
from tqdm import tqdm

from index import INDEXES
from utils.io import read_fbin, read_ibin
from utils.metrics import recall

STATS_DIR = "stats"


def get_memory_usage():
    return psutil.Process().memory_info().rss / 1024


def measure(
    index,
    dataset_path,
    query_path,
    groundtruth_path,
    dataset_size,
    step_size,
    save_every_nth,
):
    queries = read_fbin(query_path)
    groundtruth = read_ibin(groundtruth_path)[:, :1]

    construction_time = []
    memory_consumption = []
    search_time = []
    recalls = []
    chunk_idx = 1

    memory_usage_before = get_memory_usage()
    for start_idx in tqdm(range(0, dataset_size, step_size), desc=index.name):
        if index.name != "SCANN":
            chunk = read_fbin(dataset_path, start_idx, step_size)
        else:
            chunk = read_fbin(dataset_path, 0, chunk_idx * step_size)

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
                f"{STATS_DIR}/{index.name}.chunk_{chunk_idx}.npz",
                construction_time=construction_time,
                memory_consumption=memory_consumption,
                search_time=search_time,
                recalls=recalls,
            )

        chunk_idx += 1

    return construction_time, memory_consumption, search_time, recalls


if __name__ == "__main__":
    config_path = argv[1]
    data_path = argv[2]
    with open(config_path, "r") as f:
        config = load(f)

    for index_config in config["indexes"]:
        index = INDEXES[index_config["name"]](**index_config["params"])

        construction_time, memory_consumption, search_time, recalls = measure(
            index,
            config["index_vectors_path"],
            config["query_vectors_path"],
            config["groundtruth_path"],
            config["dataset_size"],
            config["step_size"],
            config["save_every_nth"],
        )

        np.savez(
            f"{STATS_DIR}/{index.name}.npz",
            construction_time=construction_time,
            memory_consumption=memory_consumption,
            search_time=search_time,
            recalls=recalls,
        )
