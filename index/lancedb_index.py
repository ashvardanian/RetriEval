import lancedb
import numpy as np

from .base_index import BaseIndex
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from functools import partial


def search_single(q: np.ndarray, k: int, db_address: str, table_name: str, metric: str):
    index = lancedb.connect(db_address)
    ids = (index[table_name].search(q).metric(metric).limit(k).to_arrow())['id']

    return [i.as_py() for i in ids]


class LanceDBIndex(BaseIndex):
    def __init__(self, dim: int, metric: str):
        self.db_address = '~/.cache/lancedb'
        self.table_name = 'vectors'
        index = lancedb.connect(self.db_address)
        metric = {'angular': 'cosine', 'l2': 'l2'}[metric]
        self.index_offset = 0

        super().__init__(index, dim, metric, 'LanceDB')

    def add(self, x: np.ndarray):
        if self.table_name in self.index.table_names():
            self.index[self.table_name].add(
                [
                    {'id': i, 'vector': v}
                    for i, v in enumerate(x, start=self.index_offset)
                ]
            )
        else:
            self.index.create_table(
                self.table_name,
                [{'id': i, 'vector': v} for i, v in enumerate(x)],
                mode='overwrite',
            )

        self.index_offset += x.shape[0]

    def search(self, x: np.ndarray, k: int):
        worker = partial(
            search_single,
            k=k,
            db_address=self.db_address,
            table_name=self.table_name,
            metric=self.metric,
        )

        with ThreadPoolExecutor(cpu_count()) as pool:
            futures = pool.map(worker, x)

        results = np.empty((x.shape[0], k), dtype=np.int32)

        for i, result in enumerate(futures):
            results[i] = result

        return results
