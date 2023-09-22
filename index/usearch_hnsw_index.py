from usearch.index import Index, MetricKind
from .base_index import BaseIndex
import numpy as np


class USearchIndex(BaseIndex):
    def __init__(
        self,
        dim: int,
        metric: str,
        m: int = 16,
        ef_construction: int = 128,
        ef_search: int = 64,
        dtype: str = 'f32',
    ):
        metric = {'angular': MetricKind.IP, 'l2': MetricKind.L2sq}[metric]

        index = Index(
            ndim=dim,
            metric=metric,
            dtype=dtype,
            connectivity=m,
            expansion_add=ef_construction,
            expansion_search=ef_search,
        )
        self.index_offset = 0
        super().__init__(
            index, dim, metric, f'USearch-HNSW-{dtype}', m, ef_construction, ef_search
        )

    def add(self, x: np.ndarray):
        if self.metric == MetricKind.IP:
            x = self.normalize(x)

        self.index.add(np.arange(x.shape[0], dtype=np.longlong) + self.index_offset, x)
        self.index_offset += x.shape[0]

    def search(self, x: np.ndarray, k: int):
        if self.metric == MetricKind.IP:
            x = self.normalize(x)

        return self.index.search(x, k).keys

    def normalize(self, x):
        return x / np.linalg.norm(x, axis=-1, keepdims=True)
