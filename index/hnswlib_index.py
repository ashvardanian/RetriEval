import hnswlib
from .base_index import BaseIndex
from numpy import ndarray


class HNSWLibIndex(BaseIndex):
    def __init__(
        self,
        dim: int,
        metric: str,
        index_size: int,
        m: int = 32,
        ef_construction: int = 40,
        ef_search: int = 16,
    ):
        metric = "cosine" if metric == "angular" else "l2"
        index = hnswlib.Index(space=metric, dim=dim)
        index.init_index(max_elements=index_size, ef_construction=ef_construction, M=m)
        index.set_ef(ef_search)

        super().__init__(index, dim, metric, "HNSWLIB", m, ef_construction, ef_search)

    def add(self, x: ndarray):
        self.index.add_items(x)

    def search(self, x: ndarray, k: int):
        return self.index.knn_query(x, k=k)[0]
