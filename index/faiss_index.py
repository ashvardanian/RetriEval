import faiss
from .base_index import BaseIndex
from numpy import ndarray, linalg


class FaissIndex(BaseIndex):
    def __init__(
        self,
        dim: int,
        metric: str,
        m: int = 32,
        ef_construction: int = 40,
        ef_search: int = 16,
        exact: bool = False,
    ):
        if exact:
            if metric == "angular":
                index = faiss.IndexFlatIP(dim)
            else:
                index = faiss.IndexFlatL2(dim)

            super().__init__(index, dim, metric, "FAISS-Exact", -1, -1, -1)
        else:
            index = faiss.IndexHNSWFlat(dim, m)
            index.hnsw.efConstruction = ef_construction
            index.hnsw.efSearch = ef_search
            super().__init__(
                index, dim, metric, "FAISS-HNSW", m, ef_construction, ef_search
            )

    def add(self, x: ndarray):
        if self.metric == "angular":
            x = self.normalize(x)

        self.index.add(x)

    def search(self, x: ndarray, k: int):
        if self.metric == "angular":
            x = self.normalize(x)

        return self.index.search(x, k)[1]

    def normalize(self, x):
        return x / linalg.norm(x, axis=-1, keepdims=True)
