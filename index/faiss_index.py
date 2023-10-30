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
        dtype: str = "f32",
    ):
        self.dtype = dtype

        if dtype == "f32":
            index = faiss.IndexHNSWFlat(dim, m)
        elif dtype == "f16":
            index = faiss.IndexHNSWSQ(dim, faiss.ScalarQuantizer.QT_fp16, m)
        elif dtype == "i8":
            index = faiss.IndexHNSWSQ(dim, faiss.ScalarQuantizer.QT_8bit, m)

        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef_search

        super().__init__(
            index, dim, metric, f"FAISS-HNSW-{dtype}", m, ef_construction, ef_search
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

    def train(self, x):
        if self.is_training_needed:
            self.index.train(x)

    @property
    def is_training_needed(self):
        return self.dtype != "f32"
