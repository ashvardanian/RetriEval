from faiss import knn
from faiss.loader import METRIC_L2, METRIC_INNER_PRODUCT
from .base_index import BaseIndex
import numpy as np


class FaissExactIndex(BaseIndex):
    def __init__(self, dim: int, metric: str):
        metric = {"angular": METRIC_INNER_PRODUCT, "l2": METRIC_L2}[metric]
        self.data = None
        super().__init__(None, dim, metric, f"FAISS-Exact")

    def add(self, x: np.ndarray):
        if self.metric == METRIC_INNER_PRODUCT:
            x = x / np.linalg.norm(x, axis=1, keepdims=True)

        if self.data is None:
            self.data = x
        else:
            self.data = np.concatenate((self.data, x))

    def search(self, x, k: int):
        return knn(x, self.data, k, metric=self.metric)[1]
