from usearch.index import search, MetricKind
from .base_index import BaseIndex
import numpy as np


class USearchExactIndex(BaseIndex):
    def __init__(self, dim: int, metric: str, dtype: str = "f32"):
        metric = {"angular": MetricKind.Cos, "l2": MetricKind.L2sq}[metric]
        self.data = None
        self.dtype = dtype
        self.np_dtype = {"f32": np.float32, "f16": np.float16, "i8": np.int8}[dtype]
        super().__init__(None, dim, metric, f"USearch-Exact-{dtype}")

    def add(self, x: np.ndarray):
        if self.data is None:
            self.data = x.astype(self.np_dtype)
        else:
            self.data = np.concatenate((self.data, x.astype(self.np_dtype)))

    def search(self, x, k: int):
        return search(
            self.data, x.astype(self.np_dtype), k, self.metric, exact=True
        ).keys
