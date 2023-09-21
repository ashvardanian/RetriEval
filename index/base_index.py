from abc import abstractmethod, ABC
from numpy import ndarray
from typing import Any


class BaseIndex(ABC):
    def __init__(
        self,
        index: Any,
        dim: int,
        metric: str,
        name: str,
        m: int = -1,
        ef_construction: int = -1,
        ef_search: int = -1,
    ):
        self.index = index
        self.dim = dim
        self.metric = metric
        self.name = name
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search

    @abstractmethod
    def add(self, x: ndarray):
        pass

    @abstractmethod
    def search(self, x: ndarray, k: int):
        pass

    def __str__(self):
        if self.m != -1 and self.ef_construction != -1 and self.ef_search != -1:
            return f"{self.name}, m={self.m}, ef_construction={self.ef_construction}, ef_search={self.ef_search}"

        return self.name
