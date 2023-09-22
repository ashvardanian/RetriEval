import scann
from .base_index import BaseIndex
from multiprocessing import cpu_count
import numpy as np


class ScannIndex(BaseIndex):
    def __init__(
        self,
        dim: int,
        metric: str,
        num_leaves=3000,
        num_leaves_to_search=300,
        anisotropic_quantization_threshold=0.2,
        dimensions_per_block=2,
        reordering_num_neighbors=100,
    ):
        self.num_leaves = num_leaves
        self.num_leaves_to_search = num_leaves_to_search
        self.anisotropic_quantization_threshold = anisotropic_quantization_threshold
        self.dimensions_per_block = dimensions_per_block
        self.reordering_num_neighbors = reordering_num_neighbors

        metric = {'angular': 'dot_product', 'l2': 'squared_l2'}[metric]

        super().__init__(None, dim, metric, 'SCANN')

    def add(self, x: np.ndarray):
        spherical = False

        if self.metric == 'dot_product':
            x = self.normalize(x)
            spherical = True

        self.index = (
            scann.scann_ops_pybind.builder(x, 1, self.metric)
            .set_n_training_threads(cpu_count())
            .tree(
                num_leaves=self.num_leaves,
                num_leaves_to_search=self.num_leaves_to_search,
                training_sample_size=x.shape[0],
                spherical=spherical,
            )
            .score_ah(
                self.dimensions_per_block,
                anisotropic_quantization_threshold=self.anisotropic_quantization_threshold,
            )
            .reorder(self.reordering_num_neighbors)
            .build()
        )

    def search(self, x: np.ndarray, k: int):
        '''if self.metric == 'dot_product':
        x = self.normalize(x)'''
        return self.index.search_batched_parallel(
            x, k, self.reordering_num_neighbors, self.num_leaves_to_search
        )[0]

    def normalize(self, x: np.ndarray):
        return x / np.linalg.norm(x, axis=-1, keepdims=True)
