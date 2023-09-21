from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    OptimizersConfigDiff,
    HnswConfigDiff,
    SearchParams,
    SearchRequest,
    CollectionStatus,
    PointStruct,
)
from qdrant_client import grpc
from multiprocessing import cpu_count
from .base_index import BaseIndex
from time import sleep
import numpy as np


TIMEOUT = 30


class QdrantIndex(BaseIndex):
    def __init__(
        self,
        dim: int,
        metric: str,
        m: int = 16,
        ef_construction: int = 100,
        ef_search: int = 128,
        is_local: bool = True,
        port: int = 6334,
        construction_batch_size=128,
    ):
        metric = {"angular": Distance.COSINE, "l2": Distance.EUCLID}[metric]

        self.construction_batch_size = construction_batch_size

        self.collection_name = "Vectors"
        if is_local:
            index = QdrantClient(":memory:")
        else:
            index = QdrantClient(f"localhost:{port}", prefer_grpc=True)

        index.recreate_collection(
            collection_name=self.collection_name,
            shard_number=2,
            vectors_config=VectorParams(size=dim, distance=metric),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=0,
                memmap_threshold=20000,
                default_segment_number=2,
                max_optimization_threads=cpu_count(),
            ),
            hnsw_config=HnswConfigDiff(
                m=m, ef_construct=ef_construction, max_indexing_threads=cpu_count()
            ),
            timeout=TIMEOUT,
        )

        self.index_offset = 0

        super().__init__(index, dim, metric, "Qdrant", m, ef_construction, ef_search)

    def add(self, x: np.ndarray):
        self.index.upload_collection(
            collection_name=self.collection_name,
            vectors=x,
            ids=list(range(self.index_offset, self.index_offset + x.shape[0])),
            batch_size=self.construction_batch_size,
            parallel=cpu_count(),
        )

        # Re-enabling indexing
        self.index.update_collection(
            collection_name=self.collection_name,
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=20000, max_optimization_threads=cpu_count()
            ),
            hnsw_config=HnswConfigDiff(
                m=self.m,
                ef_construct=self.ef_construction,
                max_indexing_threads=cpu_count(),
            ),
            timeout=TIMEOUT,
        )

        self.index_offset += x.shape[0]

    def search(self, x: np.ndarray, k: int):
        """params = SearchParams(
            exact=False,
            hnsw_ef=self.ef_search
        )
        queries = [SearchRequest(vector=v, limit=k, params=params) for v in x]

        raw_results = self.index.search_batch(
            collection_name=self.collection_name,
            requests=queries
        )

        results = np.empty((x.shape[0], k), dtype=np.uint32)

        for i, result in enumerate(raw_results):
            results[i] = [r.id for r in result]

        return results"""

        def iter_batches(iterable, batch_size):
            batch = []
            for item in iterable:
                batch.append(item)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch

        search_queries = [
            grpc.SearchPoints(
                collection_name=self.collection_name,
                vector=q.tolist(),
                limit=k,
                with_payload=grpc.WithPayloadSelector(enable=False),
                with_vectors=grpc.WithVectorsSelector(enable=False),
                params=grpc.SearchParams(hnsw_ef=self.ef_search),
            )
            for q in x
        ]

        result = []

        for request_batch in iter_batches(search_queries, 128):
            grpc_res: grpc.SearchBatchResponse = self.index.grpc_points.SearchBatch(
                grpc.SearchBatchPoints(
                    collection_name=self.collection_name,
                    search_points=request_batch,
                    read_consistency=None,
                ),
                timeout=TIMEOUT,
            )

            for r in grpc_res.result:
                result.append([hit.id.num for hit in r.result])

        return np.array(result)
