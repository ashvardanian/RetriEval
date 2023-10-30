from weaviate import Client, EmbeddedOptions
from .base_index import BaseIndex
from numpy import ndarray, array
from uuid import UUID
from multiprocessing import Pool, cpu_count
from functools import partial


def search_single_query(q, k, class_name):
    index = Client(embedded_options=EmbeddedOptions())
    results = (
        index.query.get(class_name)
        .with_additional("id")
        .with_near_vector(
            {
                "vector": q,
            }
        )
        .with_limit(k)
        .do()
    )
    return array(
        [UUID(r["_additional"]["id"]).int for r in results["data"]["Get"]["Vector"]]
    )


class WeaviateIndex(BaseIndex):
    def __init__(
        self,
        dim: int,
        metric: str,
        m: int = 64,
        ef_construction: int = 128,
        ef_search: int = 100,
        construction_batch_size: int = 100,
    ):
        index = Client(embedded_options=EmbeddedOptions())
        metric = {"angular": "cosine", "l2": "l2-squared"}[metric]

        self.construction_batch_size = construction_batch_size
        self.class_name = "Vector"
        self.index_offset = 0
        index.schema.delete_class(self.class_name)

        index.schema.create(
            {
                "classes": [
                    {
                        "class": self.class_name,
                        "properties": [
                            {
                                "name": "index",
                                "dataType": ["int"],
                            }
                        ],
                        "vectorIndexConfig": {
                            "distance": metric,
                            "maxConnections": m,
                            "efConstruction": ef_construction,
                            "ef": ef_search,
                        },
                    }
                ]
            }
        )
        super().__init__(index, dim, metric, "Weaviate", m, ef_construction, ef_search)

    def add(self, x: ndarray):
        self.index.batch.configure(batch_size=self.construction_batch_size)
        with self.index.batch as batch:
            for i, vector in enumerate(x, start=self.index_offset):
                batch.add_data_object(
                    data_object={"index": i},
                    class_name=self.class_name,
                    vector=vector,
                    uuid=UUID(int=i),
                )
        self.index_offset += x.shape[0]

    def search(self, x: ndarray, k: int):
        worker = partial(search_single_query, k=k, class_name=self.class_name)

        with Pool(cpu_count()) as pool:
            results = pool.map(worker, x)

        return array(results)
