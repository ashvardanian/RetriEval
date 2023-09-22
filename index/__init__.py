from .faiss_index import FaissIndex
from .lancedb_index import LanceDBIndex
from .qdrant_index import QdrantIndex
from .scann_index import ScannIndex
from .usearch_exact_index import USearchExactIndex
from .usearch_hnsw_index import USearchIndex
from .weaviate_index import WeaviateIndex


INDEXES = {
    'FAISS': FaissIndex,
    'LanceDB': LanceDBIndex,
    'Qdrant': QdrantIndex,
    'SCANN': ScannIndex,
    'USearchExact': USearchExactIndex,
    'USearch': USearchIndex,
    'Weaviate': WeaviateIndex,
}
