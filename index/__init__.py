from .faiss_index import FaissIndex
from .faiss_exact_index import FaissExactIndex
from .lancedb_index import LanceDBIndex
from .qdrant_index import QdrantIndex
from .scann_index import ScannIndex
from .usearch_exact_index import USearchExactIndex
from .usearch_hnsw_index import USearchIndex
from .weaviate_index import WeaviateIndex
from .hnswlib_index import HNSWLibIndex


INDEXES = {
    "FAISS": FaissIndex,
    "FAISSExact": FaissExactIndex,
    "LanceDB": LanceDBIndex,
    "Qdrant": QdrantIndex,
    "SCANN": ScannIndex,
    "USearchExact": USearchExactIndex,
    "USearch": USearchIndex,
    "Weaviate": WeaviateIndex,
    "HNSWLib": HNSWLibIndex,
}
