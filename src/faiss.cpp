/**
 *  @file faiss.cpp
 *  @brief The binary HNSW expansions FAISS's C API leaves unreachable, for `retri-eval-faiss`.
 */
#include <faiss/IndexBinaryHNSW.h>
#include <faiss/IndexIDMap.h>

/** Sets @p expansion_add and @p expansion_search, each when positive, on the binary HNSW that
 *  @p index is or wraps in an `IDMap`; nonzero when it holds none. */
extern "C" int retrieval_faiss_binary_hnsw_set_expansions(faiss::IndexBinary* index, int expansion_add,
                                                          int expansion_search) {
    if (faiss::IndexBinaryIDMap* const mapped = dynamic_cast<faiss::IndexBinaryIDMap*>(index)) index = mapped->index;
    faiss::IndexBinaryHNSW* const hnsw = dynamic_cast<faiss::IndexBinaryHNSW*>(index);
    if (!hnsw) return 1;
    if (expansion_add > 0) hnsw->hnsw.efConstruction = expansion_add;
    if (expansion_search > 0) hnsw->hnsw.efSearch = expansion_search;
    return 0;
}
