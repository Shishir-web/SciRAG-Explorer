from retrieval.dense    import dense_search
from retrieval.sparse   import sparse_search
from retrieval.rrf      import reciprocal_rank_fusion
from retrieval.reranker import rerank
from retrieval.dense    import RetrievedChunk


def retrieve(
    query:          str,
    dense_top_k:    int = 20, 
    sparse_top_k:   int = 20,
    rrf_top_k:      int = 10,
    rerank_top_k:   int = 5,
) ->   list[RetrievedChunk]:
    """Full hybrid retrieval pipeline:
      1. Dense (BGE-M3 + pgvector)
      2. Sparse (BM25)
      3. RRF merge
      4. Cohere rerank
    Returns final top-k chunks ready for the LLM."""
    dense_results  = dense_search(query, top_k=dense_top_k)
    sparse_results = sparse_search(query, top_k=sparse_top_k)
    merged         = reciprocal_rank_fusion(dense_results, sparse_results, top_k=rrf_top_k)
    final          = rerank(query, merged, top_k=rerank_top_k)
    return final
    