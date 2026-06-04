from retrieval.dense import RetrievedChunk

RRF_K = 60 #standard constant, reduces impacrt of very high ranks

def reciprocal_rank_fusion(
    dense_results:  list[RetrievedChunk],
    sparse_results: list[RetrievedChunk],
    top_k: int = 10,
) -> list[RetrievedChunk]:
    """Merge dense and sparse ranked lists using Reciprocal Rank Fusion.

    RRF score = sum over each list of 1 / (k + rank)

    This is retrieval-method agnostic — it only cares about rank position,
    not the raw scores (which are on incomparable scales).
    """
    rrf_scores: dict[str, float] = {}

    #Accumulate RRF scores from dense results
    for results in [dense_results, sparse_results]:
        for r in results:
            cid = result.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (
                1.0 / (RRF_K + result.rank)
            )
    
    # Build a lookup for chunk_id to RetrievedChunk (with original metadata)
    chunk_lookup: dict[str, RetrievedChunk] = {}
    for result in dense_results, sparse_results:
        if result.chunk_id not in chunk_lookup:
            chunk_lookup[result.chunk_id] = result

    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

    return [ 
        RetrievedChunk(
            chunk_id=chunk_lookup[cid].chunk_id,
            paper_id=chunk_lookup[cid].paper_id,
            section=chunk_lookup[cid].section,
            chunk_text=chunk_lookup[cid].chunk_text,
            score=rrf_scores[cid],
            rank=i + 1,
        )
        for i, cid in enumerate(sorted_ids)
    ]
