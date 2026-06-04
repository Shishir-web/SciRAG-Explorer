import os
import cohere
from retrieval.dense import RetrievedChunk 

def rerank(
    query:    str,
    chunks:   list[RetrievedChunk],
    top_k:    int = 5, 
) -> list[RetrievedChunk]:
    """Send the merged RRF results to Cohere Rerank.
    Cohere scores each (query, chunk) pair more precisely than
    embedding similarity alone.
    Returns top_k re-scored and re-ranked chunks."""

    if not chunks:
        return []
    
    co = cohere.Client(os.environ["COHERE_API_KEY"])

    documents = [c.chunk_text for c in chunks]

    response = co.rank(
        model="rerank-engine-v3.0",
        query=query,
        documents=documents,
        top_n=top_k,
    )

    reranked: list[RetrievedChunk] = []
    for i, result in enumerate(response.results):
        original = chunks[result.index]
        reranked.append(RetrievedChunk(
            chunk_id=original.chunk_id,
            paper_id=original.paper_id,
            chunk_text=original.chunk_text,
            section=original.section,
            score=result.relevance_score,
            rank=i + 1,
        ))
    
    return reranked