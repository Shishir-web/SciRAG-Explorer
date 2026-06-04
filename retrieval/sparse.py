import os
from rank_bm25 import BM25Okapi
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from db.chunk_models import Chunk
from retrieval.dense import RetrievedChunk

def build_bm25_index(session: Session) -> tuple[BM25Okapi, list[Chunk]]:
    """Load all chunks from DB and build an in-memory BM25 index.
    For production, this would be persisted or replaced with Elasticsearch.
    """
    chunks = session.execute(select(Chunk)).scalars().all()
    tokenized = [c.text.lower().split() for c in chunks]
    index = BM25Okapi(tokenized)
    return index, chunks

def sparse_search(query: str, top_k: int = 20) -> list[RetrievedChunk]:
    """BM25 keyword search over all chunks.
    Returns top-k results sorted by BM25 score descending.
    """
    engine = create_engine(os.environ["POSTGRES__URL"])
    tokens = query.lower().split()
    with Session(engine) as session:
        index, chunks = build_bm25_index(session)
        scores = index.get_scores(tokens)

    scored = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    return [
        RetrievedChunk(
            chunk_id=chunk.id,
            paper_id=chunk.paper_id,
            section=chunk.section,
            chunk_text=chunk.text,
            score=float(score),
            rank=i + 1,
        )
        for i, (chunk, score) in enumerate(scored)
        if score > 0
    ]