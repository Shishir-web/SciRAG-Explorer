import os
from dataclasses import dataclass
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from embeddings.bge_embedder import embed_texts


@dataclass
class RetrievedChunk:
    chunk_id:   str
    paper_id:   str
    section:    str
    chunk_text: str
    score:      float
    rank:       int


def dense_search(query: str, top_k: int = 20) -> list[RetrievedChunk]:
    """
    Embed the query and retrieve top-k chunks by cosine similarity
    using pgvector's <=> operator (cosine distance).
    """
    engine = create_engine(os.environ["POSTGRES_URL"])

    query_vector = embed_texts([query])[0]

    sql = text("""
        SELECT
            c.id          AS chunk_id,
            c.paper_id,
            c.section,
            c.text        AS chunk_text,
            1 - (c.embedding <=> CAST(:vec AS vector)) AS score
        FROM chunks c
        WHERE c.embedding IS NOT NULL
        ORDER BY c.embedding <=> CAST(:vec AS vector)
        LIMIT :k
    """)

    with Session(engine) as session:
        rows = session.execute(sql, {
            "vec": str(query_vector),
            "k":   top_k,
        }).fetchall()

    return [
        RetrievedChunk(
            chunk_id  = r.chunk_id,
            paper_id  = r.paper_id,
            section   = r.section,
            chunk_text= r.chunk_text,
            score     = float(r.score),
            rank      = i + 1,
        )
        for i, r in enumerate(rows)
    ]