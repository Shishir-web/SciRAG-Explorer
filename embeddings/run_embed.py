import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from db.chunk_models import Chunk
from embeddings.bge_embedder import embed_texts

def run():
    engine = create_engine(os.environ["POSTGRES_URL"])

    with Session(engine) as session:
        # Fetch chunks that don't have embeddings yet
        chunks = session.execute(
            select(Chunk).where(Chunk.embedding == None)
        ).scalars().all()

        print(f"Embedding {len(chunks)} chunks...")

        texts   = [c.text for c in chunks]
        vectors = embed_texts(texts)

        for chunk, vector in zip(chunks, vectors):
            chunk.embedding = vector

        session.commit()
        print("All embeddings saved.")

if __name__ == "__main__":
    run()