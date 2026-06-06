import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from db.models import Base, Paper
from db.chunk_models import Chunk
from chunking.section_splitter import chunk_paper


def run():
    engine = create_engine(os.environ["POSTGRES_URL"])
    Base.metadata.create_all(engine)
    Chunk.__table__.create(bind=engine, checkfirst=True)

    with Session(engine) as session:
        papers = session.execute(select(Paper)).scalars().all()
        total  = 0

        for paper in papers:
            existing = session.execute(
                select(Chunk).where(Chunk.paper_id == paper.id).limit(1)
            ).scalar_one_or_none()
            if existing:
                continue

            chunks = chunk_paper(paper.id, paper.abstract, paper.full_text)

            for c in chunks:
                db_chunk = Chunk(
                    id          = f"{paper.id}::chunk::{c.chunk_index}",
                    paper_id    = paper.id,
                    section     = c.section,
                    text        = c.text,
                    char_start  = c.char_start,
                    char_end    = c.char_end,
                    token_count = c.token_count,
                    chunk_index = c.chunk_index,
                    metadata    = {"source": paper.source},
                )
                session.add(db_chunk)
            total += len(chunks)

        session.commit()
        print(f"Chunked {len(papers)} papers → {total} chunks total")


if __name__ == "__main__":
    run()