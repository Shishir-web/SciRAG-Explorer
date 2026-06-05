import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert

from db.models import Base, Paper
from ingestion.arxiv_client  import fetch_arxiv
from ingestion.pubmed_client import fetch_pubmed

QUERIES = [
    "GLP-1 neuroinflammation",
    "CRISPR base editing off-target",
    "microbiome mental health",
]


def upsert_papers(session: Session, papers: list[Paper]):
    for paper in papers:
        stmt = (
            insert(Paper)
            .values(
                id          = paper.id,
                source      = paper.source,
                title       = paper.title,
                abstract    = paper.abstract,
                authors     = paper.authors,
                published   = paper.published,
                doi         = paper.doi,
                url         = paper.url,
                raw_metadata    = paper.raw_metadata,
                ingested_at = paper.ingested_at,
            )
            .on_conflict_do_nothing(index_elements=["id"])
        )
        session.execute(stmt)
    session.commit()


def run():
    engine = create_engine(os.environ["POSTGRES_URL"])
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        for query in QUERIES:
            print(f"\n→ arXiv: {query}")
            arxiv_papers = list(fetch_arxiv(query, max_results=50))
            upsert_papers(session, arxiv_papers)
            print(f"  inserted/skipped {len(arxiv_papers)} papers")

            print(f"→ PubMed: {query}")
            pm_papers = list(fetch_pubmed(query, max_results=50))
            upsert_papers(session, pm_papers)
            print(f"  inserted/skipped {len(pm_papers)} papers")

    print("\nIngestion complete.")


if __name__ == "__main__":
    run()