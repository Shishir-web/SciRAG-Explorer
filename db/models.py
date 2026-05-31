from sqlalchemy import Column, String, Text, DateTime, JSON, UniqueConstraint
from sqlalchemy.orm import DeclerativeBase
from datetime import datetime, timezone

class Base(DeclerativeBase):
    pass

class Paper(Base):
    __tablename__ = "papers"

    id = Column(String, primary_key=True) #e.g. PMID for PubMed papers
    source = Column(String, nullable=False) # "arxiv" | "pubmed"
    title = Column(String, nullable=False)
    abstract = Column(Text, nullable=False)
    full_text = Column(Text) #
    authors = Column(JSON) # list of author names
    published = Column(DateTime)
    doi = Column(String)
    url = Column(String)
    raw_metadata = Column(JSON) # store the raw metadata from the source for future reference
    ingested_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        UniqueConstraint('id',  name='uq_paper_id'),)
    