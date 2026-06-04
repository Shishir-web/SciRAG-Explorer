from sqlalchemy import Column, String, Text, Integer, JSON, ForeignKey
from pgvector.sqlalchemy import Vector
from db.models import Base

EMBEDDING_DIM = 1024


class Chunk(Base):
    __tablename__ = "chunks"

    id          = Column(String, primary_key=True)
    paper_id    = Column(String, ForeignKey("papers.id"), nullable=False)
    section     = Column(String)
    text        = Column(Text, nullable=False)
    char_start  = Column(Integer)
    char_end    = Column(Integer)
    token_count = Column(Integer)
    chunk_index = Column(Integer)
    metadata    = Column(JSON)
    embedding   = Column(Vector(EMBEDDING_DIM))