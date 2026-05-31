from sqlalchemy import Column, String, Text, Integer, JSON, ForeignKey
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from db.models import Base 

EMBEDDING_DIM = 1024

class Chunk(Base):
    __tablename__ = "chunks"

    id         = Column(String, primary_key=True) # "{paper_id}:{chunk}::{n}"
    paper_id   = Column(String, ForeignKey("papers.id"), nullable=False)
    section    = Column(String) # e.g. "abstract", "introduction", "methods", "results", "discussion"
    
    text       = Column(Text, nullable=False)
    char_start = Column(Integer) # character offset in the original paper text, used for highlighting in the UI
    char_end   = Column(Integer)
    token_count= Column(Integer)
    chunk_index= Column(Integer) # the n in the chunk id, used for ordering chunks within a paper
    metadata   = Column(JSON)

    #The embedding vector for this chunk, stored as a pgvector column
    embedding  = Column(Vector(EMBEDDING_DIM))

    paper      = relationship("Paper", back_populates="chunks")