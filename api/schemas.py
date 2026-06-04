from pydantic import BaseModel, Field
from typing import Optional

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="Scientific question to answer",
        examples=["What do recent papers say about GLP-1 and neuroinflammation?"]
    )
    stream: bool = Field(
        default=False,
        description="If true, stream the answer token by token via SSE"
    )
    rerank_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve after reranking"
    )


class CitationDetail(BaseModel):
    paper_id:  str
    grounded:  bool
    confidence: float

class ConflictDetail(BaseModel):
    paper_id_a:  str
    paper_id_b:  str
    confidence:  float

class QueryResponse(BaseModel):
    query:            str
    answer:           str
    citations:        list[str]
    critic_score:     float
    critic_feedback:  str
    retrieval_passes: int
    conflicts:        list[ConflictDetail]
    grounding:        dict[str, CitationDetail]
    has_conlicts:     bool

class HealthResponse(BaseModel):
    status:   str
    db:       str
    version:  str = "1.0.0"

class ErrorResponse(BaseModel):
    error:   str
    detail:  Optional[str] = None 