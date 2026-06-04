import os
import json
import asyncio
import concurrent.futures
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sse_starlette.sse import EventSourceResponse
from sqlalchemy.orm import Session
from sqlalchemy import text

from api.schemas      import (
    QueryRequest, QueryResponse, HealthResponse,
    ErrorResponse, CitationDetail, ConflictDetail,
)
from api.middleware   import RequestLoggingMiddleware
from api.dependencies import limiter, check_db_health, get_engine
from agents           import run_query


app = FastAPI(
    title       = "SciRAG Explorer API",
    description = "Multi-agent scientific literature RAG system",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


def format_response(state: dict) -> QueryResponse:
    conflicts = [
        ConflictDetail(
            paper_id_a = c.get("paper_id_a", ""),
            paper_id_b = c.get("paper_id_b", ""),
            confidence = c.get("confidence", 0.0),
        )
        for c in state.get("conflicts", [])
    ]

    grounding = {
        pid: CitationDetail(
            paper_id   = pid,
            grounded   = v.get("grounded", False),
            confidence = v.get("confidence", 0.0),
        )
        for pid, v in state.get("grounding", {}).items()
    }

    return QueryResponse(
        query            = state["query"],
        answer           = state["answer"],
        citations        = state.get("citations", []),
        critic_score     = state.get("critic_score", 0.0),
        critic_feedback  = state.get("critic_feedback", ""),
        retrieval_passes = state.get("retrieval_passes", 0),
        conflicts        = conflicts,
        grounding        = grounding,
        has_conflicts    = len(conflicts) > 0,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    db_status = check_db_health()
    return HealthResponse(
        status = "ok" if db_status == "ok" else "degraded",
        db     = db_status,
    )


@app.post(
    "/query",
    response_model = QueryResponse,
    responses      = {
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
@limiter.limit("30/minute")
async def query(request: Request, body: QueryRequest):
    if body.stream:
        return await stream_query(request, body)
    try:
        state = run_query(body.query)
        return format_response(state)
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail      = f"Agent graph failed: {str(e)}",
        )


async def stream_query(request: Request, body: QueryRequest):
    async def event_generator():
        try:
            yield {
                "event": "status",
                "data":  json.dumps({"message": "Retrieving relevant papers..."}),
            }
            await asyncio.sleep(0)

            state = run_query(body.query)

            yield {
                "event": "status",
                "data":  json.dumps({
                    "message": (
                        f"Retrieved {len(state['chunks'])} chunks, "
                        f"detected {len(state.get('conflicts', []))} conflicts"
                    )
                }),
            }
            await asyncio.sleep(0)

            words = state["answer"].split(" ")
            for i, word in enumerate(words):
                yield {
                    "event": "chunk",
                    "data":  json.dumps({"token": word + " "}),
                }
                if i % 5 == 0:
                    await asyncio.sleep(0.02)

            response = format_response(state)
            yield {
                "event": "done",
                "data":  response.model_dump_json(),
            }

        except Exception as e:
            yield {
                "event": "error",
                "data":  json.dumps({"error": str(e)}),
            }

    return EventSourceResponse(event_generator())


@app.get("/papers/count")
async def paper_count():
    engine = get_engine()
    with Session(engine) as session:
        papers = session.execute(
            text("SELECT count(*) FROM papers")
        ).scalar()
        chunks = session.execute(
            text("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL")
        ).scalar()
    return {"papers": papers, "chunks_embedded": chunks}