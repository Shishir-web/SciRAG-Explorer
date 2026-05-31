# SciRAG Explorer

Multi-agent scientific literature RAG system that ingests papers
from arXiv and PubMed, embeds them with BGE-M3, and stores them
in pgvector for hybrid retrieval.

## Status
- [x] Step 1 — arXiv + PubMed ingestion pipeline
- [x] Step 2 — Section-aware chunking + BGE-M3 embeddings
- [ ] Step 3 — Hybrid retrieval (dense + BM25 + Cohere rerank)
- [ ] Step 4 — LangGraph multi-agent orchestration
- [ ] Step 5 — Contradiction detection
- [ ] Step 6 — FastAPI serving layer
- [ ] Step 7 — RAGAS eval harness + CI

## Stack
Python · PostgreSQL · pgvector · BGE-M3 · HuggingFace

## Setup
```bash
cp .env.example .env
pip install -r requirements.txt
python db/init_db.py
python ingestion/run_ingest.py
python chunking/run_chunk.py
python embeddings/run_embed.py
```