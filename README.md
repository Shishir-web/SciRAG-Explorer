# 🔬 SciRAG Explorer

### Multi-Agent Scientific Literature RAG System using arXiv, PubMed, LangGraph & RAGAS

Ingest live papers → Hybrid retrieval → Multi-agent synthesis → CI-evaluated answers

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-latest-purple)
![FastAPI](https://img.shields.io/badge/FastAPI-latest-009688?logo=fastapi&logoColor=white)
![pgvector](https://img.shields.io/badge/pgvector-PostgreSQL-336791?logo=postgresql&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-BGE--M3-yellow?logo=huggingface&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

Researchers lose hours manually hunting PubMed and arXiv for relevant papers, then struggle to assess which sources actually agree. SciRAG solves this:

> **Ask a complex scientific question. Get a grounded, cited answer — with conflicts flagged when sources disagree.**

Papers are ingested daily from arXiv and PubMed APIs, chunked by section (Abstract/Methods/Results), embedded with BGE-M3, and stored in pgvector. A 3-agent LangGraph graph handles retrieval, synthesis, and self-evaluation — with NLI-based contradiction detection before any answer is generated.

---

## System Preview

```
User Query
    ↓
Retriever Agent  →  Hybrid dense (BGE-M3) + sparse (BM25) + Cohere rerank
    ↓
Contradiction Detection  →  NLI flags conflicting claims across sources
    ↓
Synthesiser Agent  →  GPT-4o-mini generates grounded cited answer
    ↓
Critic Agent  →  LLM-as-judge scores faithfulness, loops back if score < 0.7
    ↓
Grounded Answer with inline citations + conflict warnings
```

---

## Eval Results

| Metric | Score | Gate |
|--------|-------|------|
| Faithfulness | 0.874 | ≥ 0.82 ✅ |
| Answer relevancy | 0.801 | ≥ 0.75 ✅ |
| Context precision | 0.763 | ≥ 0.70 ✅ |
| Context recall | 0.712 | ≥ 0.65 ✅ |

Eval runs automatically on every PR via GitHub Actions. Build fails if faithfulness drops below 0.82.

---

## Key Features

- 📥 **Live ingestion** — 40k+ papers from arXiv & PubMed APIs, refreshed daily
- ✂️ **Section-aware chunking** — splits by Abstract/Methods/Results, not naive 512-token windows
- 🔍 **Hybrid retrieval** — BGE-M3 dense + BM25 sparse, merged with Reciprocal Rank Fusion, reranked with Cohere
- 🤖 **3-agent LangGraph graph** — Retriever → Synthesiser → Critic with self-correcting retry loop
- ⚠️ **Contradiction detection** — NLI model flags conflicting claims across sources before synthesis
- 📎 **Citation grounding** — verifies every cited paper actually supports the claim
- 📊 **Automated eval harness** — RAGAS metrics with CI-enforced faithfulness gate
- 🚀 **Streaming API** — FastAPI with SSE streaming, rate limiting, and Docker deployment

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Ingestion | arXiv API, PubMed Entrez, Unstructured.io |
| Embeddings | BGE-M3, HuggingFace Inference API |
| Vector DB | PostgreSQL + pgvector |
| Retrieval | BM25, Cohere Rerank, RRF |
| Orchestration | LangGraph, LangChain |
| LLM | GPT-4o-mini |
| Contradiction | DeBERTa NLI (cross-encoder/nli-deberta-v3-small) |
| API | FastAPI, Uvicorn, Docker |
| Eval | RAGAS, GitHub Actions CI |

---

## Project Structure

```
scirag/
├── ingestion/          # arXiv + PubMed API clients
├── db/                 # SQLAlchemy ORM models
├── chunking/           # Section-aware text splitter
├── embeddings/         # BGE-M3 embedding pipeline
├── retrieval/          # Hybrid dense + sparse + rerank
├── agents/             # LangGraph 3-agent graph
├── contradiction/      # NLI contradiction + grounding
├── api/                # FastAPI serving layer
├── eval/               # RAGAS eval harness
├── data/               # Golden dataset (50 Q&A pairs)
├── tests/              # Unit tests (54 total)
├── Dockerfile
└── docker-compose.yml
```

---

## Setup

### Prerequisites
- Python 3.11+
- Docker
- API keys: OpenAI, Cohere, HuggingFace

### Installation

```bash
# Clone the repo
git clone https://github.com/Shishir-web/scirag-explorer.git
cd scirag-explorer

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and fill in your API keys
cp .env.example .env
```

### Run with Docker

```bash
docker compose up --build
```

### Run Locally

```bash
# Start Postgres
docker run -d --name scirag-db \
  -e POSTGRES_USER=user \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=scirag \
  -p 5432:5432 pgvector/pgvector:pg16

# Ingest papers
python ingestion/run_ingest.py

# Chunk and embed
python chunking/run_chunk.py
python embeddings/run_embed.py

# Start API
uvicorn api.main:app --reload --port 8000
```

---

## API Usage

```bash
# Query the system
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What do recent papers say about GLP-1 and neuroinflammation?"}'

# Health check
curl http://localhost:8000/health

# Paper count stats
curl http://localhost:8000/papers/count

# Interactive Swagger docs
open http://localhost:8000/docs
```

### Example Response

```json
{
  "answer": "Recent studies demonstrate that GLP-1 receptor agonists
             significantly reduce IL-6 levels [pubmed:38291045]...",
  "citations": ["pubmed:38291045", "arxiv:2401.09182"],
  "critic_score": 0.871,
  "has_conflicts": false,
  "retrieval_passes": 1
}
```

---

## Run Eval Harness

```bash
# Fast sanity check
python eval/run_eval.py --difficulty easy --max 5

# Full eval run
python eval/run_eval.py --save-baseline

# CI mode — exits 1 if faithfulness < 0.82
python eval/run_eval.py --max 15 --ci
```

---

## Run Tests

```bash
pytest tests/ -v
# 54 passed
```

---

## Environment Variables

```bash
POSTGRES_URL=postgresql://user:password@localhost:5432/scirag
NCBI_EMAIL=           # required for PubMed API
NCBI_API_KEY=         # optional, increases rate limit
HF_API_TOKEN=         # huggingface.co/settings/tokens
OPENAI_API_KEY=       # platform.openai.com
COHERE_API_KEY=       # cohere.com
```

---

## License

MIT