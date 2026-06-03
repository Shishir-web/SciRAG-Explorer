# SciRAG Explorer

A production-grade multi-agent scientific literature RAG system that ingests papers from arXiv and PubMed, synthesises multi-paper evidence, and self-evaluates answer quality — targeting researchers and science journalists who need trustworthy, citation-grounded answers from the latest science.

## Architecture

```
User Query
    ↓
Retriever Agent (hybrid dense + sparse + Cohere rerank)
    ↓
Contradiction Detection (NLI-based)
    ↓
Synthesiser Agent (GPT-4o-mini + citation grounding)
    ↓
Critic Agent (LLM-as-judge, loops back if score < 0.7)
    ↓
Grounded Answer with Citations
```

## Status
- [x] Step 1 — arXiv + PubMed ingestion pipeline
- [x] Step 2 — Section-aware chunking + BGE-M3 embeddings (pgvector)
- [x] Step 3 — Hybrid retrieval (dense + BM25 + Cohere rerank)
- [x] Step 4 — LangGraph multi-agent orchestration
- [x] Step 5 — Contradiction detection (NLI)
- [x] Step 6 — FastAPI serving layer + Docker
- [x] Step 7 — RAGAS eval harness + CI pipeline

## Key Features

- **40k+ papers** ingested from arXiv and PubMed APIs with daily refresh
- **Section-aware chunking** — splits by Abstract/Methods/Results, not naive 512-token windows
- **Hybrid retrieval** — BGE-M3 dense embeddings + BM25 sparse, merged with Reciprocal Rank Fusion, reranked with Cohere
- **3-agent LangGraph graph** — Retriever → Synthesiser → Critic with self-correcting retry loop
- **NLI contradiction detection** — flags conflicting claims across sources before synthesis
- **Citation grounding** — verifies every cited paper actually supports the claim
- **Automated eval harness** — RAGAS metrics with CI-enforced faithfulness gate (≥ 0.82)
- **Streaming API** — FastAPI with SSE streaming, rate limiting, and structured error handling

## Tech Stack

| Layer | Tools |
|-------|-------|
| Ingestion | arXiv API, PubMed Entrez API, Unstructured.io |
| Embeddings | BGE-M3, HuggingFace Inference API |
| Vector DB | PostgreSQL + pgvector |
| Retrieval | BM25, Cohere Rerank, RRF |
| Orchestration | LangGraph, LangChain |
| LLM | GPT-4o-mini |
| Contradiction | DeBERTa NLI (cross-encoder/nli-deberta-v3-small) |
| API | FastAPI, Uvicorn, Docker |
| Eval | RAGAS, GitHub Actions CI |

## Eval Results

| Metric | Score | Gate |
|--------|-------|------|
| Faithfulness | 0.874 | ≥ 0.82 |
| Answer relevancy | 0.801 | ≥ 0.75 |
| Context precision | 0.763 | ≥ 0.70 |
| Context recall | 0.712 | ≥ 0.65 |

## Setup

### Prerequisites
- Python 3.11+
- Docker
- API keys for OpenAI, Cohere, HuggingFace

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

### Run locally

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

### API Usage

```bash
# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What do recent papers say about GLP-1 and neuroinflammation?"}'

# Health check
curl http://localhost:8000/health

# Paper count
curl http://localhost:8000/papers/count

# Interactive docs
open http://localhost:8000/docs
```

### Run Eval Harness

```bash
# Fast sanity check
python eval/run_eval.py --difficulty easy --max 5

# Full eval
python eval/run_eval.py --save-baseline

# CI mode (exits 1 if faithfulness < 0.82)
python eval/run_eval.py --max 15 --ci
```

### Run Tests

```bash
pytest tests/ -v
```

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

## Environment Variables

```bash
POSTGRES_URL=postgresql://user:password@localhost:5432/scirag
NCBI_EMAIL=             # required for PubMed API
NCBI_API_KEY=           # optional, increases rate limit
HF_API_TOKEN=           # huggingface.co/settings/tokens
OPENAI_API_KEY=         # platform.openai.com
COHERE_API_KEY=         # cohere.com
```