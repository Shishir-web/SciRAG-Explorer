import pytest
from unittest.mock import patch, MagicMock
from retrieval.rrf import reciprocal_rank_fusion
from retrieval.dense import RetrievedChunk

def make_chunk(chunk_id, rank, score=1.0):
    return RetrievedChunk(
        chunk_id=chunk_id,
        paper_id=f"paper:{chunk_id}",
        section="abstract",
        chunk_text=f"Text for {chunk_id}",
        score=score,
        rank=rank,
    )

# --- RRF tests (no DB/API needed) ---

def test_rrf_combines_two_lists():
    dense  = [make_chunk("A", 1), make_chunk("B", 2), make_chunk("C", 3)]
    sparse = [make_chunk("B", 1), make_chunk("A", 2), make_chunk("D", 3)]
    result = reciprocal_rank_fusion(dense, sparse, top_k=4)
    ids    = [r.chunk_id for r in result]
    # A and B appear in both lists — should rank highest
    assert ids[0] in ("A", "B")
    assert ids[1] in ("A", "B")

def test_rrf_handles_unique_chunks():
    dense  = [make_chunk("A", 1)]
    sparse = [make_chunk("B", 1)]
    result = reciprocal_rank_fusion(dense, sparse, top_k=2)
    assert len(result) == 2

def test_rrf_ranks_are_sequential():
    dense  = [make_chunk("A", 1), make_chunk("B", 2)]
    sparse = [make_chunk("A", 1), make_chunk("B", 2)]
    result = reciprocal_rank_fusion(dense, sparse, top_k=2)
    assert [r.rank for r in result] == [1, 2]

def test_rrf_score_higher_for_overlap():
    """A chunk appearing in both lists should outscore a chunk in only one."""
    dense  = [make_chunk("shared", 1), make_chunk("dense_only", 2)]
    sparse = [make_chunk("shared", 1), make_chunk("sparse_only", 2)]
    result = reciprocal_rank_fusion(dense, sparse, top_k=3)
    assert result[0].chunk_id == "shared"

def test_rrf_top_k_respected():
    dense  = [make_chunk(f"chunk{i}", i+1) for i in range(10)]
    sparse = [make_chunk(f"chunk{i}", i+1) for i in range(10)]
    result = reciprocal_rank_fusion(dense, sparse, top_k=3)
    assert len(result) == 3

# --- Integration smoke test (mocked) ---

def test_retrieve_pipeline_smoke():
    """End-to-end pipeline returns a list of RetrievedChunks."""
    mock_chunks = [make_chunk(f"c{i}", i+1) for i in range(5)]

    with patch("retrieval.dense_search",  return_value=mock_chunks), \
         patch("retrieval.sparse_search", return_value=mock_chunks), \
         patch("retrieval.rerank",        return_value=mock_chunks[:3]):

        from retrieval import retrieve
        results = retrieve("GLP-1 neuroinflammation", rerank_top_k=3)
        assert len(results) == 3
        assert all(isinstance(r, RetrievedChunk) for r in results)