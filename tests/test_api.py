import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

MOCK_STATE = {
    "query":            "GLP-1 and neuroinflammation",
    "answer":           "GLP-1 agonists reduce IL-6 [paper:001].",
    "citations":        ["paper:001"],
    "critic_score":     0.88,
    "critic_feedback":  "Well grounded.",
    "retrieval_passes": 1,
    "conflicts":        [],
    "grounding": {
        "paper:001": {
            "grounded":   True,
            "label":      "entailment",
            "confidence": 0.91,
        }
    },
    "passed": True,
    "chunks": [],
}


# ── Health check ─────────────────────────────────────────────

def test_health_endpoint_returns_200():
    with patch("api.main.check_db_health", return_value="ok"):
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_health_endpoint_degraded_when_db_down():
    with patch("api.main.check_db_health", return_value="error: timeout"):
        response = client.get("/health")
    assert response.json()["status"] == "degraded"
    assert response.json()["db"] != "ok"


# ── Query endpoint ────────────────────────────────────────────

def test_query_returns_200_with_valid_input():
    with patch("api.main.run_query", return_value=MOCK_STATE), \
         patch("asyncio.get_event_loop") as mock_loop:
        mock_loop.return_value.run_in_executor = AsyncMock(
            return_value=MOCK_STATE
        )
        response = client.post("/query", json={
            "query": "What do papers say about GLP-1 and inflammation?"
        })
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data

def test_query_response_shape():
    with patch("api.main.run_query", return_value=MOCK_STATE), \
         patch("asyncio.get_event_loop") as mock_loop:
        mock_loop.return_value.run_in_executor = AsyncMock(
            return_value=MOCK_STATE
        )
        response = client.post("/query", json={
            "query": "What do papers say about GLP-1 and inflammation?"
        })
    data = response.json()
    assert isinstance(data["citations"],        list)
    assert isinstance(data["conflicts"],        list)
    assert isinstance(data["has_conflicts"],    bool)
    assert isinstance(data["retrieval_passes"], int)
    assert 0.0 <= data["critic_score"] <= 1.0

def test_query_too_short_returns_422():
    response = client.post("/query", json={"query": "short"})
    assert response.status_code == 422

def test_query_too_long_returns_422():
    response = client.post("/query", json={"query": "x" * 1001})
    assert response.status_code == 422

def test_query_missing_field_returns_422():
    response = client.post("/query", json={})
    assert response.status_code == 422

def test_query_agent_failure_returns_500():
    with patch("api.main.run_query", side_effect=RuntimeError("Graph crashed")), \
         patch("asyncio.get_event_loop") as mock_loop:
        mock_loop.return_value.run_in_executor = AsyncMock(
            side_effect=RuntimeError("Graph crashed")
        )
        response = client.post("/query", json={
            "query": "What do papers say about GLP-1 and inflammation?"
        })
    assert response.status_code == 500

def test_has_conflicts_true_when_conflicts_present():
    state_with_conflict = {
        **MOCK_STATE,
        "conflicts": [{
            "paper_id_a": "paper:001",
            "paper_id_b": "paper:002",
            "confidence": 0.91,
        }],
    }
    with patch("api.main.run_query", return_value=state_with_conflict), \
         patch("asyncio.get_event_loop") as mock_loop:
        mock_loop.return_value.run_in_executor = AsyncMock(
            return_value=state_with_conflict
        )
        response = client.post("/query", json={
            "query": "What do papers say about GLP-1 and inflammation?"
        })
    assert response.json()["has_conflicts"] is True


# ── Paper count endpoint ──────────────────────────────────────

def test_paper_count_endpoint():
    mock_engine = MagicMock()
    mock_session = MagicMock()
    mock_session.__enter__ = lambda s: s
    mock_session.__exit__  = MagicMock(return_value=False)
    mock_session.execute.return_value.scalar.side_effect = [312, 2847]
    mock_engine.return_value = mock_engine

    with patch("api.main.get_engine", return_value=mock_engine), \
         patch("api.main.Session",    return_value=mock_session):
        response = client.get("/papers/count")

    assert response.status_code == 200