import pytest
from unittest.mock import patch
from contradiction.nli_detector import (
    detect_contradictions, ChunkPair, CONTRADICTION_THRESHOLD
)
from contradiction.grounding import check_citation_grounding
from contradiction.report    import (
    build_contradiction_report, build_grounding_report
)
from retrieval.dense import RetrievedChunk


def make_chunk(chunk_id, paper_id, text):
    return RetrievedChunk(
        chunk_id=chunk_id, paper_id=paper_id,
        section="results", chunk_text=text,
        score=0.9, rank=1,
    )


CHUNK_A = make_chunk("c1", "paper:001",
    "GLP-1 receptor agonists significantly reduced IL-6 levels "
    "by 34% in treated patients (p < 0.001).")

CHUNK_B = make_chunk("c2", "paper:002",
    "No significant reduction in IL-6 was observed in patients "
    "receiving GLP-1 receptor agonist therapy (p = 0.43).")

CHUNK_C = make_chunk("c3", "paper:003",
    "The study enrolled 120 adult participants aged 45-65.")


# --- NLI detector tests ---

def test_detect_contradictions_returns_conflicts():
    """Mock the NLI pipeline to return a contradiction label."""
    mock_result = {
        "labels": ["contradiction", "neutral", "entailment"],
        "scores": [0.92, 0.05, 0.03],
        "sequence": "test",
    }
    with patch("contradiction.nli_detector.get_nli_pipeline") as mock_pipe:
        mock_pipe.return_value = lambda **kwargs: mock_result
        conflicts = detect_contradictions([CHUNK_A, CHUNK_B])
    assert len(conflicts) >= 1
    assert all(p.is_conflict for p in conflicts)

def test_same_paper_chunks_skipped():
    """Chunks from the same paper should never be compared."""
    same_paper = [
        make_chunk("c1", "paper:001", "IL-6 decreased significantly."),
        make_chunk("c2", "paper:001", "TNF-alpha also decreased."),
    ]
    mock_result = {
        "labels": ["contradiction", "neutral", "entailment"],
        "scores": [0.99, 0.005, 0.005],
        "sequence": "test",
    }
    with patch("contradiction.nli_detector.get_nli_pipeline") as mock_pipe:
        mock_pipe.return_value = lambda **kwargs: mock_result
        conflicts = detect_contradictions(same_paper)
    # Same paper — should return no conflicts regardless of NLI score
    assert len(conflicts) == 0

def test_low_confidence_not_flagged():
    """Contradictions below threshold should not be flagged."""
    mock_result = {
        "labels": ["contradiction", "neutral", "entailment"],
        "scores": [0.50, 0.30, 0.20],   # below CONTRADICTION_THRESHOLD
        "sequence": "test",
    }
    with patch("contradiction.nli_detector.get_nli_pipeline") as mock_pipe:
        mock_pipe.return_value = lambda **kwargs: mock_result
        conflicts = detect_contradictions([CHUNK_A, CHUNK_B])
    assert len(conflicts) == 0


# --- Grounding checker tests ---

def test_grounding_marks_entailed_as_grounded():
    answer = "IL-6 was reduced [paper:001]."
    mock_result = {
        "labels": ["entailment", "neutral", "contradiction"],
        "scores": [0.88, 0.08, 0.04],
        "sequence": "test",
    }
    with patch("contradiction.grounding.get_nli_pipeline") as mock_pipe:
        mock_pipe.return_value = lambda **kwargs: mock_result
        result = check_citation_grounding(answer, [CHUNK_A])
    assert result["paper:001"]["grounded"] is True

def test_grounding_marks_contradiction_as_ungrounded():
    answer = "IL-6 was reduced [paper:002]."
    mock_result = {
        "labels": ["contradiction", "neutral", "entailment"],
        "scores": [0.87, 0.09, 0.04],
        "sequence": "test",
    }
    with patch("contradiction.grounding.get_nli_pipeline") as mock_pipe:
        mock_pipe.return_value = lambda **kwargs: mock_result
        result = check_citation_grounding(answer, [CHUNK_B])
    assert result["paper:002"]["grounded"] is False

def test_uncited_papers_not_checked():
    """Papers not mentioned in the answer should not appear in results."""
    answer = "Some generic finding."   # no paper_id cited
    result = check_citation_grounding(answer, [CHUNK_A, CHUNK_B])
    assert len(result) == 0


# --- Report builder tests ---

def test_contradiction_report_empty_when_no_conflicts():
    assert build_contradiction_report([]) == ""

def test_contradiction_report_contains_paper_ids():
    conflict = ChunkPair(
        chunk_id_a="c1", chunk_id_b="c2",
        paper_id_a="paper:001", paper_id_b="paper:002",
        text_a="IL-6 decreased.", text_b="IL-6 did not decrease.",
        label="contradiction", confidence=0.91, is_conflict=True,
    )
    report = build_contradiction_report([conflict])
    assert "paper:001" in report
    assert "paper:002" in report
    assert "CONTRADICTION" in report

def test_grounding_report_empty_when_all_grounded():
    grounding = {
        "paper:001": {"grounded": True, "label": "entailment", "confidence": 0.9},
    }
    assert build_grounding_report(grounding) == ""

def test_grounding_report_flags_ungrounded():
    grounding = {
        "paper:001": {"grounded": False, "label": "contradiction", "confidence": 0.85},
    }
    report = build_grounding_report(grounding)
    assert "paper:001" in report
    assert "WEAK GROUNDING" in report