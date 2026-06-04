import pytest
import json
import tempfile
from pathlib import Path
from eval.golden_dataset import (
    load_golden_dataset, filter_by_topic,
    filter_by_difficulty, GoldenSample,
)
from eval.regression import (
    build_report, compute_mean,
    RegressionReport, THRESHOLDS,
)
from eval.ragas_runner import EvalResult


# ── Helpers ──────────────────────────────────────────────────

def make_result(sample_id="q001", faithfulness=0.9,
                answer_relevancy=0.85, context_precision=0.80,
                context_recall=0.75, critic_score=0.88,
                has_conflicts=False, error=None):
    return EvalResult(
        sample_id         = sample_id,
        query             = "test query",
        answer            = "test answer",
        faithfulness      = faithfulness,
        answer_relevancy  = answer_relevancy,
        context_precision = context_precision,
        context_recall    = context_recall,
        has_conflicts     = has_conflicts,
        critic_score      = critic_score,
        retrieval_passes  = 1,
        error             = error,
    )


MINIMAL_DATASET = [
    {
        "id": "q001",
        "query": "What is the effect of GLP-1 on IL-6?",
        "reference_answer": "GLP-1 reduces IL-6.",
        "ground_truth_paper_ids": ["pubmed:001"],
        "expected_has_conflict": False,
        "topic": "GLP-1 neuroinflammation",
        "difficulty": "easy",
    },
    {
        "id": "q002",
        "query": "Do studies agree on CRISPR off-target effects?",
        "reference_answer": "Studies disagree on off-target rates.",
        "ground_truth_paper_ids": ["arxiv:001", "pubmed:002"],
        "expected_has_conflict": True,
        "topic": "CRISPR base editing",
        "difficulty": "hard",
    },
]


# ── Dataset loader tests ──────────────────────────────────────

def test_load_golden_dataset_from_tempfile():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(MINIMAL_DATASET, f)
        tmp_path = Path(f.name)

    samples = load_golden_dataset(tmp_path)
    assert len(samples) == 2
    assert samples[0].id == "q001"
    assert samples[1].expected_has_conflict is True

def test_load_raises_on_missing_field():
    bad = [{"id": "q001", "query": "test"}]   # missing most fields
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(bad, f)
        tmp_path = Path(f.name)

    with pytest.raises(ValueError, match="missing"):
        load_golden_dataset(tmp_path)

def test_filter_by_difficulty():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(MINIMAL_DATASET, f)
        tmp_path = Path(f.name)

    samples = load_golden_dataset(tmp_path)
    easy    = filter_by_difficulty(samples, "easy")
    hard    = filter_by_difficulty(samples, "hard")
    assert len(easy) == 1
    assert len(hard) == 1
    assert easy[0].difficulty == "easy"

def test_filter_by_topic():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(MINIMAL_DATASET, f)
        tmp_path = Path(f.name)

    samples = load_golden_dataset(tmp_path)
    glp1    = filter_by_topic(samples, "GLP-1 neuroinflammation")
    assert len(glp1) == 1
    assert glp1[0].topic == "GLP-1 neuroinflammation"


# ── Regression report tests ───────────────────────────────────

def test_report_passes_above_all_thresholds():
    results = [make_result(
        faithfulness=0.90, answer_relevancy=0.85,
        context_precision=0.80, context_recall=0.75,
    )]
    report = build_report(results)
    assert report.passed is True
    assert report.threshold_failures == []

def test_report_fails_below_faithfulness_threshold():
    results = [make_result(faithfulness=0.70)]   # below 0.82
    report  = build_report(results)
    assert report.passed is False
    assert any("faithfulness" in f for f in report.threshold_failures)

def test_report_excludes_errored_samples_from_means():
    results = [
        make_result("q001", faithfulness=0.90),
        make_result("q002", faithfulness=0.0, error="API timeout"),
    ]
    report = build_report(results)
    # Mean should be 0.90, not 0.45 — errored sample excluded
    assert report.mean_faithfulness == pytest.approx(0.90, abs=0.01)
    assert report.error_count == 1

def test_conflict_detection_rate():
    results = [
        make_result("q001", has_conflicts=True),
        make_result("q002", has_conflicts=True),
        make_result("q003", has_conflicts=False),
        make_result("q004", has_conflicts=False),
    ]
    report = build_report(results)
    assert report.conflict_detection_rate == pytest.approx(0.5, abs=0.01)

def test_compute_mean_ignores_errors():
    results = [
        make_result("q001", faithfulness=0.8),
        make_result("q002", faithfulness=0.0, error="timeout"),
        make_result("q003", faithfulness=0.9),
    ]
    mean = compute_mean(results, "faithfulness")
    assert mean == pytest.approx(0.85, abs=0.01)

def test_total_samples_count():
    results = [make_result(f"q00{i}") for i in range(10)]
    report  = build_report(results)
    assert report.total_samples == 10