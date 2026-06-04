import json
from dataclasses import dataclass, asdict
from pathlib import Path
from eval.ragas_runner import EvalResult

# Thresholds — these are the CI gates
THRESHOLDS = {
    "faithfulness":       0.82,   # hard gate — CI fails below this
    "answer_relevancy":   0.75,
    "context_precision":  0.70,
    "context_recall":     0.65,
}

BASELINE_PATH = Path("eval/baseline_scores.json")


@dataclass
class RegressionReport:
    mean_faithfulness:       float
    mean_answer_relevancy:   float
    mean_context_precision:  float
    mean_context_recall:     float
    mean_critic_score:       float
    total_samples:           int
    error_count:             int
    conflict_detection_rate: float
    threshold_failures:      list[str]
    passed:                  bool


def compute_mean(results: list[EvalResult], field: str) -> float:
    values = [
        getattr(r, field) for r in results
        if not r.error and getattr(r, field) is not None
    ]
    return sum(values) / len(values) if values else 0.0


def build_report(results: list[EvalResult]) -> RegressionReport:
    """Compute aggregate metrics and check against thresholds."""
    means = {k: compute_mean(results, k) for k in THRESHOLDS}

    threshold_failures = [
        f"{metric}={means[metric]:.3f} < {threshold}"
        for metric, threshold in THRESHOLDS.items()
        if means[metric] < threshold
    ]

    conflict_samples = [r for r in results if r.has_conflicts]

    return RegressionReport(
        mean_faithfulness       = means["faithfulness"],
        mean_answer_relevancy   = means["answer_relevancy"],
        mean_context_precision  = means["context_precision"],
        mean_context_recall     = means["context_recall"],
        mean_critic_score       = compute_mean(results, "critic_score"),
        total_samples           = len(results),
        error_count             = sum(1 for r in results if r.error),
        conflict_detection_rate = len(conflict_samples) / len(results)
                                  if results else 0.0,
        threshold_failures      = threshold_failures,
        passed                  = len(threshold_failures) == 0,
    )


def save_baseline(report: RegressionReport):
    """Persist current scores as the new baseline."""
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_PATH, "w") as f:
        json.dump(asdict(report), f, indent=2)
    print(f"Baseline saved to {BASELINE_PATH}")


def load_baseline() -> RegressionReport | None:
    if not BASELINE_PATH.exists():
        return None
    with open(BASELINE_PATH) as f:
        data = json.load(f)
    return RegressionReport(**data)


def print_report(report: RegressionReport):
    print("\n" + "="*55)
    print("SCIRAG EVAL REPORT")
    print("="*55)
    print(f"  Samples evaluated : {report.total_samples}")
    print(f"  Errors            : {report.error_count}")
    print(f"  Conflict rate     : {report.conflict_detection_rate:.1%}")
    print()
    print(f"  Faithfulness      : {report.mean_faithfulness:.3f}  "
          f"(gate: {THRESHOLDS['faithfulness']})")
    print(f"  Answer relevancy  : {report.mean_answer_relevancy:.3f}  "
          f"(gate: {THRESHOLDS['answer_relevancy']})")
    print(f"  Context precision : {report.mean_context_precision:.3f}  "
          f"(gate: {THRESHOLDS['context_precision']})")
    print(f"  Context recall    : {report.mean_context_recall:.3f}  "
          f"(gate: {THRESHOLDS['context_recall']})")
    print(f"  Critic score      : {report.mean_critic_score:.3f}")
    print()

    if report.threshold_failures:
        print("  ✗ THRESHOLD FAILURES:")
        for f in report.threshold_failures:
            print(f"    - {f}")
    else:
        print("  ✓ All thresholds passed")

    print(f"\n  RESULT: {'PASS ✓' if report.passed else 'FAIL ✗'}")
    print("="*55 + "\n")