import os
import sys
import argparse
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from eval.golden_dataset import load_golden_dataset, filter_by_difficulty
from eval.ragas_runner   import run_full_eval
from eval.regression     import (
    build_report, print_report,
    save_baseline, load_baseline,
)


def parse_args():
    p = argparse.ArgumentParser(description="SciRAG eval harness")
    p.add_argument("--max",      type=int,  default=None,
                   help="Limit number of samples (for fast CI runs)")
    p.add_argument("--save-baseline", action="store_true",
                   help="Save scores as new baseline after run")
    p.add_argument("--difficulty", choices=["easy","medium","hard"],
                   default=None, help="Filter by difficulty")
    p.add_argument("--ci", action="store_true",
                   help="CI mode: exit 1 on threshold failure")
    return p.parse_args()


def main():
    args    = parse_args()
    samples = load_golden_dataset()

    if args.difficulty:
        samples = filter_by_difficulty(samples, args.difficulty)

    print(f"\nRunning eval on {min(len(samples), args.max or len(samples))} samples...")

    results = run_full_eval(samples, max_samples=args.max)
    report  = build_report(results)

    print_report(report)

    if args.save_baseline:
        save_baseline(report)

    # Compare against saved baseline if it exists
    baseline = load_baseline()
    if baseline:
        delta = report.mean_faithfulness - baseline.mean_faithfulness
        print(f"Faithfulness vs baseline: {delta:+.3f}")
        if delta < -0.05:
            print("⚠️  Faithfulness regressed more than 0.05 from baseline")

    if args.ci and not report.passed:
        print("CI gate failed. Exiting with code 1.")
        sys.exit(1)


if __name__ == "__main__":
    main()