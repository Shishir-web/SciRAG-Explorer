import os
from dataclasses import dataclass
from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from eval.golden_dataset import GoldenSample
from agents import run_query


@dataclass
class EvalResult:
    sample_id:         str
    query:             str
    answer:            str
    faithfulness:      float
    answer_relevancy:  float
    context_precision: float
    context_recall:    float
    has_conflicts:     bool
    critic_score:      float
    retrieval_passes:  int
    error:             str | None = None


def run_single_eval(sample: GoldenSample) -> EvalResult:
    """
    Run the full agent graph on one golden sample and compute
    RAGAS metrics against the reference answer.
    """
    try:
        state = run_query(sample.query)

        contexts = [c.chunk_text for c in state.get("chunks", [])]
        answer   = state.get("answer", "")

        ragas_data = Dataset.from_dict({
            "question":     [sample.query],
            "answer":       [answer],
            "contexts":     [contexts],
            "ground_truth": [sample.reference_answer],
        })

        llm = LangchainLLMWrapper(
            ChatOpenAI(model="gpt-4o-mini", temperature=0)
        )
        embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(model="text-embedding-3-small")
        )

        scores = evaluate(
            dataset   = ragas_data,
            metrics   = [
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm        = llm,
            embeddings = embeddings,
        )

        df  = scores.to_pandas()
        row = df.iloc[0]

        return EvalResult(
            sample_id         = sample.id,
            query             = sample.query,
            answer            = answer,
            faithfulness      = float(row.get("faithfulness",      0.0)),
            answer_relevancy  = float(row.get("answer_relevancy",  0.0)),
            context_precision = float(row.get("context_precision", 0.0)),
            context_recall    = float(row.get("context_recall",    0.0)),
            has_conflicts     = len(state.get("conflicts", [])) > 0,
            critic_score      = state.get("critic_score", 0.0),
            retrieval_passes  = state.get("retrieval_passes", 1),
        )

    except Exception as e:
        return EvalResult(
            sample_id         = sample.id,
            query             = sample.query,
            answer            = "",
            faithfulness      = 0.0,
            answer_relevancy  = 0.0,
            context_precision = 0.0,
            context_recall    = 0.0,
            has_conflicts     = False,
            critic_score      = 0.0,
            retrieval_passes  = 0,
            error             = str(e),
        )


def run_full_eval(samples: list[GoldenSample],
                  max_samples: int | None = None) -> list[EvalResult]:
    """Run eval on the full golden dataset or a subset for CI speed."""
    if max_samples:
        samples = samples[:max_samples]

    results = []
    for i, sample in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] {sample.id}: {sample.query[:60]}...")
        result = run_single_eval(sample)
        status = f"✓ faith={result.faithfulness:.2f}" if not result.error \
                 else f"✗ {result.error}"
        print(f"    {status}")
        results.append(result)

    return results