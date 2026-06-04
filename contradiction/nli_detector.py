from dataclasses import dataclass
from functools import lru_cache
from transformers import pipeline

NLI_MODEL = "cross-encoder/nli-deberta-v3-small"

CONTRADICTION_THRESHOLD = 0.75
MAX_PAIRS               = 20


@dataclass
class ChunkPair:
    chunk_id_a:   str
    chunk_id_b:   str
    paper_id_a:   str
    paper_id_b:   str
    text_a:       str
    text_b:       str
    label:        str
    confidence:   float
    is_conflict:  bool


@lru_cache(maxsize=1)
def get_nli_pipeline():
    return pipeline(
        "zero-shot-classification",
        model=NLI_MODEL,
        device=-1,
    )


def truncate(text: str, max_chars: int = 400) -> str:
    return text[:max_chars] if len(text) > max_chars else text


def classify_pair(text_a: str, text_b: str) -> tuple[str, float]:
    """
    Run NLI on a (premise, hypothesis) pair.
    Returns (label, confidence).
    """
    nli    = get_nli_pipeline()
    result = nli(
        sequences          = truncate(text_a),
        candidate_labels   = ["entailment", "neutral", "contradiction"],
        hypothesis_template= "{}",
        multi_label        = False,
    )

    top_label = result["labels"][0]
    top_score = result["scores"][0]

    return top_label, top_score


def detect_contradictions(chunks: list) -> list[ChunkPair]:
    """
    Run pairwise NLI across all retrieved chunks.
    Only compares chunks from DIFFERENT papers.
    Returns list of ChunkPair objects filtered to conflicts only.
    """
    pairs:    list[ChunkPair] = []
    compared: int             = 0

    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            if compared >= MAX_PAIRS:
                break

            a, b = chunks[i], chunks[j]

            # Skip same-paper comparisons
            if a.paper_id == b.paper_id:
                continue

            label, confidence = classify_pair(a.chunk_text, b.chunk_text)
            is_conflict = (
                label == "contradiction"
                and confidence >= CONTRADICTION_THRESHOLD
            )

            pairs.append(ChunkPair(
                chunk_id_a  = a.chunk_id,
                chunk_id_b  = b.chunk_id,
                paper_id_a  = a.paper_id,
                paper_id_b  = b.paper_id,
                text_a      = a.chunk_text,
                text_b      = b.chunk_text,
                label       = label,
                confidence  = confidence,
                is_conflict = is_conflict,
            ))
            compared += 1

    return [p for p in pairs if p.is_conflict]