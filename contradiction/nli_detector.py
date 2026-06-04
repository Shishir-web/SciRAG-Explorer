from dataclasses import dataclass
from functools import lru_cache
from transformers import pipeline

NLI_MODEL = "cross-encoder/nli-debarta-v3-small"
# small but accurate - runs on CPU, ~250MB download on first use

CONTRADICTION_THRESHOLD = 0.5   # score above which we consider the answer contradicted by the context
MAX_PAIRS               = 20    # cap pairwise comparisons for speed

@dataclass
class ChunkPair:
    chunk_id_a:    str
    chunk_id_b:    str
    paper_id_a:    str
    paper_id_b:    str
    text_a:        str
    text_b:        str
    label:         str
    confidence:    float # "entailment", "neutral", "contradiction"
    is_conflict:   bool

@lru_cache(maxsize=1)
def get_nli_pipeline():
    """Load the NLI pipeline once and cache it.
       lru_cache ensures the model loads only on first call.
    """
    return pipeline(
        "zero-shot-classification",
        model=NLI_MODEL,
        device=-1,          # -1 = CPU; set to 0 for GPU
    )

def truncate(text: str, max_chars: int = 400) -> str:
    """NLI works best on shorter premise/hypothesis pairs."""
    return text[:max_chars] if len(text) > max_chars else text


def classify_pair(text_a: str, text_b: str) -> str:
    """Run NLI on a (premise, hypothesis) pair.
    Returns (label, confidence).
    
    We frame it as: does text_b contradict text_a?
    candidate_labels order matters — the model scores each.
    """
    nli = get_nli_pipeline()
    result = nli(
        sequences          = truncate(text_a),
        candidate_labels   = ["entailment", "neutral", "contradiction"],
        hypothesis_template= "{}",
        multi_label        = False,
    )

    # result["Labels"] and result["scores"] are parallel lists
    label_score = dict(zip(result["lables"], result["scores"]))
    top_label   = result["labels"][0]
    top_score   = result["scores"][0]

    return top_label, top_score

def detect_contradictions(chunks: list) -> list[ChunkPair]:
    """Run pairwise NLI across all retrieved chunks.
    Only compares chunks from DIFFERENT papers (same-paper chunks
    are usually consistent by definition).
    
    Returns list of ChunkPair objects, filtered to conflicts only.
    """
    pairs: list[ChunkPair] = []
    compared = 0

    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            if compared >= MAX_PAIRS:
                break

            a, b = chunks[i], chunks[j]

            #Skip same-paper comparisons
            if a.paper_id == b.paper_id:
                continue

            label, confidence = classify_pair(a.chunk_text, b.chunk_text)
            is_conflict = (
                label == "contradiction"
                and confidence >= CONTRADICTION_THRESHOLD
            )

            pairs.append(ChunkPair(
                chunk_id_a = a.chunk_id,
                chunk_id_b = b.chunk_id,
                paper_id_a = a.paper_id,
                paper_id_b = b.paper_id,
                text_a     = a.chunk_text,
                text_b     = b.chunk_text,
                label      = label,
                confidence = confidence,
                is_conflict= is_conflict,
            ))
            compared += 1


    return [p for p in pairs if p.is_conflict]

