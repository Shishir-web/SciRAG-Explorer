from contradiction.nli_detector import classify_pair, truncate, get_nli_pipeline
from retrieval.dense import RetrievedChunk


def check_citation_grounding(
    answer: str,
    chunks: list[RetrievedChunk],
) -> dict[str, dict]:
    """
    For each paper_id cited in the answer, find its chunks and
    verify the answer text is entailed by at least one chunk.
    Returns { paper_id: { grounded, label, confidence } }
    """
    paper_chunks: dict[str, list[RetrievedChunk]] = {}
    for c in chunks:
        paper_chunks.setdefault(c.paper_id, []).append(c)

    results: dict[str, dict] = {}

    for paper_id, paper_chunk_list in paper_chunks.items():
        if paper_id not in answer:
            continue

        best_label      = "neutral"
        best_confidence = 0.0

        for chunk in paper_chunk_list:
            label, conf = classify_pair(
                chunk.chunk_text, truncate(answer, 300)
            )
            if conf > best_confidence:
                best_label      = label
                best_confidence = conf

        results[paper_id] = {
            "grounded":   best_label == "entailment",
            "label":      best_label,
            "confidence": best_confidence,
        }

    return results