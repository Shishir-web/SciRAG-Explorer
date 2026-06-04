from contradiction.nli_detector import classify_pair, truncate
from retrieval.dense import RetrievedChunk

def check_citation_grounding(
        answer:    str,
        chunks:    list[RetrievedChunk],
) -> dict[str, dict]:
    """ For each paper_id cited in the answer, find its chunks and
    verify the answer text is entailed by at least one chunk.

    Returns a dict: { paper_id: { "grounded": bool, "confidence": float } }
    """
    # Build lookup: paper_id -> list of chunks
    paper_chunks: dict[str, list[RetrievedChunk]] = {}
    for c in chunks:
        paper_chunks.setdefault(c.paper_id, []).append(c)

    results: dict[str, dict] = {}

    for paper_id, paper_chunk_list in paper_chunks.items():
        if paper_id not in answer:
            continue    # paper wasn't cited - skip

        # Try each chunk as the premis, answer as hypothesis
        # If ANY chunk entails the answer, citation is grounded
        best_label      = "neutral"
        best_confidence = 0.0

        for chunk in paper_chunk_list:
            label, conf = classify_pair(chunk.chunk_text, truncate(answer, 3000))
            if conf > best_confidence:
                best_label      = label
                best_confidence = conf

        results[paper_id] = {
            "grounded":   best_label == "entailment",
            "label":      best_label,
            "confidence": best_confidence,
        }
    
    return results