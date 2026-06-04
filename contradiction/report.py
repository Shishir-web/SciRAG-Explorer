from contradiction.nli_detector import ChunkPair

def build_contradiction_report(conflicts: list[ChunkPair]) -> str:
    """
    Format detected contradictions into a prompt-injectable string.
    Returns empty string if no conflicts.
    """
    if not conflicts:
        return ""
    
    lines = [
        f"\n⚠️  DETECTED {len(conflicts)} CONTRADICTION(S) IN SOURCES:\n"
    ]

    for i, pair in enumerate(conflicts, 1):
        lines.append(
            f"Conflict {i}:\n"
            f" [{pair.paper_id_a}] claims: \"{pair.text_a[:200]}...\"\n"
            f" [{pair.paper_id_b}] claims: \"{pair.text_b[:200]}...\"\n"
            f" Confidence: {pair.confidence:.2f}\n"
        )

    lines.append(
        "You MUST acknowledge these contradictions explicitly in your answer. \n"
        "Do not pick one side silently.\n"
    )

    return "\n".join(lines)

def build_grounding_report(grounding: dict[str, dict]) -> str:
    """
    Format citation grounding results into a prompt-injectable warning.
    Only flags papers where grounding check FAILED.
    """
    ungrounded = [
        pid for pid, result in grounding.items()
        if not result["grounded"]
    ]

    if not ungrounded:
        return ""
    
    return(
        f"\n⚠️ WEAK GROUNDING for citations: {ungrounded}.\n"
        "Use these citations cautiously or omit them if unsupported.\n"
    )
