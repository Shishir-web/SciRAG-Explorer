from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from agents.state import AgentState
from retrieval.dense import RetrievedChunk
from contradiction.nli_detector import detect_contradictions
from contradiction.grounding    import check_citation_grounding
from contradiction.report       import (
    build_contradiction_report,
    build_grounding_report,
)

LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

BASE_SYSTEM_PROMPT = """You are a scientific research assistant.
Answer the user's question using ONLY the provided context chunks.
Rules:
- Cite every factual claim with [paper_id] inline.
- If chunks contradict each other, explicitly note the disagreement.
- If the context is insufficient, say so — do not speculate.
- Write in clear, precise scientific prose. No bullet points.
"""

def format_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for c in chunks:
        parts.append(
            f"[{c.paper_id}] (section: {c.section}, "
            f"relevance: {c.score:.2f}):\n{c.chunk_text}"
        )
    return "\n\n---\n\n".join(parts)

def extract_citations(answer: str, chunks: list[RetrievedChunk]) -> list[str]:
    """Extract cited paper_ids from the answer text."""
    cited = []
    for c in chunks:
        if f"[{c.paper_id}]" in answer:
            cited.append(c.paper_id)
    return list(set(cited))  # deduplicate


def synthesiser_node(state: AgentState) -> AgentState:
    chunks = state["chunks"]

    # --- Contradiction detection ---
    conflicts   = detect_contradictions(chunks)
    contra_note = build_contradiction_report(conflicts)

    # --- Build system prompt with injected signals ---
    SYSTEM_PROMPT = BASE_SYSTEM_PROMPT + contra_note

    context = format_context(chunks)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Question: {state['query']}\n\n"
            f"Context:\n{context}"
        )),
    ]

    response = LLM.invoke(messages)
    answer = response.content
    
     # --- Citation grounding check (post-generation) ---
    grounding      = check_citation_grounding(answer, chunks)
    grounding_note = build_grounding_report(grounding)

    # If grounding issues found, append a note to the answer
    if grounding_note:
        answer += (
            "\n\n_Note: Some citations could not be fully verified "
            "against source chunks. Please consult primary sources._"
        )

    citations = extract_citations(answer, chunks)

    return {
        **state,
        "answer":    answer,
        "citations": citations,
        # Store for eval harness in Step 7
        "conflicts":  [vars(c) for c in conflicts],
        "grounding":  grounding,
    }