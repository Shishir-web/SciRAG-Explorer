import json
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from agents.state import AgentState


CRITIC_LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

CRITIC_SYSTEM = """You are an expert scientific fact-checker.
Evaluate the answer against the provided context chunks.

Score the answer on three dimensions (0.0–1.0 each):
1. faithfulness   — every claim is supported by a cited chunk
2. coverage       — the answer addresses all key aspects of the question
3. citation_use   — citations are accurate and not hallucinated

Respond ONLY with valid JSON in this exact format:
{
  "faithfulness": 0.0,
  "coverage": 0.0,
  "citation_use": 0.0,
  "overall": 0.0,
  "feedback": "one sentence explanation"
}"""

PASS_THRESHOLD = 0.70     # minimum overall score to pass

def critic_node(state: AgentState) -> AgentState:
    context_summary = "\n".join(
        f"[{c.paper_id}]: {c.chunk_text[:200]}..." 
        for c in state["chunks"]
    )

    message = [
        SystemMessage(content=CRITIC_SYSTEM),
        HumanMessage(content=(
            f"Question: {state['query']}\n\n"
            f"Answer: {state['answer']}\n\n"
            f"Context:\n{context_summary}"
        )),
    ]

    response = CRITIC_LLM.invoke(message)

    #Parse JSON verdict
    try:
        raw = response.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        verdict = json.loads(raw.strip())
    except (json.JSONDecodeError, IndexError):
        #Fallback if LLM output is malformed
        verdict = {
            "faithfulness": 0.0,
            "coverage": 0.0,
            "citation_use": 0.0,
            "overall": 0.0,
            "feedback": "could not parse critic response.",
        }
    
    overall = verdict.get("overall", 0.0)
    passed = (
        overall >= PASS_THRESHOLD
        or state.get("retrieval_passes", 0) >= 2 # force exit after 2 pasess
    )

    return {
        **state,
        "critic_score": overall,
        "critic_feedback": verdict.get("feedback", ""),
        "passed": passed,
    }