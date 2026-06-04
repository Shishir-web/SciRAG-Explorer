from agents.graph import scirag_graph
from agents.state import AgentState

def run_query(query: str) -> AgentState:
    """Main entry point. Run the full 3-agent graph for a query.
    Returns the final AgentState with answer, citations, and critic score.
    """
    initial_state: AgentState = {
        "query":            query,
        "chunks":           [],
        "retrieval_passes": 0,
        "answer":           "",
        "citations":        [],
        "critic_score":     0.0,
        "critic_feedback":  "",
        "passed":           False,
    }

    final_state = scirag_graph.invoke(initial_state)
    return final_state