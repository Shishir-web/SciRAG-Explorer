from langgraph.graph import StateGraph, END
from agents.state       import AgentState
from agents.retriever   import retriever_node
from agents.synthesiser import synthesiser_node
from agents.critic      import critic_node


def should_retry(state: AgentState) -> str:
    """
    Conditional edge function.
    Returns the name of the next node to route to.
    """

    if state["passed"]:
        return "end"
    return "retriever" # loop back for another pass

def build_graph() -> StateGraph:
    graph = StateGraph()

    # Register nodes
    graph.add_node("retriever",   retriever_node)
    graph.add_node("synthesiser", synthesiser_node)
    graph.add_node("critic",      critic_node)

    # Linear edges
    graph.add_edge("retriever", "synthesiser")
    graph.add_edge("synthesiser", "critic")

    # Conditional edge from critic back to retriever for another pass
    graph.add_conditonal_edges(
        "critic",
        should_retry,
        {
            "end":       END,
            "retriever": "retriever",
        }
    )

    # Entry point
    graph.set_entry_point("retriever")

    return graph.compile()

# Module-level compiled graph - import this everywhere
scirag_graph = build_graph()
