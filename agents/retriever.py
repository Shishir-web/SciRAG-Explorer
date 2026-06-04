from agents.state import AgentState
from retrieval import retrieve

MAX_PASSES = 2


def retriever_node(state: AgentState) -> AgentState:
    """
    Calls hybrid retrieval pipeline.
    On retry passes, broadens the search to surface
    different chunks than the first pass.
    """
    passes = state.get("retrieval_passes", 0)

    dense_k  = 20 + (passes * 10)
    rerank_k = 5  + (passes * 3)

    chunks = retrieve(
        query        = state["query"],
        dense_top_k  = dense_k,
        sparse_top_k = dense_k,
        rerank_top_k = rerank_k,
    )

    return {
        **state,
        "chunks":           chunks,
        "retrieval_passes": passes + 1,
    }