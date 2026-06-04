from agents.state import AgentState
from retrieval import retrieve

MAX_PASSES = 2   # cap retired to avoid infinite loops

def retrieve_node(state: AgentState) -> AgentState:
    """Calls hybrid retrieval pipeline.
    On retry passes, broadens the search (more top_k) to surface
    different chunks than the first pass.
    """
    passes = state.get("retrival_passes", 0)

    # widen search on retries to surface different chunks
    dense_k = 20 + (passes * 10)
    rerank_k = 5 + (passes * 3)

    chunks = retrieve(
        query         = state["query"],
        dense_top_k   =dense_k,
        sparse_top_k  = dense_k,
        rerank_top_k  = rerank_k,
    )

    return {
        **state,
        "chunks": chunks,
        "retrieval_passes": passes + 1,
    }