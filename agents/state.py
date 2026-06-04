from typing import TypedDict
from retrieval.dense import RetrievedChunk

class AgentState(TypedDict):
    query:            str
    chunks:           list[RetrievedChunk]
    retrieval_passes: int
    answer:           str
    citations:        list[str]
    critic_score:     float
    critic_feedback:  str
    passed:           bool
    # New in Step 5
    conflicts:        list[dict]   # serialised ChunkPair objects
    grounding:        dict         # { paper_id: { grounded, label, confidence } }