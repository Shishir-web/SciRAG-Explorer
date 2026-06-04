import pytest
from unittest.mock import patch, MagicMock
from agents.state import AgentState
from agents.retriever import retriever_node
from agents.critic import critic_node, PASS_THRESHOLD
from agents.graph import should_retry, build_graph
from retrieval.dense import RetrievedChunk


def make_state(**overrides) -> AgentState:
    base: AgentState = {
        "query":            "test query",
        "chunks":           [],
        "retrieval_passes": 0,
        "answer":           "test answer",
        "citations":        [],
        "critic_score":     0.0,
        "critic_feedback":  "",
        "passed":           False,
        "conflicts":        [],
        "grounding":        {},
    }
    return {**base, **overrides}


def make_chunk(n=1):
    return RetrievedChunk(
        chunk_id=f"c{n}", paper_id=f"paper:{n}",
        section="abstract", chunk_text="GLP-1 reduces IL-6.",
        score=0.9, rank=n,
    )


# --- should_retry routing ---

def test_should_retry_returns_end_when_passed():
    state = make_state(passed=True)
    assert should_retry(state) == "end"

def test_should_retry_returns_retriever_when_not_passed():
    state = make_state(passed=False)
    assert should_retry(state) == "retriever"


# --- Retriever node ---

def test_retriever_increments_passes():
    mock_chunks = [make_chunk(1), make_chunk(2)]
    with patch("agents.retriever.retrieve", return_value=mock_chunks):
        result = retriever_node(make_state(retrieval_passes=0))
    assert result["retrieval_passes"] == 1
    assert len(result["chunks"]) == 2

def test_retriever_widens_search_on_retry():
    mock_chunks = [make_chunk(1)]
    with patch("agents.retriever.retrieve", return_value=mock_chunks) as mock_ret:
        retriever_node(make_state(retrieval_passes=1))
        call_kwargs = mock_ret.call_args.kwargs
        assert call_kwargs["dense_top_k"] > 20


# --- Critic node ---

def test_critic_sets_passed_true_above_threshold():
    mock_response = MagicMock()
    mock_response.content = (
        '{"faithfulness":0.9,"coverage":0.85,'
        '"citation_use":0.9,"overall":0.88,'
        '"feedback":"Good answer."}'
    )
    with patch("agents.critic.ChatOpenAI") as mock_llm_class:
        mock_llm_class.return_value.invoke.return_value = mock_response
        result = critic_node(make_state(chunks=[make_chunk()]))
    assert result["passed"] is True
    assert result["critic_score"] >= PASS_THRESHOLD

def test_critic_sets_passed_false_below_threshold():
    mock_response = MagicMock()
    mock_response.content = (
        '{"faithfulness":0.4,"coverage":0.5,'
        '"citation_use":0.4,"overall":0.43,'
        '"feedback":"Missing citations."}'
    )
    with patch("agents.critic.ChatOpenAI") as mock_llm_class:
        mock_llm_class.return_value.invoke.return_value = mock_response
        result = critic_node(make_state(chunks=[make_chunk()]))
    assert result["passed"] is False

def test_critic_force_exits_after_max_passes():
    mock_response = MagicMock()
    mock_response.content = (
        '{"faithfulness":0.3,"coverage":0.3,'
        '"citation_use":0.3,"overall":0.30,'
        '"feedback":"Poor."}'
    )
    with patch("agents.critic.ChatOpenAI") as mock_llm_class:
        mock_llm_class.return_value.invoke.return_value = mock_response
        result = critic_node(make_state(
            chunks=[make_chunk()],
            retrieval_passes=2,
        ))
    assert result["passed"] is True


# --- Full graph smoke test ---

def test_graph_compiles():
    graph = build_graph()
    assert graph is not None

def test_full_graph_returns_answer():
    mock_chunks = [make_chunk(1), make_chunk(2)]
    mock_answer = MagicMock()
    mock_answer.content = "GLP-1 reduces neuroinflammation [paper:1]."
    mock_critic = MagicMock()
    mock_critic.content = (
        '{"faithfulness":0.9,"coverage":0.9,'
        '"citation_use":0.9,"overall":0.9,'
        '"feedback":"Excellent."}'
    )

    with patch("agents.retriever.retrieve",                    return_value=mock_chunks), \
         patch("agents.synthesiser.ChatOpenAI")             as mock_synth_llm, \
         patch("agents.critic.ChatOpenAI")                  as mock_critic_llm, \
         patch("agents.synthesiser.detect_contradictions",   return_value=[]), \
         patch("agents.synthesiser.check_citation_grounding",return_value={}):

        mock_synth_llm.return_value.invoke.return_value  = mock_answer
        mock_critic_llm.return_value.invoke.return_value = mock_critic

        from agents import run_query
        result = run_query("GLP-1 neuroinflammation")

    assert result["answer"] != ""
    assert result["critic_score"] >= PASS_THRESHOLD
    assert result["passed"] is True