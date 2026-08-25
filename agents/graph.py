"""
Builds the LangGraph StateGraph implementing the active Researcher <-> Reviewer
handoff loop (not a linear pipeline):

    START -> researcher -> reviewer --(REJECTED & loops remain)--> researcher
                               |
                               +--(APPROVED, or loop budget exhausted)--> finalize -> END

The reviewer can route control back to the researcher at most
config.MAX_REVIEW_LOOPS times. If the answer is still rejected after the
budget is exhausted, we surface it to the user as "unverified" rather than
silently presenting it as confirmed — the system never claims an answer is
grounded when the Reviewer couldn't confirm it.
"""
from langgraph.graph import StateGraph, END
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

import config
from agents.state import QAState
from agents.researcher import make_researcher_node
from agents.reviewer import make_reviewer_node


def _finalize_node(state: QAState) -> QAState:
    draft = state.get("draft_answer", "")
    verdict = state.get("verdict", "REJECTED")

    if state.get("refused"):
        final_answer = draft.split("REFUSE:", 1)[-1].strip()
        final_answer = f"I can't answer that from the ingested documentation. {final_answer}"
        final_verdict = "REFUSED (correctly, per reviewer)" if verdict == "APPROVED" else "REFUSED"
    elif verdict == "APPROVED":
        final_answer = draft
        final_verdict = "APPROVED"
    else:
        # Loop budget exhausted and still rejected: don't present as confidently
        # grounded. Be transparent with the user instead of hiding the risk.
        final_answer = (
            draft
            + "\n\n⚠️ Note: the Reviewer could not fully verify all claims above "
              "against the retrieved documentation after revision. Treat this "
              "answer as unverified."
        )
        final_verdict = f"UNVERIFIED — {state.get('verdict_reason', 'reviewer rejected the revision')}"

    return {**state, "final_answer": final_answer, "final_verdict": final_verdict}


def _route_after_review(state: QAState) -> str:
    if state.get("verdict") == "APPROVED":
        return "finalize"
    if state.get("loop_count", 0) <= config.MAX_REVIEW_LOOPS:
        return "researcher"  # active handoff back to the researcher
    return "finalize"


def get_vectorstore() -> QdrantVectorStore:
    client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    embeddings = HuggingFaceEmbeddings(
    model_name=config.EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
    return QdrantVectorStore(
        client=client,
        collection_name=config.QDRANT_COLLECTION,
        embedding=embeddings,
    )


def build_graph():
    vectorstore = get_vectorstore()

    graph = StateGraph(QAState)
    graph.add_node("researcher", make_researcher_node(vectorstore, config.RETRIEVAL_K))
    graph.add_node("reviewer", make_reviewer_node())
    graph.add_node("finalize", _finalize_node)

    graph.set_entry_point("researcher")
    graph.add_edge("researcher", "reviewer")
    graph.add_conditional_edges(
        "reviewer",
        _route_after_review,
        {"researcher": "researcher", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)

    return graph.compile()


def run_qa(query: str) -> QAState:
    app = build_graph()
    initial_state: QAState = {"query": query, "loop_count": 0}
    return app.invoke(initial_state)
