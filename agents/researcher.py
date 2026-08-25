"""
Agent 1: Researcher.

Responsibilities:
1. Search the remote Qdrant collection for passages relevant to the user's
   query (and, on a re-run after reviewer rejection, relevant to the
   reviewer's feedback too).
2. Draft an answer grounded ONLY in the retrieved passages, with inline
   citation markers like [1], [2] that map to the returned source list.
3. If the retrieved passages do not support an answer, explicitly refuse
   rather than guessing.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from agents.llm import get_chat_model
from agents.state import QAState

SYSTEM_PROMPT = """You are the Researcher agent in a grounded Q&A system about \
LangChain and Qdrant documentation.

You are given a user question and a numbered list of retrieved documentation \
passages. Your job:

1. Answer the question using ONLY information present in the passages below.
2. Every factual claim in your answer must be followed by a citation marker \
like [1] or [2] referencing the passage number it came from.
3. If the passages do not contain enough information to answer the question, \
you MUST refuse. Start your answer with exactly "REFUSE:" followed by a one \
sentence explanation of what's missing. Do not guess or use outside knowledge.
4. Do not fabricate passage numbers. Do not cite a passage for a claim it \
doesn't support.
5. Be concise and technically precise.

If reviewer feedback is provided, treat it as instructions for what to fix or \
re-search for in this revision."""

HUMAN_TEMPLATE = """Question: {query}

{feedback_block}

Retrieved passages:
{passages_block}

Write the grounded answer now (or REFUSE: ... if unsupported)."""


def _format_passages(passages):
    lines = []
    for i, p in enumerate(passages, start=1):
        snippet = p["text"].strip().replace("\n", " ")
        if len(snippet) > 900:
            snippet = snippet[:900] + "..."
        lines.append(f"[{i}] (source: {p['source']})\n{snippet}")
    return "\n\n".join(lines) if lines else "(no passages retrieved)"


def build_search_query(state: QAState) -> str:
    """On a re-run after rejection, fold reviewer feedback into the search query
    so the researcher can pull different/additional passages instead of looping
    on the same context."""
    if state.get("reviewer_feedback"):
        return f"{state['query']} — {state['reviewer_feedback']}"
    return state["query"]


def retrieve(vectorstore, state: QAState, k: int) -> list:
    query = build_search_query(state)
    docs = vectorstore.similarity_search(query, k=k)
    passages = []
    for d in docs:
        passages.append(
            {
                "text": d.page_content,
                "source": d.metadata.get("source", "unknown"),
                "title": d.metadata.get("title", d.metadata.get("source", "unknown")),
            }
        )
    return passages


def make_researcher_node(vectorstore, k: int):
    llm = get_chat_model(temperature=0.0)
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", HUMAN_TEMPLATE)]
    )
    chain = prompt | llm | StrOutputParser()

    def researcher_node(state: QAState) -> QAState:
        passages = retrieve(vectorstore, state, k)

        feedback_block = ""
        if state.get("reviewer_feedback"):
            feedback_block = (
                f"The reviewer rejected your previous draft with this feedback, "
                f"address it in this revision:\n{state['reviewer_feedback']}"
            )

        draft = chain.invoke(
            {
                "query": state["query"],
                "feedback_block": feedback_block,
                "passages_block": _format_passages(passages),
            }
        )

        refused = draft.strip().upper().startswith("REFUSE:")

        return {
            **state,
            "passages": passages,
            "draft_answer": draft,
            "refused": refused,
        }

    return researcher_node
