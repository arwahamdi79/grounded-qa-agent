"""
Agent 2: Reviewer.

Responsibilities:
1. Check every claim in the Researcher's draft against the retrieved passages.
2. Return a verdict: APPROVED or REJECTED.
3. If REJECTED, list the unsupported claims and give concrete feedback the
   Researcher can act on (e.g. "re-search for X", "passage 2 doesn't say Y").

This node only ever routes back to the Researcher ONCE (enforced by the
graph's conditional edge using loop_count / MAX_REVIEW_LOOPS), so a
persistently unsupported answer still terminates instead of looping forever.
"""
import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from agents.llm import get_chat_model
from agents.state import QAState

SYSTEM_PROMPT = """You are the Reviewer agent in a grounded Q&A system. You \
audit the Researcher's draft answer against the retrieved passages it was \
given, and you are the last line of defense against unsupported claims.

Respond with ONLY a JSON object (no markdown fences, no prose) with this \
exact shape:
{{
  "verdict": "APPROVED" or "REJECTED",
  "reason": "one or two sentence justification",
  "unsupported_claims": ["claim 1", "claim 2"],
  "feedback_for_researcher": "concrete instructions for what to fix or re-search, empty string if approved"
}}

Rules:
- If the draft starts with "REFUSE:", APPROVE it as long as the refusal is \
reasonable given the passages (the researcher correctly declined) — a proper \
refusal is a correct outcome, not a failure.
- REJECT if any claim in the draft is not actually supported by the cited \
passage, if a citation number doesn't match what it's citing, or if the \
draft states something with more confidence than the passages support.
- Be strict but fair: minor phrasing differences are fine; factual drift is not."""

HUMAN_TEMPLATE = """Question: {query}

Retrieved passages:
{passages_block}

Researcher's draft answer:
{draft_answer}

Return the JSON verdict now."""


def _format_passages(passages):
    lines = []
    for i, p in enumerate(passages, start=1):
        snippet = p["text"].strip().replace("\n", " ")
        if len(snippet) > 900:
            snippet = snippet[:900] + "..."
        lines.append(f"[{i}] (source: {p['source']})\n{snippet}")
    return "\n\n".join(lines) if lines else "(no passages retrieved)"


def _safe_parse(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fail closed: if we can't parse the verdict, treat as rejected so a
        # human/UI sees "unverified" rather than silently approving.
        return {
            "verdict": "REJECTED",
            "reason": "Reviewer output could not be parsed; failing closed.",
            "unsupported_claims": [],
            "feedback_for_researcher": "Re-draft the answer with clear, simple citations.",
        }


def make_reviewer_node():
    llm = get_chat_model(temperature=0.0)
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", HUMAN_TEMPLATE)]
    )
    chain = prompt | llm | StrOutputParser()

    def reviewer_node(state: QAState) -> QAState:
        raw = chain.invoke(
            {
                "query": state["query"],
                "passages_block": _format_passages(state.get("passages", [])),
                "draft_answer": state.get("draft_answer", ""),
            }
        )
        parsed = _safe_parse(raw)

        verdict = parsed.get("verdict", "REJECTED").upper()
        if verdict not in ("APPROVED", "REJECTED"):
            verdict = "REJECTED"

        return {
            **state,
            "verdict": verdict,
            "verdict_reason": parsed.get("reason", ""),
            "unsupported_claims": parsed.get("unsupported_claims", []),
            "reviewer_feedback": parsed.get("feedback_for_researcher", ""),
            "loop_count": state.get("loop_count", 0) + 1,
        }

    return reviewer_node
