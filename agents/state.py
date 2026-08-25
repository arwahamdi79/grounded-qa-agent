from typing import TypedDict, List, Optional


class Passage(TypedDict):
    text: str
    source: str
    title: str


class QAState(TypedDict, total=False):
    query: str                     # original user question
    passages: List[Passage]        # retrieved context, refreshed each researcher pass
    draft_answer: str              # researcher's current draft
    refused: bool                  # True if researcher decided context is insufficient
    verdict: str                   # "APPROVED" | "REJECTED"
    verdict_reason: str            # reviewer's justification
    unsupported_claims: List[str]  # specific claims the reviewer flagged
    reviewer_feedback: str         # instructions sent back to researcher on rejection
    loop_count: int                # number of researcher<->reviewer round trips so far
    final_answer: str              # answer actually shown to the user
    final_verdict: str             # verdict shown to the user (may note "approved after revision")
