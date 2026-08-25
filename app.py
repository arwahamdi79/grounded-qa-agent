"""
Streamlit chat UI for the two-agent grounded Q&A assistant.

Run with:
    streamlit run app.py
"""
import streamlit as st

import config

st.set_page_config(page_title="Grounded Docs Q&A (Researcher + Reviewer)", page_icon="📚")

st.title("📚 Grounded Q&A — LangChain & Qdrant Docs")
st.caption(
    "Two-agent system: a **Researcher** retrieves passages from a remote Qdrant "
    "collection and drafts a cited answer; a **Reviewer** checks every claim "
    "against those passages and can send the draft back for revision before "
    "anything is shown to you."
)

try:
    config.validate_config()
except EnvironmentError as e:
    st.error(str(e))
    st.stop()

from agents.graph import run_qa  # noqa: E402  (imported after config validation)


def render_meta(meta: dict):
    verdict = meta["final_verdict"]
    if verdict.startswith("APPROVED"):
        st.success(f"Reviewer verdict: {verdict}")
    elif verdict.startswith("REFUSED"):
        st.info(f"Reviewer verdict: {verdict}")
    else:
        st.warning(f"Reviewer verdict: {verdict}")

    if meta.get("verdict_reason"):
        st.caption(f"Reviewer note: {meta['verdict_reason']}")

    passages = meta.get("passages", [])
    if passages:
        with st.expander(f"Cited sources ({len(passages)} passages retrieved)"):
            for i, p in enumerate(passages, start=1):
                st.markdown(f"**[{i}]** [{p['title']}]({p['source']})")

    if meta.get("loop_count", 0) > 1:
        st.caption(
            f"↺ The Reviewer sent this back to the Researcher "
            f"{meta['loop_count'] - 1} time(s) before finalizing."
        )


if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay history (Streamlit reruns the whole script on every interaction)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("meta"):
            render_meta(msg["meta"])

query = st.chat_input("Ask a question about LangChain or Qdrant...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Researcher is retrieving passages and drafting an answer..."):
            result = run_qa(query)

        st.markdown(result["final_answer"])
        meta = {
            "final_verdict": result.get("final_verdict", ""),
            "verdict_reason": result.get("verdict_reason", ""),
            "passages": result.get("passages", []),
            "loop_count": result.get("loop_count", 0),
        }
        render_meta(meta)

    st.session_state.messages.append(
        {"role": "assistant", "content": result["final_answer"], "meta": meta}
    )
