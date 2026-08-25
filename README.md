# Grounded Q&A Assistant — Researcher + Reviewer (LangGraph + Qdrant + Streamlit)

A two-agent, grounded question-answering assistant over the official **LangChain**
and **Qdrant** documentation. It refuses to answer anything not supported by
the retrieved context.

- **Agent 1 — Researcher**: searches a remote Qdrant collection and drafts a
  cited answer from the retrieved passages only.
- **Agent 2 — Reviewer**: audits every claim in the draft against those same
  passages, and returns a verdict. If any claim is unsupported, it routes the
  draft **back to the Researcher** (once) with concrete feedback before a
  final answer is ever shown to the user.
- **Orchestration**: a [LangGraph](https://langchain-ai.github.io/langgraph/)
  `StateGraph` with a real conditional edge from Reviewer → Researcher — an
  active handoff loop, not a linear pipeline.

```
        ┌───────────────┐        ┌───────────────┐
 START → │  Researcher   │ ────▶ │   Reviewer     │
        └──────▲────────┘        └──────┬────────┘
               │  REJECTED (loop budget remains)
               └────────────────────────┘
                                          │ APPROVED / budget exhausted
                                          ▼
                                    ┌───────────┐
                                    │ Finalize  │ → END → shown in Streamlit
                                    └───────────┘
```

## Repository layout

```
.
├── app.py                # Streamlit chat UI
├── ingest.py              # Crawls & ingests LangChain + Qdrant docs into Qdrant
├── config.py               # Loads all config/secrets from environment variables
├── agents/
│   ├── state.py             # Shared LangGraph state schema
│   ├── llm.py                 # Chat model factory (OpenAI or Anthropic)
│   ├── researcher.py           # Agent 1
│   ├── reviewer.py              # Agent 2
│   └── graph.py                  # LangGraph wiring + handoff loop
├── requirements.txt
├── .env.example
└── .gitignore
```

## 1. Prerequisites

- Python 3.10+
- A **remote Qdrant** cluster (Qdrant Cloud free tier works). Get the
  **cluster URL** and **API key** from the Qdrant Cloud web UI.
- An **OpenAI API key** (used for embeddings; also used as the default chat
  model).
- Optionally, an **Anthropic API key** if you set `LLM_PROVIDER=anthropic` to
  use Claude for the Researcher/Reviewer chat model instead of GPT.

## 2. Setup

```bash
git clone <this-repo-url>
cd <this-repo>

python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Now edit .env and fill in:
#   QDRANT_URL, QDRANT_API_KEY, OPENAI_API_KEY (and ANTHROPIC_API_KEY if used)
```

All secrets are read from environment variables via `python-dotenv` — nothing
is hardcoded, and `.env` is git-ignored.

## 3. Ingest the documentation corpus

This crawls a focused set of official LangChain and Qdrant documentation
pages, chunks them, embeds them, and upserts them into your remote Qdrant
collection (created automatically if it doesn't exist).

```bash
python ingest.py --recreate
```

Flags:
- `--recreate` — drop and recreate the collection first (use on a first run
  or to fully refresh the corpus).
- `--max-pages N` — cap the crawl for a quick smoke test, e.g.
  `python ingest.py --recreate --max-pages 20`.

Ingestion typically takes a few minutes depending on crawl depth and your
Qdrant plan's rate limits.

## 4. Run the app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (default `http://localhost:8501`). Ask a
question about LangChain or Qdrant. For each answer you'll see:

- The final grounded answer with inline citation markers `[1]`, `[2]`, ...
- An expandable **Cited sources** panel linking each marker to its source URL.
- The **Reviewer's verdict** (`APPROVED`, `REFUSED`, or `UNVERIFIED` with a
  reason), plus a note if the Reviewer sent the draft back for revision.

Try a question outside the ingested corpus (e.g. "What's the weather in
Cairo?") to see the Researcher refuse rather than fabricate an answer.

## How the handoff loop works

1. `researcher` node retrieves the top-`RETRIEVAL_K` passages for the query
   (or, on a revision pass, for the query *plus* the Reviewer's feedback —
   so the re-run can pull different context, not just re-answer from the same
   passages) and drafts a cited answer, or explicitly refuses.
2. `reviewer` node checks each claim against the retrieved passages and
   returns a structured verdict (`APPROVED` / `REJECTED`) plus feedback.
3. A conditional edge routes back to `researcher` if `REJECTED` **and** the
   loop budget (`MAX_REVIEW_LOOPS`, default 1) hasn't been used yet.
4. Otherwise it proceeds to `finalize`, which decides what the user actually
   sees:
   - Researcher refused → shown as a clean refusal.
   - Reviewer approved → shown as-is with sources.
   - Still rejected after the revision budget → shown with an explicit
     "⚠️ unverified" note rather than presented as confidently grounded.

## Configuration reference (`.env`)

| Variable | Required | Description |
|---|---|---|
| `QDRANT_URL` | yes | Remote Qdrant cluster URL |
| `QDRANT_API_KEY` | yes | Remote Qdrant API key |
| `QDRANT_COLLECTION` | no | Collection name (default `langchain_qdrant_docs`) |
| `OPENAI_API_KEY` | yes | Used for embeddings; default chat model provider |
| `LLM_PROVIDER` | no | `openai` (default) or `anthropic` |
| `ANTHROPIC_API_KEY` | only if `LLM_PROVIDER=anthropic` | Anthropic API key |
| `EMBEDDING_MODEL` | no | Default `text-embedding-3-small` |
| `CHAT_MODEL_OPENAI` | no | Default `gpt-4o-mini` |
| `CHAT_MODEL_ANTHROPIC` | no | Default `claude-sonnet-4-6` |
| `RETRIEVAL_K` | no | Passages retrieved per search (default `6`) |
| `MAX_REVIEW_LOOPS` | no | Max Reviewer→Researcher handoffs (default `1`) |

## Troubleshooting

- **"Missing required environment variables"** on app start → you haven't
  copied `.env.example` to `.env` or filled in a required key.
- **Empty / irrelevant retrieval results** → re-run `python ingest.py
  --recreate` to make sure the collection was actually populated (check the
  point count in the Qdrant Cloud web UI).
- **Qdrant collection dimension mismatch** → the collection was created with
  a different embedding size previously; run `python ingest.py --recreate`
  to rebuild it against the current `EMBEDDING_MODEL`.
