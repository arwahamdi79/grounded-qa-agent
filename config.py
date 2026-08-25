"""
Central configuration. All secrets/config come from environment variables
(loaded from a local .env file via python-dotenv when running locally).
Never hardcode API keys here.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- Qdrant (remote cluster) ---
QDRANT_URL = os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "langchain_qdrant_docs")

# --- LLM / Embeddings provider ---
# Supported: "openai" (default) or "anthropic" for the chat model.
# Embeddings currently use OpenAI embeddings (required even if chat model is Anthropic),
# since Qdrant needs a fixed embedding space for the ingested corpus.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai").lower()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHAT_MODEL_OPENAI = os.environ.get("CHAT_MODEL_OPENAI", "gpt-4o-mini")
CHAT_MODEL_ANTHROPIC = os.environ.get("CHAT_MODEL_ANTHROPIC", "claude-sonnet-4-6")

# --- Retrieval / agent behavior ---
RETRIEVAL_K = int(os.environ.get("RETRIEVAL_K", "6"))
MAX_REVIEW_LOOPS = int(os.environ.get("MAX_REVIEW_LOOPS", "1"))  # reviewer sends back at most once

REQUIRED_VARS = ["QDRANT_URL", "QDRANT_API_KEY"]


def validate_config():
    missing = [v for v in REQUIRED_VARS if not os.environ.get(v)]
    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if LLM_PROVIDER == "anthropic" and not ANTHROPIC_API_KEY:
        missing.append("ANTHROPIC_API_KEY")
    # embeddings always need an OpenAI key in this implementation
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY (required for embeddings)")
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}. "
            f"Copy .env.example to .env and fill in the values."
        )
