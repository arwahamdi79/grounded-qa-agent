"""
Ingests official LangChain and Qdrant documentation into the remote Qdrant
collection defined in the environment.

Usage:
    python ingest.py                # ingest default doc set
    python ingest.py --recreate     # drop & recreate the collection first
    python ingest.py --max-pages 40 # cap total pages fetched (useful for a quick test run)

This uses LangChain's RecursiveUrlLoader to crawl a small set of seed pages
for each documentation site, strips HTML down to text, chunks it, embeds it
with OpenAI embeddings, and upserts into Qdrant with source metadata
(source URL, title) so the Researcher agent can cite it later.
"""
import argparse
import sys

from bs4 import BeautifulSoup
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

import config

# Seed pages to crawl. Kept shallow (max_depth) and domain-scoped so the crawl
# stays focused on real documentation rather than the whole internet.
SEED_SOURCES = [
    {
        "url": "https://python.langchain.com/docs/introduction/",
        "max_depth": 2,
    },
    {
        "url": "https://python.langchain.com/docs/concepts/",
        "max_depth": 2,
    },
    {
        "url": "https://langchain-ai.github.io/langgraph/",
        "max_depth": 2,
    },
    {
        "url": "https://qdrant.tech/documentation/",
        "max_depth": 2,
    },
    {
        "url": "https://qdrant.tech/documentation/concepts/",
        "max_depth": 2,
    },
]


def extractor(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def load_documents(max_pages: int | None):
    all_docs = []
    for source in SEED_SOURCES:
        print(f"Crawling {source['url']} (max_depth={source['max_depth']}) ...")
        loader = RecursiveUrlLoader(
            url=source["url"],
            max_depth=source["max_depth"],
            extractor=extractor,
            prevent_outside=True,
            timeout=15,
        )
        try:
            docs = loader.load()
        except Exception as e:
            print(f"  WARNING: failed to crawl {source['url']}: {e}", file=sys.stderr)
            continue
        print(f"  -> fetched {len(docs)} pages")
        all_docs.extend(docs)
        if max_pages and len(all_docs) >= max_pages:
            all_docs = all_docs[:max_pages]
            break
    return all_docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate the Qdrant collection")
    parser.add_argument("--max-pages", type=int, default=None, help="Cap number of pages fetched (for quick test runs)")
    args = parser.parse_args()

    config.validate_config()

    print(f"Loading documents (max_pages={args.max_pages}) ...")
    raw_docs = load_documents(args.max_pages)
    if not raw_docs:
        print("No documents were fetched. Aborting.", file=sys.stderr)
        sys.exit(1)

    print(f"Fetched {len(raw_docs)} raw pages. Splitting into chunks ...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    chunks = splitter.split_documents(raw_docs)

    # Normalize metadata: keep only source + title, drop huge/irrelevant fields
    for chunk in chunks:
        src = chunk.metadata.get("source", "unknown")
        title = chunk.metadata.get("title", src)
        chunk.metadata = {"source": src, "title": title}

    print(f"Produced {len(chunks)} chunks. Connecting to Qdrant at {config.QDRANT_URL} ...")
    client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)

    embeddings = HuggingFaceEmbeddings(
    model_name=config.EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
    )

    if args.recreate and client.collection_exists(config.QDRANT_COLLECTION):
        print(f"Dropping existing collection '{config.QDRANT_COLLECTION}' ...")
        client.delete_collection(config.QDRANT_COLLECTION)

    if not client.collection_exists(config.QDRANT_COLLECTION):
        # text-embedding-3-small = 1536 dims
        print(f"Creating collection '{config.QDRANT_COLLECTION}' ...")
        client.create_collection(
            collection_name=config.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=config.QDRANT_COLLECTION,
        embedding=embeddings,
    )

    print(f"Upserting {len(chunks)} chunks into Qdrant ...")
    batch_size = 64
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vectorstore.add_documents(batch)
        print(f"  upserted {min(i + batch_size, len(chunks))}/{len(chunks)}")

    print("Ingestion complete.")


if __name__ == "__main__":
    main()
