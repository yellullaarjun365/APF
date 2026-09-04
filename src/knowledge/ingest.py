"""APF -- knowledge base ingestion (V3 RAG, part 1).

Reads all .md files under data/knowledge/, splits them into paragraph-level
chunks, embeds them with a local sentence-transformers model, and stores
them in a persistent local ChromaDB collection under data/knowledge_db/.

Run this once after adding/changing any file in data/knowledge/:
    python -m src.knowledge.ingest
"""
import pathlib
import chromadb
from sentence_transformers import SentenceTransformer

KNOWLEDGE_DIR = pathlib.Path("data/knowledge")
DB_DIR = pathlib.Path("data/knowledge_db")
COLLECTION_NAME = "tilapia_aquaculture"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, local, no API key


def chunk_markdown(text: str, source: str) -> list[dict]:
    """Split on blank-line-separated paragraphs. Simple and predictable --
    good enough for a handful of curated source documents; revisit with a
    smarter splitter only if retrieval quality demands it."""
    chunks = []
    for para in text.split("\n\n"):
        para = para.strip()
        if len(para) > 30:  # skip stray headers/whitespace fragments
            chunks.append({"text": para, "source": source})
    return chunks


def main():
    DB_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading embedding model: {EMBED_MODEL_NAME} ...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    client = chromadb.PersistentClient(path=str(DB_DIR))
    # Fresh rebuild each run -- simplest correct behavior for a small,
    # infrequently-changing knowledge base.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    all_chunks = []
    for path in KNOWLEDGE_DIR.glob("*.md"):
        text = path.read_text(encoding="utf-8")
        all_chunks.extend(chunk_markdown(text, source=path.name))

    if not all_chunks:
        print(f"No .md files found in {KNOWLEDGE_DIR} -- nothing to ingest.")
        return

    print(f"Embedding {len(all_chunks)} chunks from {len(list(KNOWLEDGE_DIR.glob('*.md')))} file(s) ...")
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    collection.add(
        ids=[f"chunk_{i}" for i in range(len(all_chunks))],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"source": c["source"]} for c in all_chunks],
    )
    print(f"Ingested {len(all_chunks)} chunks into '{COLLECTION_NAME}' at {DB_DIR}")


if __name__ == "__main__":
    main()
