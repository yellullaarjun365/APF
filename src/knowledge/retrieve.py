"""APF -- knowledge base retrieval (V3 RAG, part 2).

Loads the persistent ChromaDB collection built by ingest.py and exposes a
single function: retrieve top-k relevant chunks for a farmer's question.
"""
import pathlib
import chromadb
from sentence_transformers import SentenceTransformer

DB_DIR = pathlib.Path("data/knowledge_db")
COLLECTION_NAME = "tilapia_aquaculture"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

_model = None
_collection = None


def _load():
    global _model, _collection
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    if _collection is None:
        client = chromadb.PersistentClient(path=str(DB_DIR))
        _collection = client.get_collection(COLLECTION_NAME)


def retrieve(query: str, k: int = 4) -> list[dict]:
    """Returns up to k chunks: [{"text": ..., "source": ...}, ...],
    ranked by relevance. Returns [] if the DB is missing or empty --
    callers must handle that (no source material found) explicitly rather
    than silently proceeding as if retrieval succeeded."""
    try:
        _load()
        query_embedding = _model.encode([query]).tolist()
        results = _collection.query(query_embeddings=query_embedding, n_results=k)
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        return [{"text": d, "source": m.get("source", "unknown")} for d, m in zip(docs, metas)]
    except Exception as e:
        print(f"[knowledge] retrieval failed: {e}")
        return []


if __name__ == "__main__":
    # Quick manual test: python -m src.knowledge.retrieve
    for r in retrieve("what temperature does tilapia prefer"):
        print(f"[{r['source']}] {r['text'][:100]}...")
