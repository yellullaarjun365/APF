"""Query both tilapia docs and farmer experience collections."""
import pathlib
import chromadb
from sentence_transformers import SentenceTransformer


DB_DIR = pathlib.Path("data/knowledge_db")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

_model = None
_client = None


def _load():
    global _model, _client

    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)

    if _client is None:
        _client = chromadb.PersistentClient(
            path=str(DB_DIR)
        )


def _query_collection(
    name: str,
    query_embedding: list,
    k: int
):
    try:
        coll = _client.get_collection(name)

        results = coll.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        return [
            {
                "text": d,
                "source": m.get("source", name)
            }
            for d, m in zip(docs, metas)
        ]

    except Exception as e:
        print(
            f"[knowledge] collection '{name}' failed: {e}"
        )
        return []


def retrieve(
    query: str,
    k: int = 4
) -> list:

    try:
        _load()

        emb = _model.encode(
            [query]
        ).tolist()[0]

        docs = _query_collection(
            "tilapia_aquaculture",
            emb,
            k=k
        )

        exp = _query_collection(
            "farmer_experience",
            emb,
            k=min(2, k)
        )

        seen = set()
        merged = []

        for c in docs + exp:
            if c["text"] not in seen:
                seen.add(c["text"])
                merged.append(c)

        return merged[:k]

    except Exception as e:
        print(
            f"[knowledge] retrieval failed: {e}"
        )
        return []
