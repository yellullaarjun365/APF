"""Ingest farmer extraction logs into ChromaDB."""
import json
import pathlib

from sentence_transformers import SentenceTransformer
import chromadb


DB_DIR = pathlib.Path("data/knowledge_db")
COLLECTION = "farmer_experience"
MODEL_NAME = "all-MiniLM-L6-v2"


def ingest_logs():
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(
        path=str(DB_DIR)
    )

    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass

    coll = client.create_collection(COLLECTION)

    log_file = pathlib.Path(
        "data/logs/extractions.jsonl"
    )

    if not log_file.exists():
        print("No extractions log found.")
        return

    chunks = []
    metas = []

    with open(
        log_file,
        encoding="utf-8"
    ) as f:

        for line in f:
            try:
                rec = json.loads(line)

                ext = rec.get(
                    "extracted",
                    {}
                )

                pred = rec.get(
                    "prediction",
                    {}
                )

                text = (
                    f"Farmer pond: "
                    f"{ext.get('pond_area_ha', '?')} ha, "
                    f"{ext.get('stocking_count', '?')} fish, "
                    f"{ext.get('culture_days', '?')} days, "
                    f"temp {ext.get('mean_temperature_c', '?')}C, "
                    f"DO {ext.get('mean_do_mg_l', '?')} mg/L, "
                    f"pH {ext.get('mean_ph', '?')}. "
                    f"Forecast: "
                    f"{pred.get('point_estimate_kg', '?')} kg."
                )

                chunks.append(text)

                metas.append({
                    "source": "farmer_log",
                    "raw": json.dumps(rec)
                })

            except Exception:
                continue

    if not chunks:
        print("No valid entries.")
        return

    embeddings = model.encode(
        chunks
    ).tolist()

    coll.add(
        ids=[
            f"log_{i}"
            for i in range(len(chunks))
        ],
        embeddings=embeddings,
        documents=chunks,
        metadatas=metas
    )

    print(
        f"Ingested {len(chunks)} farmer log entries."
    )


if __name__ == "__main__":
    ingest_logs()
