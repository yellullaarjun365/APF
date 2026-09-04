"""APF -- RAG answer generation (V3, part 3).

Takes a farmer's aquaculture question, retrieves relevant chunks, and asks
Ollama to answer ONLY from those chunks -- same "translate, don't invent"
boundary already enforced for the explanation and follow-up layers. If
retrieval finds nothing relevant, says so honestly instead of letting
Ollama free-associate.
"""
import os
import requests
from src.knowledge.retrieve import retrieve

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT_S = float(os.environ.get("OLLAMA_TIMEOUT_S", "120"))

NO_MATCH_REPLY = (
    "I don't have grounded information on that in my current knowledge base "
    "(right now it covers core Nile tilapia biology and habitat). I don't "
    "want to guess on something this specific -- feel free to rephrase, or "
    "ask about temperature tolerance, feeding, reproduction, or general "
    "tilapia farming history."
)


def answer_knowledge_question(question: str) -> dict:
    """Returns {"answer": str, "sources": [str, ...]}. Sources list is
    empty when no relevant chunks were found (answer is the honest
    no-match message in that case, not a hallucinated one)."""
    chunks = retrieve(question, k=4)
    if not chunks:
        return {"answer": NO_MATCH_REPLY, "sources": []}

    context = "\n\n".join(f"[{c['source']}] {c['text']}" for c in chunks)
    sources = sorted(set(c["source"] for c in chunks))

    prompt = f"""You are answering a farmer's question about Nile tilapia and aquaculture, using ONLY the reference material below. Do not add information not present in it. If the material doesn't fully answer the question, say what it does cover and note the gap honestly -- do not fill gaps with invented facts.

Reference material:
{context}

Farmer's question: {question}

Write a clear, friendly 2-4 sentence answer using only the material above. Plain sentences, no markdown."""

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "keep_alive": "30m"},
            timeout=OLLAMA_TIMEOUT_S,
        )
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
        if text:
            return {"answer": text, "sources": sources}
    except Exception as e:
        print(f"[rag_answer] Ollama call failed: {e}")

    return {
        "answer": "I found relevant information but couldn't generate a response right now -- please try again.",
        "sources": sources,
    }


if __name__ == "__main__":
    result = answer_knowledge_question("What temperature does tilapia prefer?")
    print(result["answer"])
    print("Sources:", result["sources"])
