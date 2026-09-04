"""APF -- RAG answer generation (V3, part 3).

Takes a farmer's question -- about their own aquaculture operation, tilapia
biology, or general marine-life topics -- retrieves relevant chunks, and
asks Ollama to answer ONLY from those chunks -- same "translate, don't
invent" boundary already enforced for the explanation and follow-up
layers. If retrieval finds nothing relevant, says so honestly instead of
letting Ollama free-associate.

Also detects if the question is about a specific animal/marine species and,
if so, attaches a few real reference images (Wikimedia Commons) alongside
the text answer.
"""
import os
import requests
from knowledge.retrieve import retrieve
from knowledge.species_images import extract_species_name, get_species_images

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT_S = float(os.environ.get("OLLAMA_TIMEOUT_S", "120"))

NO_MATCH_REPLY = (
    "I don't have grounded information on that in my current knowledge base "
    "(right now it covers Nile tilapia biology and farming, general marine "
    "life, whales and marine mammals, and global seafood/regional patterns). "
    "I don't want to guess on something this specific -- feel free to "
    "rephrase, or ask about a species, habitat, or aquaculture practice."
)


def answer_knowledge_question(question: str) -> dict:
    """Returns {"answer": str, "sources": [str, ...], "images": [dict, ...]}.
    Sources list is empty when no relevant chunks were found (answer is the
    honest no-match message in that case, not a hallucinated one). Images
    list is empty whenever the question isn't about a specific species or
    the lookup fails -- never assume it's populated."""
    chunks = retrieve(question, k=4)

    # Species image lookup runs independently of retrieval success --
    # a question can be about a real species even if our knowledge base
    # doesn't have a matching chunk for it. NOTE: extract_species_name is
    # tuned to only fire when the question is actually ASKING ABOUT the
    # animal (e.g. "tell me about X", "what is X") -- not merely mentioning
    # a species name inside a practice/management question (e.g. "why does
    # pH matter for tilapia" should NOT trigger images). See species_images.py.
    species_name = extract_species_name(question)
    images = get_species_images(species_name) if species_name else []

    if not chunks:
        return {"answer": NO_MATCH_REPLY, "sources": [], "images": images}

    context = "\n\n".join(f"[{c['source']}] {c['text']}" for c in chunks)
    sources = sorted(set(c["source"] for c in chunks))

    # Topic-agnostic framing -- do NOT hardcode a domain (e.g. "farmer" /
    # "tilapia") here. The retrieved context itself determines what this
    # question is actually about; the prompt should work equally well for
    # a pond-management question and a "tell me about blue whales" question.
    prompt = f"""Answer the question below using ONLY the reference material provided. Do not add information not present in it. If the material doesn't fully answer the question, say what it does cover and note the gap honestly -- do not fill gaps with invented facts. Do not assume the question is about farming or aquaculture unless the reference material itself is about that.

Reference material:
{context}

Question: {question}

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
            return {"answer": text, "sources": sources, "images": images}
    except Exception as e:
        print(f"[rag_answer] Ollama call failed: {e}")

    return {
        "answer": "I found relevant information but couldn't generate a response right now -- please try again.",
        "sources": sources,
        "images": images,
    }


if __name__ == "__main__":
    result = answer_knowledge_question("tell me about blue whales")
    print(result["answer"])
    print("Sources:", result["sources"])
    print("Images:", [img["url"] for img in result["images"]])
