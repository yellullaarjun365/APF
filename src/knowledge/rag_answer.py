"""Async RAG answer with caching."""
from knowledge.retrieve import retrieve
from knowledge.species_images import extract_species_name, get_species_images
from llm.async_client import ollama_generate


NO_MATCH_REPLY = (
    "I do not have grounded information on that in my current knowledge base. "
    "Feel free to rephrase, or ask about a species, habitat, or aquaculture practice."
)


async def answer_knowledge_question(question: str) -> dict:
    chunks = retrieve(question, k=4)

    species_name = extract_species_name(question)
    images = (
        get_species_images(species_name)
        if species_name
        else []
    )

    if not chunks:
        return {
            "answer": NO_MATCH_REPLY,
            "sources": [],
            "images": images
        }

    context = "\n\n".join(
        f"[{c['source']}] {c['text']}"
        for c in chunks
    )

    sources = sorted(
        set(c["source"] for c in chunks)
    )

    prompt = f"""Answer using ONLY the reference material below. Do not add outside info.

Reference:
{context}

Question: {question}

Write 2-4 clear, friendly sentences. Plain text only."""

    try:
        text = await ollama_generate(
            prompt,
            temperature=0.2
        )

        if text:
            return {
                "answer": text,
                "sources": sources,
                "images": images
            }

    except Exception as e:
        print(
            f"[rag_answer] Ollama failed: {e}"
        )

    return {
        "answer": (
            "I found relevant info but couldn't generate a response. "
            "Please try again."
        ),
        "sources": sources,
        "images": images
    }
