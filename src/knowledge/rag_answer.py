"""Async RAG answer with caching. Runs species-image lookup concurrently
with the main answer generation (they're independent -- no reason to wait
for one before starting the other), roughly halving the LLM-bound portion
of latency for a typical knowledge question."""
import asyncio

from knowledge.retrieve import retrieve
from knowledge.species_images import extract_species_name, get_species_images
from llm.async_client import ollama_generate


NO_MATCH_REPLY = (
    "I do not have grounded information on that in my current knowledge base. "
    "Feel free to rephrase, or ask about a species, habitat, or aquaculture practice."
)


async def _get_images_for(question: str) -> list[dict]:
    """Species classification (LLM, cached) + image fetch (HTTP), bundled
    so callers can run this concurrently with answer generation via
    asyncio.gather. get_species_images is sync/fast (single HTTP GET) so
    it's dispatched via to_thread rather than blocking the event loop."""
    species_name = await extract_species_name(question)
    if not species_name:
        return []
    return await asyncio.to_thread(get_species_images, species_name)


async def answer_knowledge_question(question: str) -> dict:
    chunks = retrieve(question, k=4)

    if not chunks:
        # Still worth checking for images even with no text match (e.g.
        # narwhals: honest "don't know" text + real reference photos).
        images = await _get_images_for(question)
        return {"answer": NO_MATCH_REPLY, "sources": [], "images": images}

    context = "\n\n".join(f"[{c['source']}] {c['text']}" for c in chunks)
    sources = sorted(set(c["source"] for c in chunks))

    prompt = f"""Answer using ONLY the reference material below. Do not add outside info.

Reference:
{context}

Question: {question}

Write 2-4 clear, friendly sentences. Plain text only."""

    # Run the main answer generation and the species-image lookup
    # concurrently -- they're independent LLM/network calls, no reason to
    # serialize them. This is the main latency win: previously these ran
    # one after another (and species lookup even ran TWICE, once here and
    # once again redundantly in main.py's /chat handler -- that duplicate
    # call has been removed; main.py now uses this function's images
    # directly instead of recomputing them).
    try:
        text, images = await asyncio.gather(
            ollama_generate(prompt, temperature=0.2),
            _get_images_for(question),
        )
    except Exception as e:
        print(f"[rag_answer] Ollama failed: {e}")
        return {
            "answer": "I found relevant info but couldn't generate a response. Please try again.",
            "sources": sources,
            "images": [],
        }

    if text:
        return {"answer": text, "sources": sources, "images": images}

    return {
        "answer": "I found relevant info but couldn't generate a response. Please try again.",
        "sources": sources,
        "images": images,
    }
