"""Async RAG answer with caching. Runs species-image lookup concurrently
with the main answer generation (they're independent -- no reason to wait
for one before starting the other), roughly halving the LLM-bound portion
of latency for a typical knowledge question.

Also history-aware (V1 fix): a bare follow-up like "tell me more" or
"i mean the text information" carries no retrievable content on its own --
it only makes sense next to the conversation it's replying to. Everything
here now optionally takes `history` and uses it to (a) rewrite the message
into a standalone question before retrieval/species-detection, (b) tell the
model what it already said so it adds NEW detail instead of repeating
itself, and (c) skip re-showing images for a species already shown in the
last few turns, or whenever the farmer's own words say they just want text.
"""
import asyncio

from knowledge.retrieve import retrieve
from knowledge.species_images import extract_species_name, get_species_images
from llm.async_client import ollama_generate


NO_MATCH_REPLY = (
    "I do not have grounded information on that in my current knowledge base. "
    "Feel free to rephrase, or ask about a species, habitat, or aquaculture practice."
)

# Cheap keyword check -- no LLM round-trip needed for this one. Catches the
# exact complaint that prompted this fix: "i mean the text information".
_TEXT_ONLY_PHRASES = (
    "just text", "text only", "text information", "the text information",
    "no image", "no images", "without image", "without images",
    "no picture", "no pictures", "skip the image", "skip images",
)


def _wants_text_only(raw_question: str) -> bool:
    q = raw_question.lower()
    return any(p in q for p in _TEXT_ONLY_PHRASES)


def _last_assistant_reply(history: list) -> str:
    for h in reversed(history or []):
        if h.get("role") == "assistant":
            return h.get("content", "")
    return ""


def _species_recently_shown(species_name: str, history: list) -> bool:
    """True if this species was already discussed in the last few assistant
    turns -- used to avoid re-fetching/re-displaying the same photos every
    time the farmer says "tell me more" about the same animal."""
    if not species_name:
        return False
    name_lower = species_name.lower()
    for h in (history or [])[-6:]:
        if h.get("role") == "assistant" and name_lower in h.get("content", "").lower():
            return True
    return False


async def _standalone_query(question: str, history: list) -> str:
    """Rewrite a possibly-fragmentary follow-up ('tell me more', 'i mean
    the text information') into a standalone question, using recent
    conversation history, so retrieval and species-detection have
    something real to work with. Falls back to the original question
    unchanged on empty history or any failure -- this step should never
    make an already-fine query worse."""
    if not history:
        return question
    history_lines = "\n".join(
        f"{h.get('role', '?')}: {h.get('content', '')}" for h in history[-6:]
    )
    prompt = (
        "Conversation so far:\n" + history_lines +
        "\n\nThe farmer's latest message is: \"" + question + "\"\n\n"
        "Rewrite ONLY the latest message as a standalone, specific question "
        "a knowledge assistant could answer with no other context. If it's "
        "a request for MORE/deeper detail on the same topic just discussed, "
        "say so explicitly (e.g. \"give more detail about X beyond what was "
        "already covered\"). If the latest message is already standalone, "
        "return it unchanged. Reply with ONLY the rewritten question, "
        "nothing else, no quotes."
    )
    try:
        rewritten = await ollama_generate(prompt, temperature=0.0)
        rewritten = rewritten.strip().strip('"').strip("'")
        return rewritten or question
    except Exception as e:
        print(f"[rag_answer] standalone-query rewrite failed, using original: {e}")
        return question


async def _get_images_for(question: str, history: list) -> list[dict]:
    """Species classification (LLM, cached) + image fetch (HTTP), bundled
    so callers can run this concurrently with answer generation via
    asyncio.gather. get_species_images is sync/fast (single HTTP GET) so
    it's dispatched via to_thread rather than blocking the event loop.
    Skips re-fetching if the same species was already shown recently."""
    species_name = await extract_species_name(question)
    if not species_name:
        return []
    if _species_recently_shown(species_name, history):
        return []
    return await asyncio.to_thread(get_species_images, species_name)


async def answer_knowledge_question(question: str, history: list = None) -> dict:
    history = history or []
    text_only = _wants_text_only(question)
    standalone_question = await _standalone_query(question, history)
    prev_answer = _last_assistant_reply(history)

    chunks = retrieve(standalone_question, k=4)

    if not chunks:
        # Still worth checking for images even with no text match (e.g.
        # narwhals: honest "don't know" text + real reference photos) --
        # unless the farmer explicitly said they just want text.
        images = [] if text_only else await _get_images_for(standalone_question, history)
        return {"answer": NO_MATCH_REPLY, "sources": [], "images": images}

    context = "\n\n".join(f"[{c['source']}] {c['text']}" for c in chunks)
    sources = sorted(set(c["source"] for c in chunks))

    already_told = (
        f"\nYou already told the farmer this in your previous reply -- do NOT "
        f"repeat it verbatim. Add NEW details from the reference material "
        f"instead, or say plainly if there is nothing further to add:\n"
        f"\"{prev_answer}\"\n"
        if prev_answer else ""
    )

    prompt = f"""Answer using ONLY the reference material below. Do not add outside info.

Reference:
{context}
{already_told}
Question: {standalone_question}

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
            (asyncio.sleep(0, result=[]) if text_only else _get_images_for(standalone_question, history)),
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
