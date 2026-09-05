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

Web fallback (V1 fix): when the local knowledge base has nothing
(`retrieve()` returns zero chunks), fall back to a free Wikipedia lookup
(src/knowledge/web_fallback.py) instead of just saying "I don't know".
Local knowledge is always tried first and stays authoritative -- this
only fires on a genuine local miss.
"""
import asyncio

from knowledge.retrieve import retrieve
from knowledge.species_images import extract_species_name, get_species_images
from knowledge.web_fallback import fetch_wikipedia_summary
from llm.async_client import ollama_generate


NO_MATCH_REPLY = (
    "I do not have grounded information on that in my current knowledge base, "
    "and couldn't find anything on Wikipedia either. Feel free to rephrase, or "
    "ask about a species, habitat, or aquaculture practice."
)

_FOLLOWUP_MARKERS = (
    "more", "again", "it", "that", "this", "those", "these", "mean",
    "also", "further", "instead",
)


def _looks_like_followup(text: str) -> bool:
    words = text.strip().lower().split()
    if not words:
        return False
    if len(words) <= 4:
        return True
    return any(w.strip(".,!?") in _FOLLOWUP_MARKERS for w in words)

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
    if not species_name:
        return False
    name_lower = species_name.lower()
    for h in (history or [])[-6:]:
        if h.get("role") == "assistant" and name_lower in h.get("content", "").lower():
            return True
    return False


async def _standalone_query(question: str, history: list) -> str:
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
    species_name = await extract_species_name(question)
    if not species_name:
        return []
    if _species_recently_shown(species_name, history):
        return []
    return await asyncio.to_thread(get_species_images, species_name)


async def _web_fallback_answer(standalone_question: str, text_only: bool, history: list) -> dict:
    """Local knowledge base had nothing -- try free Wikipedia lookup before
    giving up entirely. Runs the Wikipedia HTTP call and the (optional)
    species-image lookup concurrently, same reasoning as the main path:
    they're independent, no reason to serialize them."""
    images_task = (
        asyncio.sleep(0, result=[]) if text_only
        else _get_images_for(standalone_question, history)
    )
    web, images = await asyncio.gather(
        asyncio.to_thread(fetch_wikipedia_summary, standalone_question),
        images_task,
    )
    if web:
        source = f"Wikipedia: {web['title']}"
        if web.get("url"):
            source += f" ({web['url']})"
        return {"answer": web["extract"], "sources": [source], "images": images}
    return {"answer": NO_MATCH_REPLY, "sources": [], "images": images}


async def answer_knowledge_question(question: str, history: list = None) -> dict:
    history = history or []
    text_only = _wants_text_only(question)
    standalone_question = (
        await _standalone_query(question, history)
        if _looks_like_followup(question) else question
    )
    prev_answer = _last_assistant_reply(history)

    chunks = retrieve(standalone_question, k=4)

    if not chunks:
        # Local knowledge base has nothing -- try the free Wikipedia
        # fallback before giving up. Still worth checking for images even
        # on a text miss (e.g. narwhals: honest "don't know" text + real
        # reference photos) -- unless the farmer explicitly said they just
        # want text.
        return await _web_fallback_answer(standalone_question, text_only, history)

    context = "\n\n".join(f"[{c['source']}] {c['text']}" for c in chunks)
    sources = sorted(set(c["source"] for c in chunks))

    already_told = (
        f"\nYou already told the farmer this in your previous reply -- do NOT "
        f"repeat it verbatim. Add NEW details from the reference material "
        f"instead, or say plainly if there is nothing further to add:\n"
        f"\"{prev_answer}\"\n"
        if (prev_answer and _looks_like_followup(question)) else ""
    )

    prompt = f"""Answer using ONLY the reference material below. Do not add outside info.

Reference:
{context}
{already_told}
Question: {standalone_question}

Write 2-4 clear, friendly sentences. Plain text only."""

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
