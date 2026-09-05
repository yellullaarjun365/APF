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
last few turns, or whenever the farmer's own words say they just want text
-- UNLESS the farmer explicitly asks to see images now, which always wins.

Web fallback (V1 fix): when the local knowledge base has nothing
(`retrieve()` returns zero chunks), OR the LLM's own answer indicates it
had nothing new/refuses to go beyond the reference material, fall back to
a free Wikipedia lookup (src/knowledge/web_fallback.py) instead of just
returning that as final. Local knowledge is always tried first and stays
authoritative -- this only fires when the local answer is genuinely a
dead end.

Refusal-detection fix (V1): the small local model sometimes refuses a
"tell me more" follow-up outright ("I can't fulfill this request... not
allowed to add outside information") rather than saying plainly it has
nothing new -- same underlying situation, different wording. The prompt
itself has also been softened to discourage that refusal framing in the
first place, since pattern-matching every possible refusal phrasing is a
losing game long-term.
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


# An explicit ask to see images should always win over the "recently
# shown" suppression below -- that heuristic exists to avoid pestering
# with repeat photos on unrelated follow-ups, not to refuse a direct
# request. Checked against the RAW farmer message, not the standalone-
# rewritten one, since the rewrite step is free to drop phrasing it
# doesn't consider essential to the informational content of the question.
_IMAGE_REQUEST_MARKERS = (
    "show me", "show the image", "show image", "show pic", "show the pic",
    "image of", "images of", "picture of", "pictures of", "photo of",
    "photos of", "see it", "see them", "let me see",
)


_IMAGE_NOUNS = ("image", "images", "picture", "pictures", "photo", "photos", "pic", "pics")


def _wants_images_explicitly(raw_question: str) -> bool:
    q = raw_question.lower()
    if any(p in q for p in _IMAGE_REQUEST_MARKERS):
        return True
    # BUG FIX: the marker-phrase list above only matched specific
    # prepositional patterns ("images of", "show me", "photo of"). An
    # equally natural phrasing like "provide me its images" or "give me
    # the photos" doesn't use any of those exact prepositions and was
    # silently missed -- force stayed False, so the "recently shown"
    # suppression in _get_images_for correctly (by its own logic) hid
    # images for a species already mentioned in the last few turns, even
    # though the farmer had just explicitly asked to see them. If the
    # message is short and contains an image-noun at all, that alone is
    # a reliable enough signal -- a farmer isn't likely to casually
    # mention "images"/"photos" in an unrelated short message.
    words = q.split()
    if len(words) <= 6 and any(w.strip(".,!?'\"") in _IMAGE_NOUNS for w in words):
        return True
    return False


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


async def _get_images_for(question: str, history: list, force: bool = False) -> list[dict]:
    """force=True (an explicit "show me images" ask) always fetches,
    bypassing the recently-shown suppression -- see _wants_images_explicitly."""
    species_name = await extract_species_name(question)
    if not species_name:
        return []
    if not force and _species_recently_shown(species_name, history):
        return []
    return await asyncio.to_thread(get_species_images, species_name)


async def _web_fallback_answer(standalone_question: str, text_only: bool, history: list, force_images: bool = False) -> dict:
    """Try free Wikipedia lookup before giving up entirely. Runs the
    Wikipedia HTTP call and the (optional) species-image lookup
    concurrently -- they're independent, no reason to serialize them."""
    images_task = (
        asyncio.sleep(0, result=[]) if text_only
        else _get_images_for(standalone_question, history, force=force_images)
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


# Phrases the LLM itself uses when local chunks were found but had nothing
# NEW to add (e.g. answering a "tell me more" follow-up), OR when it
# outright refuses to answer beyond the reference material. Both are the
# same underlying situation from the farmer's point of view -- "I didn't
# get new information" -- so both trigger the web fallback rather than
# being returned as a final answer. This list is inherently a losing
# game long-term (infinite possible refusal phrasings), which is why the
# prompt itself has also been softened below to discourage refusal
# framing in the first place -- this list is a safety net, not the
# primary fix.
_NO_NEW_INFO_MARKERS = (
    "couldn't find any additional", "could not find any additional",
    "no additional detail", "don't have additional", "do not have additional",
    "nothing further", "nothing more to add", "no further detail",
    "not covered in", "not mentioned in the reference",
    "can't fulfill", "cannot fulfill", "unable to fulfill",
    "not allowed to", "i'm not able to", "i am not able to",
    "unable to provide", "cannot provide", "can't provide",
    "i don't have enough information", "i do not have enough information",
    "sorry, i can't", "sorry, i cannot",
)


def _indicates_no_new_info(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in _NO_NEW_INFO_MARKERS)


async def answer_knowledge_question(question: str, history: list = None) -> dict:
    history = history or []
    text_only = _wants_text_only(question)
    force_images = _wants_images_explicitly(question)
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
        return await _web_fallback_answer(standalone_question, text_only, history, force_images=force_images)

    context = "\n\n".join(f"[{c['source']}] {c['text']}" for c in chunks)
    sources = sorted(set(c["source"] for c in chunks))

    already_told = (
        f"\nYou already told the farmer this in your previous reply -- do NOT "
        f"repeat it verbatim. Add NEW details from the reference material "
        f"instead, or say plainly and briefly if there is nothing further to "
        f"add -- never refuse to answer, just say what's missing:\n"
        f"\"{prev_answer}\"\n"
        if (prev_answer and _looks_like_followup(question)) else ""
    )

    # Softened from a hard "Do not add outside info" instruction -- that
    # framing was making the local model outright refuse follow-ups
    # ("I can't fulfill this request... not allowed to add outside
    # information") instead of just saying plainly it has nothing more.
    # The goal is grounding (no hallucinated facts), not a refusal
    # posture -- asking for honesty gets that without the refusal.
    prompt = f"""Answer the question using the reference material below as your source of facts. Do not invent facts that aren't in it -- but never refuse to respond. If the reference material only partially answers the question, share what it does say, then briefly and plainly note what it doesn't cover.

Reference:
{context}
{already_told}
Question: {standalone_question}

Write 2-4 clear, friendly sentences. Plain text only."""

    try:
        text, images = await asyncio.gather(
            ollama_generate(prompt, temperature=0.2),
            (asyncio.sleep(0, result=[]) if text_only else _get_images_for(standalone_question, history, force=force_images)),
        )
    except Exception as e:
        print(f"[rag_answer] Ollama failed: {e}")
        return {
            "answer": "I found relevant info but couldn't generate a response. Please try again.",
            "sources": sources,
            "images": [],
        }

    if text:
        if _indicates_no_new_info(text):
            # Local chunks existed but had nothing NEW, or the model
            # refused outright -- try Wikipedia before accepting that as
            # final. If Wikipedia also comes up empty, fall back to the
            # LLM's own answer rather than the generic NO_MATCH_REPLY,
            # since unlike a hard retrieval miss, we do have real local
            # sources to cite here.
            web_result = await _web_fallback_answer(standalone_question, text_only, history, force_images=force_images)
            if web_result["sources"]:
                web_result["sources"] = sources + web_result["sources"]
                return web_result
            return {"answer": text, "sources": sources, "images": images}
        return {"answer": text, "sources": sources, "images": images}

    return {
        "answer": "I found relevant info but couldn't generate a response. Please try again.",
        "sources": sources,
        "images": images,
    }
