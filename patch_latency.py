"""
Latency fix patch. Run from repo root: python patch_latency.py

Does three things to src/api/main.py:
1. Adds `from llm.async_client import ollama_chat` import
2. Replaces sync _classify_intent (blocking requests.post) with an async
   version using the shared cached ollama_chat client
3. Removes the redundant duplicate species-image lookup in the /chat
   knowledge_question branch -- rag_result["images"] (computed once,
   concurrently with the answer, inside answer_knowledge_question) is
   used directly instead of calling extract_species_name a second time
"""
import re
from pathlib import Path

MAIN_PY = Path("src/api/main.py")
content = MAIN_PY.read_text(encoding="utf-8")

if "ollama_chat" in content and "async def _classify_intent" in content:
    print("Already patched. Nothing to do.")
    raise SystemExit

backup = MAIN_PY.with_suffix(".py.bak4")
backup.write_text(content, encoding="utf-8")
print(f"Backup written to {backup}")

# 1. Add ollama_chat import
old_import = "from llm.async_client import OLLAMA_URL, OLLAMA_MODEL, TIMEOUT_S"
new_import = "from llm.async_client import OLLAMA_URL, OLLAMA_MODEL, TIMEOUT_S, ollama_chat"
if old_import not in content:
    print("ERROR: import line not found as expected.")
    raise SystemExit
content = content.replace(old_import, new_import, 1)

# 2. Replace _classify_intent with async version
pattern = re.compile(
    r"def _classify_intent\(text: str\) -> str:.*?(?=\n(?:def |FIELD_LABELS_AND_WHY|@app\.))",
    re.DOTALL,
)
match = pattern.search(content)
if not match:
    print("ERROR: could not locate _classify_intent boundaries.")
    raise SystemExit

new_func = '''async def _classify_intent(text: str) -> str:
    """Ask the local Ollama model (via the shared async/cached client) --
    small latency win: identical repeated questions during testing or
    real usage hit the disk cache instead of a fresh LLM round-trip, and
    this no longer blocks the event loop the way the old synchronous
    requests.post call did."""
    try:
        label = await ollama_chat(_INTENT_SYSTEM_PROMPT, text, temperature=0.0)
        label = label.strip().lower()
        print(f"[chat] intent classifier raw output: {label!r}")
        for candidate in ("knowledge_question", "predict_command", "pond_data", "chat"):
            if candidate in label:
                return candidate
    except Exception as e:
        print(f"[chat] intent classification failed, defaulting to pond_data: {e}")
    return "pond_data"'''

content = content[:match.start()] + new_func + content[match.end():]

# 3. Fix the call site: _classify_intent is now async, needs await
content = content.replace(
    "intent = _classify_intent(text)",
    "intent = await _classify_intent(text)",
    1,
)

# 4. Remove the redundant duplicate species lookup in the /chat handler --
# use rag_result["images"] (already computed, concurrently, inside
# answer_knowledge_question) instead of calling extract_species_name again
old_block = '''        rag_result = await answer_knowledge_question(request.farmer_text)
        images = []
        species_name = extract_species_name(request.farmer_text)
        if species_name:
            images = get_species_images(species_name)
        return {
            "status": "knowledge_answer",
            "reply": rag_result["answer"],
            "sources": rag_result["sources"],
            "images": images,
            "known_fields": known_fields,
        }'''
new_block = '''        rag_result = await answer_knowledge_question(request.farmer_text)
        # images computed once, concurrently with the answer, inside
        # answer_knowledge_question -- no need to call extract_species_name
        # a second time here (that used to happen and doubled latency for
        # zero benefit, since this second call's result overwrote the
        # first one's images anyway).
        return {
            "status": "knowledge_answer",
            "reply": rag_result["answer"],
            "sources": rag_result["sources"],
            "images": rag_result.get("images", []),
            "known_fields": known_fields,
        }'''

if old_block not in content:
    print("WARNING: exact /chat duplicate-lookup block not found -- skipping that part.")
    print("You may need to remove the duplicate extract_species_name call in main.py manually.")
else:
    content = content.replace(old_block, new_block, 1)
    print("Removed duplicate species lookup in /chat handler.")

MAIN_PY.write_text(content, encoding="utf-8")
print("Patched successfully.")
