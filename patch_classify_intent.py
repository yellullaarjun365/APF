"""
One-time patch script: replaces the old _classify_intent function in
src/api/main.py with the fixed version (uses /api/chat + few-shot examples
instead of raw /api/generate). Run once from the repo root:

    python patch_classify_intent.py

Safe to re-run -- it checks whether the old function is still present
before touching anything, and makes a .bak backup first.
"""
import re
from pathlib import Path

MAIN_PY = Path("src/api/main.py")

OLD_FUNC = '''def _classify_intent(text: str) -> str:
    """Ask the local Ollama model whether this message is small talk,
    a pond-data description, or an explicit predict command. Falls back
    to "pond_data" on any Ollama failure so extraction still runs --
    the safest default when we can't classify."""
    prompt = (
        "Classify this farmer chat message into exactly one word: "
        "\\"chat\\" (greeting, small talk, question about the app itself), "
        "\\"predict_command\\" (explicitly asking for the forecast/prediction now, e.g. \\"predict\\", \\"go ahead\\", \\"calculate it\\"), "
        "\\"pond_data\\" (contains or describes THIS farmer's own pond/fish/farm parameters, e.g. area, count, days, temperature of their pond), "
        "or \\"knowledge_question\\" (a general question about Nile tilapia biology or aquaculture practice, not about this farmer's own pond data, e.g. \\"why does pH matter for tilapia\\", \\"what temperature do tilapia prefer\\", \\"how do tilapia reproduce\\"). "
        "Reply with ONLY that one word, nothing else.\\n\\n"
        f"Message: {text}"
    )
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "keep_alive": "30m"},
            timeout=OLLAMA_TIMEOUT_S,
        )
        resp.raise_for_status()
        label = resp.json().get("response", "").strip().lower()
        print(f"[chat] intent classifier raw output: {label!r}")
        for candidate in ("knowledge_question", "predict_command", "pond_data", "chat"):
            if candidate in label:
                return candidate
    except Exception as e:
        print(f"[chat] intent classification failed, defaulting to pond_data: {e}")
    return "pond_data"'''

NEW_FUNC = '''_INTENT_SYSTEM_PROMPT = (
    "You are a strict one-word classifier for a farmer chat assistant. "
    "Reply with ONLY one of these exact words, nothing else, no punctuation, "
    "no explanation: chat, predict_command, pond_data, knowledge_question\\n\\n"
    "chat = greeting, small talk, or a question about the app itself.\\n"
    "predict_command = explicitly asking for the forecast/prediction now "
    "(e.g. \\"predict\\", \\"go ahead\\", \\"calculate it\\").\\n"
    "pond_data = contains or describes THIS farmer's own pond/fish/farm "
    "parameters (area, count, days, temperature of their specific pond).\\n"
    "knowledge_question = a general question about a species, biology, "
    "aquaculture practice, water quality thresholds, or farming technique "
    "-- NOT about this farmer's own pond data. This includes questions "
    "asking to see or learn about any animal (tilapia or otherwise), and "
    "general threshold/practice questions that don't mention the farmer's "
    "own numbers.\\n\\n"
    "Examples:\\n"
    "hello -> chat\\n"
    "predict -> predict_command\\n"
    "go ahead and calculate it -> predict_command\\n"
    "my pond is 0.5 hectares with 3000 tilapia -> pond_data\\n"
    "why does pH matter for tilapia -> knowledge_question\\n"
    "what temperature do tilapia prefer -> knowledge_question\\n"
    "how do tilapia reproduce -> knowledge_question\\n"
    "show me a tilapia -> knowledge_question\\n"
    "tell me about blue whales -> knowledge_question\\n"
    "what dissolved oxygen level is dangerous -> knowledge_question\\n"
    "how much do I feed my tilapia -> knowledge_question"
)


def _classify_intent(text: str) -> str:
    """Ask the local Ollama model (via /api/chat, system/user role split --
    NOT raw /api/generate, which chat-tuned models don't reliably follow
    for classification tasks) whether this message is small talk, a
    pond-data description, an explicit predict command, or a general
    knowledge question. Falls back to "pond_data" on any Ollama failure
    so extraction still runs -- the safest default when we can't classify."""
    try:
        resp = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": _INTENT_SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                "stream": False,
                "keep_alive": "30m",
                "options": {"temperature": 0.0},
            },
            timeout=OLLAMA_TIMEOUT_S,
        )
        resp.raise_for_status()
        label = resp.json().get("message", {}).get("content", "").strip().lower()
        print(f"[chat] intent classifier raw output: {label!r}")
        for candidate in ("knowledge_question", "predict_command", "pond_data", "chat"):
            if candidate in label:
                return candidate
    except Exception as e:
        print(f"[chat] intent classification failed, defaulting to pond_data: {e}")
    return "pond_data"'''


def main():
    if not MAIN_PY.exists():
        print(f"ERROR: {MAIN_PY} not found. Run this from the repo root (APF/).")
        return

    content = MAIN_PY.read_text(encoding="utf-8")

    if NEW_FUNC.split("\n")[0] in content and "_INTENT_SYSTEM_PROMPT" in content:
        print("Already patched -- _INTENT_SYSTEM_PROMPT found. Nothing to do.")
        return

    if OLD_FUNC not in content:
        print("ERROR: Could not find the exact old _classify_intent function text.")
        print("The file may have been edited since. Manual patch needed --")
        print("see main_classify_intent_patch.py for the target function.")
        return

    # Backup first
    backup_path = MAIN_PY.with_suffix(".py.bak")
    backup_path.write_text(content, encoding="utf-8")
    print(f"Backup written to {backup_path}")

    new_content = content.replace(OLD_FUNC, NEW_FUNC)

    # Also insert OLLAMA_CHAT_URL if not already present (in case the
    # manual paste attempt earlier didn't actually land in the file)
    import_line = "from explain.llm_explain import generate_explanation, OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT_S"
    if "OLLAMA_CHAT_URL" not in new_content:
        if import_line in new_content:
            new_content = new_content.replace(
                import_line,
                import_line + '\nOLLAMA_CHAT_URL = OLLAMA_URL.replace("/api/generate", "/api/chat")',
            )
            print("Inserted OLLAMA_CHAT_URL line (wasn't found already present).")
        else:
            print("WARNING: could not find the import line to attach OLLAMA_CHAT_URL to.")
            print("Add this manually near the top of the file:")
            print('  OLLAMA_CHAT_URL = OLLAMA_URL.replace("/api/generate", "/api/chat")')

    MAIN_PY.write_text(new_content, encoding="utf-8")
    print(f"Patched {MAIN_PY} successfully.")


if __name__ == "__main__":
    main()
