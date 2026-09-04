import re
from pathlib import Path

MAIN_PY = Path("src/api/main.py")
content = MAIN_PY.read_text(encoding="utf-8")

if "_INTENT_SYSTEM_PROMPT" in content:
    print("Already patched -- _INTENT_SYSTEM_PROMPT found. Nothing to do.")
    raise SystemExit

NEW_FUNC = '''_INTENT_SYSTEM_PROMPT = (
    "You are a strict one-word classifier for a farmer chat assistant. "
    "Reply with ONLY one of these exact words, nothing else, no punctuation, "
    "no explanation: chat, predict_command, pond_data, knowledge_question\n\n"
    "chat = greeting, small talk, or a question about the app itself.\n"
    "predict_command = explicitly asking for the forecast/prediction now "
    "(e.g. \"predict\", \"go ahead\", \"calculate it\").\n"
    "pond_data = contains or describes THIS farmer's own pond/fish/farm "
    "parameters (area, count, days, temperature of their specific pond).\n"
    "knowledge_question = a general question about a species, biology, "
    "aquaculture practice, water quality thresholds, or farming technique "
    "-- NOT about this farmer's own pond data. This includes questions "
    "asking to see or learn about any animal (tilapia or otherwise), and "
    "general threshold/practice questions that don't mention the farmer's "
    "own numbers.\n\n"
    "Examples:\n"
    "hello -> chat\n"
    "predict -> predict_command\n"
    "go ahead and calculate it -> predict_command\n"
    "my pond is 0.5 hectares with 3000 tilapia -> pond_data\n"
    "why does pH matter for tilapia -> knowledge_question\n"
    "what temperature do tilapia prefer -> knowledge_question\n"
    "how do tilapia reproduce -> knowledge_question\n"
    "show me a tilapia -> knowledge_question\n"
    "tell me about blue whales -> knowledge_question\n"
    "what dissolved oxygen level is dangerous -> knowledge_question\n"
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

# Match from "def _classify_intent" through to (but not including) the next
# top-level "def " or a top-level assignment like "FIELD_LABELS_AND_WHY ="
# -- i.e. everything up to the next line that starts at column 0 with
# something that isn't blank/comment/indented.
pattern = re.compile(
    r"def _classify_intent\(text: str\) -> str:.*?(?=\n(?:def |FIELD_LABELS_AND_WHY|@app\.))",
    re.DOTALL,
)

match = pattern.search(content)
if not match:
    print("ERROR: could not locate _classify_intent function boundaries.")
    print("Paste back the output of:  Select-String -Path src\\api\\main.py -Pattern \"_classify_intent|FIELD_LABELS_AND_WHY\" -Context 0,0")
    raise SystemExit

backup = MAIN_PY.with_suffix(".py.bak")
backup.write_text(content, encoding="utf-8")
print(f"Backup written to {backup}")

new_content = content[:match.start()] + NEW_FUNC + content[match.end():]
MAIN_PY.write_text(new_content, encoding="utf-8")
print("Patched successfully.")
