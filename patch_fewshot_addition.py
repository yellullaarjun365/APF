"""
Adds one missing few-shot example to _INTENT_SYSTEM_PROMPT in main.py to fix
the "how many fish should I stock per hectare" -> pond_data misclassification.
Run from repo root: python patch_fewshot_addition.py
"""
from pathlib import Path

MAIN_PY = Path("src/api/main.py")
content = MAIN_PY.read_text(encoding="utf-8")

ANCHOR = "how much do I feed my tilapia -> knowledge_question"
NEW_LINE = '\n    "how many fish should I stock per hectare -> knowledge_question\\n"'

if "how many fish should I stock per hectare" in content:
    print("Already patched. Nothing to do.")
    raise SystemExit

if ANCHOR not in content:
    print("ERROR: anchor line not found -- prompt may have changed since last patch.")
    raise SystemExit

new_content = content.replace(
    f'"{ANCHOR}"',
    f'"{ANCHOR}\\n"{NEW_LINE.strip()[1:]}',
    1,
)
# Simpler, safer approach: just insert a new line right after the anchor line's quote+\n
old_snippet = f'"{ANCHOR}"'
new_snippet = f'"{ANCHOR}\\n"\n    "how many fish should I stock per hectare -> knowledge_question"'
new_content = content.replace(old_snippet, new_snippet, 1)

backup = MAIN_PY.with_suffix(".py.bak3")
backup.write_text(content, encoding="utf-8")
print(f"Backup written to {backup}")

MAIN_PY.write_text(new_content, encoding="utf-8")
print("Patched successfully -- added stocking-density few-shot example.")
