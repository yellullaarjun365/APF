"""Adds the missing stocking-density few-shot example to _INTENT_SYSTEM_PROMPT
in main.py. Run from repo root: python patch_fewshot_v2.py"""
from pathlib import Path

MAIN_PY = Path("src/api/main.py")
content = MAIN_PY.read_text(encoding="utf-8")

if "how many fish should I stock per hectare -> knowledge_question" in content:
    print("Already patched. Nothing to do.")
    raise SystemExit

anchor = '"how much do I feed my tilapia -> knowledge_question"'
if anchor not in content:
    print("ERROR: anchor not found -- prompt text may have drifted since last patch.")
    print("Search main.py manually for _INTENT_SYSTEM_PROMPT and add this line by hand:")
    print('    "how many fish should I stock per hectare -> knowledge_question\\n"')
    raise SystemExit

backup = MAIN_PY.with_suffix(".py.bak5")
backup.write_text(content, encoding="utf-8")
print(f"Backup written to {backup}")

replacement = (
    '"how much do I feed my tilapia -> knowledge_question\\n"\n'
    '    "how many fish should I stock per hectare -> knowledge_question"'
)
content = content.replace(anchor, replacement, 1)
MAIN_PY.write_text(content, encoding="utf-8")
print("Patched successfully -- added stocking-density few-shot example.")
