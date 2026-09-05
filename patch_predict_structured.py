"""Fixes /predict/structured returning an un-awaited coroutine after
predict() became async in the v1.2 rewrite. Run from repo root:
    python patch_predict_structured.py"""
from pathlib import Path

MAIN_PY = Path("src/api/main.py")
content = MAIN_PY.read_text(encoding="utf-8")

old = '''@app.post("/predict/structured", response_model=PredictionResponse)
def predict_structured(params: PondParameters):
    """Alias for /predict."""
    return predict(params)'''

new = '''@app.post("/predict/structured", response_model=PredictionResponse)
async def predict_structured(params: PondParameters):
    """Alias for /predict. NOTE: must be async and await predict() --
    predict() became async in the v1.2 rewrite; a plain `def` calling
    `return predict(params)` returns an un-awaited coroutine object
    instead of the actual result, which FastAPI cannot serialize."""
    return await predict(params)'''

if new.split("\n")[0] in content and "await predict(params)" in content:
    print("Already patched.")
    raise SystemExit

if old not in content:
    print("ERROR: exact old block not found -- file may have changed.")
    raise SystemExit

backup = MAIN_PY.with_suffix(".py.bak6")
backup.write_text(content, encoding="utf-8")
print(f"Backup written to {backup}")

content = content.replace(old, new, 1)
MAIN_PY.write_text(content, encoding="utf-8")
print("Patched successfully -- /predict/structured now correctly awaits predict().")
