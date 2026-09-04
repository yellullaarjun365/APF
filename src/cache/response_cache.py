"""Disk-backed LRU cache for Ollama responses."""
import hashlib, json, os, time
from pathlib import Path

CACHE_DIR = Path("data/cache/ollama")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MAX_CACHE_AGE_HOURS = float(os.environ.get("CACHE_TTL_HOURS", "24"))

def _key(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest()[:32]

def get_cached(model: str, prompt: str):
    key = _key(model, prompt)
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        age_h = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_h < MAX_CACHE_AGE_HOURS:
            return json.loads(cache_file.read_text())
    return None

def set_cached(model: str, prompt: str, result: dict) -> None:
    key = _key(model, prompt)
    (CACHE_DIR / f"{key}.json").write_text(json.dumps(result))
