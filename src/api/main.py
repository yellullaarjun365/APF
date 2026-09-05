"""APF V1 -- FastAPI prediction and extraction backend (M3).

Endpoints:
  GET  /health              -> health check
  POST /predict             -> structured params -> prediction + interval
  POST /predict/extract     -> free text -> extraction -> prediction
  POST /predict/structured  -> alias for /predict

Run:  uvicorn src.api.main:app --reload --port 8000
"""
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features.build_features import build_features
from explain.llm_explain import generate_explanation
from llm.async_client import OLLAMA_URL, OLLAMA_MODEL, TIMEOUT_S, ollama_chat
OLLAMA_TIMEOUT_S = TIMEOUT_S  # back-compat alias for _classify_intent
OLLAMA_CHAT_URL = OLLAMA_URL.replace("/api/generate", "/api/chat")
from knowledge.rag_answer import answer_knowledge_question, _looks_like_followup
from knowledge.species_images import extract_species_name, get_species_images
import requests

# ------------------------------------------------------------------
# Load model artifacts (fail fast at import time if missing)
# ------------------------------------------------------------------
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "models" / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
INTERVAL_PATH = ARTIFACTS_DIR / "interval.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

if not MODEL_PATH.exists():
    raise RuntimeError(f"Model artifact not found: {MODEL_PATH}. Run src/models/train_baseline.py first.")
if not INTERVAL_PATH.exists():
    raise RuntimeError(f"Interval artifact not found: {INTERVAL_PATH}. Run src/models/train_baseline.py first.")

with open(MODEL_PATH, "rb") as f:
    MODEL = pickle.load(f)

with open(INTERVAL_PATH, "r") as f:
    INTERVAL = json.load(f)

with open(METRICS_PATH, "r") as f:
    METRICS = json.load(f)

# ------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------
app = FastAPI(
    title="AquaPredict API",
    description="Nile Tilapia production forecasting backend",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------------
class PondParameters(BaseModel):
    pond_area_ha: float = Field(..., gt=0, le=10, description="Pond area in hectares")
    stocking_count: int = Field(..., gt=0, le=100000, description="Number of fingerlings stocked")
    initial_weight_g: float = Field(..., gt=0, le=200, description="Average initial weight in grams")
    culture_days: int = Field(..., ge=60, le=365, description="Expected culture duration in days")
    mean_temperature_c: float = Field(..., ge=15, le=40, description="Mean water temperature (C)")
    season: str = Field(..., description="summer | winter | monsoon")
    intensity: str = Field(..., description="extensive | semi-intensive | intensive")
    feed_protein_pct: float = Field(30.0, ge=20, le=50, description="Feed protein percentage")
    mean_do_mg_l: float = Field(..., gt=0, le=20, description="Mean dissolved oxygen (mg/L)")
    min_do_mg_l: float = Field(..., gt=0, le=20, description="Minimum dissolved oxygen (mg/L)")
    mean_ph: float = Field(..., ge=4, le=11, description="Mean pH")
    min_ph: float = Field(..., ge=4, le=11, description="Minimum pH")
    max_temp_c: float = Field(..., ge=15, le=45, description="Maximum temperature (C)")
    min_temp_c: float = Field(..., ge=10, le=40, description="Minimum temperature (C)")

class PredictionResponse(BaseModel):
    status: str
    point_estimate_kg: float
    lower_bound_kg: float
    upper_bound_kg: float
    confidence_level: float
    model_version: str
    dataset_version: str
    top_factors: list
    explanation: str

class TextExtractRequest(BaseModel):
    farmer_text: str = Field(..., min_length=3, max_length=2000, description="Free-text pond description")

class ChatRequest(BaseModel):
    farmer_text: str = Field(..., min_length=1, max_length=2000, description="Free-text chat message")
    known_fields: dict = Field(default_factory=dict, description="Fields accumulated from earlier turns")
    history: list = Field(default_factory=list, description="Last few {role, content} turns, most recent last")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    dataset_version: str
    timestamp: str

# ------------------------------------------------------------------
# Helper: build a single-row DataFrame from params, run prediction
# ------------------------------------------------------------------
def _predict_from_params(params: dict) -> dict:
    """Run the trained model on structured parameters."""
    df = pd.DataFrame([params])
    X, _ = build_features(df)

    # Predict
    pred_kg = float(MODEL.predict(X)[0])

    # Interval
    half_width = INTERVAL.get("half_width", 134.0)
    lower = max(0.0, pred_kg - half_width)
    upper = pred_kg + half_width

    # Feature importance for top factors
    model_step = MODEL.named_steps["model"]
    feat_names = MODEL.named_steps["pre"].get_feature_names_out()
    importances = model_step.feature_importances_
    total_importance = importances.sum() or 1.0
    top_idx = np.argsort(importances)[-5:][::-1]

    def _clean_feature_name(name: str) -> str:
        # Strip ColumnTransformer prefixes like "num__"/"cat__" before display
        return name.split("__", 1)[-1] if "__" in name else name

    top_factors = [
        {
            "feature": _clean_feature_name(feat_names[i]),
            "importance_pct": round(float(importances[i]) / total_importance * 100, 1),
        }
        for i in top_idx
    ]

    return {
        "point_estimate_kg": round(pred_kg, 1),
        "lower_bound_kg": round(lower, 1),
        "upper_bound_kg": round(upper, 1),
        "confidence_level": INTERVAL.get("coverage", 0.90),
        "top_factors": top_factors,
    }

# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_version=METRICS.get("model_version", "unknown"),
        dataset_version=METRICS.get("dataset_version", "unknown"),
        timestamp=datetime.utcnow().isoformat(),
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(params: PondParameters):
    try:
        result = _predict_from_params(params.model_dump())

        # LLM explanation layer (src/explain/llm_explain.py) -- translates
        # the already-computed numbers into plain language; never touches
        # the forecast itself. See PROJECT_MANUAL.md §3.
        explanation = await generate_explanation(params.model_dump(), result)

        return PredictionResponse(
            status="complete",
            point_estimate_kg=result["point_estimate_kg"],
            lower_bound_kg=result["lower_bound_kg"],
            upper_bound_kg=result["upper_bound_kg"],
            confidence_level=result["confidence_level"],
            model_version=METRICS.get("model_version", "unknown"),
            dataset_version=METRICS.get("dataset_version", "unknown"),
            top_factors=result["top_factors"],
            explanation=explanation,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/structured", response_model=PredictionResponse)
async def predict_structured(params: PondParameters):
    """Alias for /predict. NOTE: must be async and await predict() --
    predict() became async in the v1.2 rewrite; a plain `def` calling
    `return predict(params)` returns an un-awaited coroutine object
    instead of the actual result, which FastAPI cannot serialize."""
    return await predict(params)

@app.post("/predict/extract")
async def predict_extract(request: TextExtractRequest):
    """
    Extract structured parameters from free text, then predict.

    For V1, this uses a simple rule-based extractor.
    In V2+, this should call an LLM for robust extraction.
    """
    text = request.farmer_text.lower()

    # Simple keyword-based extraction (placeholder for LLM)
    extracted = _extract_params_from_text(text)

    # Check for missing required fields
    required = ["pond_area_ha", "stocking_count", "culture_days", "mean_temperature_c"]
    missing = [f for f in required if f not in extracted or extracted[f] is None]

    if missing:
        return {
            "status": "incomplete",
            "missing_fields": missing,
            "follow_up_question": _generate_followup(missing),
            "extracted": extracted,
        }

    # Fill defaults for optional fields
    defaults = {
        "initial_weight_g": 15.0,
        "season": "summer",
        "intensity": "semi-intensive",
        "feed_protein_pct": 30.0,
        "mean_do_mg_l": 7.5,
        "min_do_mg_l": 5.0,
        "mean_ph": 7.5,
        "min_ph": 6.8,
        "max_temp_c": 32.0,
        "min_temp_c": 24.0,
    }
    for k, v in defaults.items():
        if k not in extracted or extracted[k] is None:
            extracted[k] = v

    # Run prediction
    result = _predict_from_params(extracted)
    explanation = await generate_explanation(extracted, result)

    return {
        "status": "complete",
        "prediction": {
            "point_estimate_kg": result["point_estimate_kg"],
            "lower_bound_kg": result["lower_bound_kg"],
            "upper_bound_kg": result["upper_bound_kg"],
            "confidence_level": result["confidence_level"],
            "top_factors": result["top_factors"],
        },
        "explanation": explanation,
        "extracted": extracted,
    }

_INTENT_SYSTEM_PROMPT = (
    "You are a strict one-word classifier for a farmer chat assistant. "
    "Do NOT answer, explain, or engage with the content of the message -- "
    "your only job is to output the single matching label word. "
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
    "how much do I feed my tilapia -> knowledge_question\n"
    "how many fish should I stock per hectare -> knowledge_question\n\n"
    "Recent conversation:\n"
    "user: tell me about blue whales\n"
    "assistant: Blue whales are the largest animals known to have ever existed.\n\n"
    "Latest message: tell me more -> knowledge_question\n\n"
    "Recent conversation:\n"
    "user: tell me about blue whales\n"
    "assistant: Blue whales are the largest animals known to have ever existed.\n\n"
    "Latest message: even more -> knowledge_question\n\n"
    "Recent conversation:\n"
    "user: my pond is 0.5 hectares\n"
    "assistant: Got it -- how many fish are stocked?\n\n"
    "Latest message: 3000 -> pond_data"
)


_PREDICT_MARKERS = ("predict", "go ahead", "calculate", "forecast now")


def _looks_like_predict_command(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in _PREDICT_MARKERS)


def _last_assistant_was_knowledge(history: list) -> bool:
    """Heuristic, not perfect: True if the most recent assistant turn was
    a knowledge answer rather than the canned chat greeting or a
    pond-field request. Used only to bias ambiguous follow-up
    classification -- never overrides a message with real pond numbers
    or an explicit predict command (see _classify_intent)."""
    for h in reversed(history or []):
        if h.get("role") == "assistant":
            content = (h.get("content") or "")
            if content.startswith("I\'m AquaPredict AI"):
                return False
            lc = content.lower()
            if "pond area" in lc or "stocking count" in lc or "culture day" in lc or "i need your" in lc:
                return False
            return True
    return False


async def _classify_intent(text: str, history: list = None) -> str:
    """Ask the local Ollama model (via the shared async/cached client) --
    small latency win: identical repeated questions during testing or
    real usage hit the disk cache instead of a fresh LLM round-trip, and
    this no longer blocks the event loop the way the old synchronous
    requests.post call did.

    BUG FIX: this used to classify the latest message in total isolation,
    with no conversation context at all -- so a terse follow-up like
    "i mean the text information" (which only makes sense as a reply to
    the previous knowledge-question turn) had nothing to go on and got
    misclassified as "chat", producing the generic "I'm AquaPredict AI..."
    intro instead of continuing the actual topic.

    REGRESSION FIX (same day): the first attempt at this wrapped EVERY
    message in a "Recent conversation: ...\\nLatest message: ..." block,
    which broke clear questions like "tell me about blue whales" -- that
    input shape doesn't match the plain one-line examples in the system
    prompt at all, and the small local model started defaulting to "chat"
    for messages that used to classify correctly. Now only wraps messages
    that actually look like vague follow-ups (short, or built from
    context-dependent words like "more"/"that"/"mean") -- a clear
    standalone question is classified exactly as it was before any of
    this history-awareness work, unchanged."""
    # Deterministic short-circuit (added after the wrapped-format few-shot
    # fix still wasn't reliable for every phrasing of "more"): an
    # ambiguous follow-up right after a knowledge answer almost certainly
    # continues that same topic. Don't gamble this on the LLM's one-shot
    # generalization -- unless the message has real pond numbers or is an
    # explicit predict command, in which case let normal classification
    # (or the pond-data extraction path) handle it as before.
    if (
        _looks_like_followup(text)
        and not _looks_like_predict_command(text)
        and not any(ch.isdigit() for ch in text)
        and _last_assistant_was_knowledge(history)
    ):
        return "knowledge_question"

    user_input = text
    if history and _looks_like_followup(text):
        history_lines = "\n".join(
            f"{h.get('role', '?')}: {h.get('content', '')}" for h in history[-6:]
        )
        user_input = f"Recent conversation:\n{history_lines}\n\nLatest message: {text}"
    try:
        label = await ollama_chat(_INTENT_SYSTEM_PROMPT, user_input, temperature=0.0, max_tokens=6)
        label = label.strip().lower()
        print(f"[chat] intent classifier raw output: {label!r}")
        for candidate in ("knowledge_question", "predict_command", "pond_data", "chat"):
            if candidate in label:
                return candidate
        # BUG FIX: llama3.2 frequently ignores the "reply with ONLY one
        # word" instruction for genuinely interesting questions and answers
        # them in full instead (observed: asked to classify "tell me about
        # blue whales", it returned "blue whales are the largest animals
        # known to have ever existed" -- a real answer, not a label). That
        # raw text matches none of the four candidates above, so the loop
        # falls through here. Previously this silently defaulted to "chat"
        # every time, producing the generic canned intro instead of routing
        # to the knowledge-question/RAG path. A long unstructured response
        # that isn't pond data and isn't a predict command is itself strong
        # evidence the model treated this as a real question worth
        # answering -- so treat it as knowledge_question rather than chat.
        if not _looks_like_predict_command(text) and not any(ch.isdigit() for ch in text):
            print("[chat] classifier answered instead of labeling -- defaulting to knowledge_question")
            return "knowledge_question"
    except Exception as e:
        print(f"[chat] intent classification failed: {e}")
    # No confident label -- either Ollama errored, or its output didn't
    # match a known candidate. Default to continuing the previous topic
    # rather than assuming pond_data, which produces a confusing
    # off-topic reply when the classifier simply hiccuped on an
    # unrelated question (see patch_intent_fallback_default.py).
    if _last_assistant_was_knowledge(history):
        return "knowledge_question"
    return "chat"
FIELD_LABELS_AND_WHY = {
    "pond_area_ha": ("pond area (hectares)", "it sets the stocking density and total biomass the pond can realistically support"),
    "stocking_count": ("number of fish stocked", "total fish count is the single biggest driver of total harvest weight"),
    "culture_days": ("culture duration (days)", "longer culture periods mean more growth time, directly changing expected harvest size"),
    "mean_temperature_c": ("water temperature", "tilapia growth rate is highly temperature-dependent -- too cold or too hot slows growth or raises stress"),
}


def _generate_dynamic_followup(missing_fields: list, known_fields: dict, history: list) -> str:
    """Ask Ollama to naturally acknowledge known fields, ask for missing
    ones with a brief reason why each matters, and respond sensibly if
    the farmer's last message was a question (e.g. "why do you need
    that") rather than new data. Falls back to the static template on
    any Ollama failure."""
    known_str = ", ".join(f"{k}={v}" for k, v in known_fields.items()) or "none yet"
    missing_info = "; ".join(
        f"{FIELD_LABELS_AND_WHY.get(f, (f, 'it affects the forecast'))[0]} (needed because {FIELD_LABELS_AND_WHY.get(f, (f, 'it affects the forecast'))[1]})"
        for f in missing_fields
    )
    history_lines = "\n".join(f"{h.get('role', '?')}: {h.get('content', '')}" for h in (history or [])[-6:])

    prompt = f"""You are a friendly assistant helping a farmer provide pond details for a Nile tilapia harvest forecast.

Recent conversation:
{history_lines}

Already known: {known_str}
Still missing: {missing_info}

Write a short, natural reply (2-4 sentences). Briefly acknowledge what's already known, then ask for the missing field(s), explaining briefly why each matters. If the farmer's last message was a question (e.g. asking why something is needed) rather than new pond data, answer that question directly and warmly instead of just repeating the request. Do not invent field values. Do not use markdown formatting -- plain sentences only."""

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "keep_alive": "30m"},
            timeout=OLLAMA_TIMEOUT_S,
        )
        resp.raise_for_status()
        out = resp.json().get("response", "").strip()
        if out:
            return out
    except Exception as e:
        print(f"[chat] dynamic followup failed, using static fallback: {e}")
    return _generate_followup(missing_fields)


@app.post("/chat")
async def chat(request: ChatRequest):
    """Dynamic multi-turn chat endpoint (V1-M5). Classifies intent, merges
    newly-extracted fields into known_fields server-side, and predicts
    either when all required fields are known or the farmer explicitly
    asks for a prediction (whichever comes first)."""
    text = request.farmer_text.lower()
    known_fields = dict(request.known_fields or {})
    history = request.history or []

    intent = await _classify_intent(text, history)

    if intent == "chat":
        return {"status": "chat", "reply": (
            "I'm AquaPredict AI -- tell me about your pond (area, fish stocked, "
            "culture days, temperature, DO, pH) and I'll forecast your harvest. "
            "You can give details across a few messages; just say \"predict\" when you're ready."
        ), "known_fields": known_fields}

    if intent == "knowledge_question":
        rag_result = await answer_knowledge_question(request.farmer_text, history)
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
        }

    newly_extracted = _extract_params_from_text(text)
    known_fields.update({k: v for k, v in newly_extracted.items() if v is not None})

    required = ["pond_area_ha", "stocking_count", "culture_days", "mean_temperature_c"]
    missing = [f for f in required if f not in known_fields or known_fields[f] is None]

    if missing:
        return {
            "status": "need_more",
            "known_fields": known_fields,
            "missing_fields": missing,
            "follow_up_question": _generate_dynamic_followup(missing, known_fields, history),
        }

    defaults = {
        "initial_weight_g": 15.0, "season": "summer", "intensity": "semi-intensive",
        "feed_protein_pct": 30.0, "mean_do_mg_l": 7.5, "min_do_mg_l": 5.0,
        "mean_ph": 7.5, "min_ph": 6.8, "max_temp_c": 32.0, "min_temp_c": 24.0,
    }
    full_params = dict(known_fields)
    for k, v in defaults.items():
        if k not in full_params or full_params[k] is None:
            full_params[k] = v

    result = _predict_from_params(full_params)
    explanation = await generate_explanation(full_params, result)

    return {
        "status": "predicted",
        "known_fields": known_fields,
        "prediction": {
            "point_estimate_kg": result["point_estimate_kg"],
            "lower_bound_kg": result["lower_bound_kg"],
            "upper_bound_kg": result["upper_bound_kg"],
            "confidence_level": result["confidence_level"],
            "top_factors": result["top_factors"],
        },
        "explanation": explanation,
    }


# ------------------------------------------------------------------
# Simple rule-based text extractor (placeholder for LLM)
# ------------------------------------------------------------------
def _extract_params_from_text(text: str) -> dict:
    import re
    extracted = {}

    # Pond area
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:ha|hectare|hectares|acre|acres)', text)
    if m:
        val = float(m.group(1))
        if 'acre' in text:
            val *= 0.4047  # acres to hectares
        extracted["pond_area_ha"] = round(val, 2)

    # Stocking count
    m = re.search(r'(\d+(?:,\d+)*)\s*(?:tilapia\s+)?(?:fish|fingerlings|stocked|stocking)', text)
    if not m:
        m = re.search(r'(\d+(?:,\d+)*)\s*tilapia', text)
    if m:
        extracted["stocking_count"] = int(m.group(1).replace(',', ''))

    # Culture days
    m = re.search(r'(\d+)\s*culture\s*days?', text)
    if not m:
        m = re.search(r'(\d+)\s*days?\s*(?:of\s*)?culture', text)
    if not m:
        m = re.search(r'culture\s*(?:period|duration)?\s*(?:of|is)?\s*(\d+)\s*days?', text)
    if not m:
        m = re.search(r'(\d+)\s*(?:day|days|week|weeks|month|months)', text)
    if m:
        val = int(m.group(1))
        if 'week' in text:
            val *= 7
        elif 'month' in text:
            val *= 30
        extracted["culture_days"] = val

    # Temperature
    # BUG FIX: none of the three patterns here had re.IGNORECASE, so "28°C"
    # (capital C, which is how basically everyone writes it) matched NONE
    # of them -- the degree-symbol pattern needed a lowercase "c", and the
    # bare-"c" fallback needed a lowercase "c" too. Added IGNORECASE to all
    # three; the fallback's `(?![a-z])` lookahead already correctly guards
    # against matching mid-word (e.g. the "c" in "culture"), so that part
    # didn't need to change.
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:\u00b0c|degrees?\s*c)\b', text, re.IGNORECASE)
    if not m:
        m = re.search(r'(\d+(?:\.\d+)?)\s*c(?![a-z])', text, re.IGNORECASE)
    if not m:
        m = re.search(r'(?:temperature|temp|water)[^\d]{0,20}(\d+(?:\.\d+)?)\s*degrees?\b', text, re.IGNORECASE)
    if m:
        extracted["mean_temperature_c"] = float(m.group(1))

    # DO (dissolved oxygen)
    # BUG FIX: only matched number-THEN-unit order ("7.5 mg/L"), but natural
    # phrasing is often label-THEN-number ("dissolved oxygen 7.5 mg/L",
    # "DO 7.5") -- silently matched nothing for that far more common order.
    # Now tries both orders; case-insensitive so "mg/L" matches too.
    m = re.search(r'(\d+(?:\.\d+)?)\s*mg\s*/\s*l\b', text, re.IGNORECASE)
    if not m:
        m = re.search(r'(?:dissolved\s*oxygen|\bdo\b)\D{0,12}?(\d+(?:\.\d+)?)', text, re.IGNORECASE)
    if m:
        extracted["mean_do_mg_l"] = float(m.group(1))

    # pH
    # BUG FIX: no re.IGNORECASE -- only matched lowercase "ph", never "pH"
    # (capital P), which is how it's conventionally written.
    m = re.search(r'ph\s*(?:is|of|=)?\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
    if m:
        extracted["mean_ph"] = float(m.group(1))

    # Initial weight
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:g|grams?)', text)
    if m:
        val = float(m.group(1))
        if val < 5:  # likely kg, convert to g
            val *= 1000
        extracted["initial_weight_g"] = val

    # Season
    for s in ["summer", "winter", "monsoon"]:
        if s in text:
            extracted["season"] = s
            break

    # Intensity
    for i in ["extensive", "semi-intensive", "intensive"]:
        if i in text:
            extracted["intensity"] = i
            break

    # Derive min values from means if not provided
    if "mean_do_mg_l" in extracted and "min_do_mg_l" not in extracted:
        extracted["min_do_mg_l"] = extracted["mean_do_mg_l"] - 1.5
    if "mean_ph" in extracted and "min_ph" not in extracted:
        extracted["min_ph"] = extracted["mean_ph"] - 0.3
    if "mean_temperature_c" in extracted:
        if "max_temp_c" not in extracted:
            extracted["max_temp_c"] = extracted["mean_temperature_c"] + 3
        if "min_temp_c" not in extracted:
            extracted["min_temp_c"] = extracted["mean_temperature_c"] - 3

    return extracted

def _generate_followup(missing_fields: list) -> str:
    """Generate a natural follow-up question for missing fields."""
    field_names = {
        "pond_area_ha": "pond area",
        "stocking_count": "number of fish stocked",
        "culture_days": "culture duration",
        "mean_temperature_c": "water temperature",
    }
    names = [field_names.get(f, f) for f in missing_fields]
    if len(names) == 1:
        return f"I need your {names[0]} to make a forecast. Could you provide that?"
    return f"I need a few more details: {', '.join(names)}. Could you provide these?"



# ------------------------------------------------------------------
# Run with: uvicorn src.api.main:app --reload --port 8000
# ------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
