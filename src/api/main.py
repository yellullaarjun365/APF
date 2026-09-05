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
    top_idx = np.argsort(importances)[-5:][::-1]
    top_factors = [
        {"feature": feat_names[i], "importance": float(importances[i])}
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
def predict(params: PondParameters):
    try:
        result = _predict_from_params(params.model_dump())

        # Simple explanation (placeholder for LLM explanation layer)
        explanation = _generate_explanation(params.model_dump(), result)

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
def predict_structured(params: PondParameters):
    """Alias for /predict."""
    return predict(params)

@app.post("/predict/extract")
def predict_extract(request: TextExtractRequest):
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
    explanation = _generate_explanation(extracted, result)

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

# ------------------------------------------------------------------
# Simple rule-based text extractor (placeholder for LLM)
# ------------------------------------------------------------------
def _extract_params_from_text(text: str) -> dict:
    import re
    extracted = {}

    # Pond area
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:ha|hectares?|acres?)\b', text, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        if 'acre' in text.lower():
            val *= 0.4047  # acres to hectares
        extracted["pond_area_ha"] = round(val, 2)

    # Stocking count
    m = re.search(r'(\d+(?:,\d+)*)\s*(?:fish|fingerlings|stocked|stocking)', text, re.IGNORECASE)
    if m:
        extracted["stocking_count"] = int(m.group(1).replace(',', ''))

    # Culture days
    # BUG FIX: the original pattern required the number and the unit word
    # ("day"/"days"/etc) to be directly adjacent. "120 culture days" has
    # "culture" in between, so it never matched at all -- the extractor
    # reported culture_days as missing even though it was right there in
    # the text. Now allows one optional descriptive word (specifically
    # "culture", the term farmers actually use) between the number and unit.
    m = re.search(r'(\d+)\s*(?:culture\s+)?(?:days?|weeks?|months?)\b', text, re.IGNORECASE)
    if m:
        val = int(m.group(1))
        unit = m.group(0).lower()
        if 'week' in unit:
            val *= 7
        elif 'month' in unit:
            val *= 30
        extracted["culture_days"] = val

    # Temperature
    # BUG FIX: the original pattern included a bare "c" as a valid unit
    # match with no word boundary -- so `re.search` matched the "c" in
    # "culture" right after an unrelated number ("120 culture days" ->
    # matched "120 c" as if it meant 120 degrees Celsius), and because
    # re.search only returns the first/leftmost match, the REAL
    # temperature ("28°C") later in the sentence was never even reached.
    # Now requires an actual degree symbol before "c", or the word
    # "degree(s)" -- never a bare letter. Case-insensitive for "°C"/"°c".
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:\u00b0\s*c\b|degrees?\b)', text, re.IGNORECASE)
    if m:
        extracted["mean_temperature_c"] = float(m.group(1))

    # DO (dissolved oxygen)
    # BUG FIX: the original pattern only matched number-THEN-unit order
    # ("7.5 mg/L"), but natural phrasing is often label-THEN-number
    # ("dissolved oxygen 7.5 mg/L", "DO 7.5") -- so it silently matched
    # nothing for the more common phrasing. Now tries both orders, and is
    # case-insensitive so "mg/L" (capital L) matches too.
    m = re.search(r'(\d+(?:\.\d+)?)\s*mg\s*/\s*l\b', text, re.IGNORECASE)
    if not m:
        m = re.search(r'(?:dissolved\s*oxygen|\bdo\b)\D{0,12}?(\d+(?:\.\d+)?)', text, re.IGNORECASE)
    if m:
        extracted["mean_do_mg_l"] = float(m.group(1))

    # pH
    # BUG FIX: original pattern was case-sensitive and only matched lowercase
    # "ph" -- "pH" (capital P, as basically everyone writes it) never matched.
    m = re.search(r'ph\s*(?:is|of|=)?\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
    if m:
        extracted["mean_ph"] = float(m.group(1))

    # Initial weight
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:g|grams?)\b', text, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        if val < 5:  # likely kg, convert to g
            val *= 1000
        extracted["initial_weight_g"] = val

    # Season
    for s in ["summer", "winter", "monsoon"]:
        if s in text.lower():
            extracted["season"] = s
            break

    # Intensity
    for i in ["extensive", "semi-intensive", "intensive"]:
        if i in text.lower():
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

def _generate_explanation(params: dict, result: dict) -> str:
    """Generate a simple explanation of the prediction."""
    pe = result["point_estimate_kg"]
    lb = result["lower_bound_kg"]
    ub = result["upper_bound_kg"]
    area = params.get("pond_area_ha", 0.5)
    density = params.get("stocking_count", 3000) / area if area > 0 else 0

    explanation = (
        f"Based on your pond parameters, I estimate a harvest of **{pe:.0f} kg** "
        f"({lb:.0f}--{ub:.0f} kg at 90% confidence). "
        f"With {params.get('stocking_count', 0):,} fish in {area:.1f} ha "
        f"(density ~{density:,.0f} fish/ha) over {params.get('culture_days', 0)} days, "
        f"this yield is consistent with {params.get('intensity', 'semi-intensive')} "
        f"Nile tilapia culture under {params.get('mean_temperature_c', 28)}C conditions."
    )

    # Add water quality commentary
    do = params.get("mean_do_mg_l", 7.5)
    temp = params.get("mean_temperature_c", 28)
    if do < 4:
        explanation += " Note: Your DO levels are low -- consider aeration to avoid mortality."
    elif temp > 32:
        explanation += " Note: High temperatures increase stress risk -- monitor DO closely."

    return explanation

# ------------------------------------------------------------------
# Run with: uvicorn src.api.main:app --reload --port 8000
# ------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
