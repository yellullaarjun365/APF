"""APF V1 -- FastAPI prediction endpoint (M3).
Serves the trained baseline model. Accepts structured JSON or
free-text (via the NLP extraction module) and returns a point
estimate + 90% prediction interval + top driving factors.

Run:  uvicorn src.api.main:app --reload --port 8000
"""
import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features.build_features import build_features

# ------------------------------------------------------------------
# Load artifacts
# ------------------------------------------------------------------
ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "models" / "artifacts"

try:
    with open(ARTIFACT_DIR / "model.pkl", "rb") as f:
        MODEL = pickle.load(f)
    with open(ARTIFACT_DIR / "interval.json", "r") as f:
        INTERVAL = json.load(f)
    MODEL_LOADED = True
except Exception as e:
    MODEL = None
    INTERVAL = {"coverage": 0.90, "half_width": 200.0, "method": "fallback", "fitted_on": "unknown"}
    MODEL_LOADED = False
    print(f"WARNING: Could not load model artifact: {e}")

# Try to get expected feature names; fall back to build_features columns
if MODEL_LOADED:
    try:
        EXPECTED_FEATURES = list(MODEL.named_steps["pre"].get_feature_names_out())
    except Exception:
        EXPECTED_FEATURES = None
else:
    EXPECTED_FEATURES = None

app = FastAPI(title="APF V1 -- Aquaculture Production Forecasting")

# ------------------------------------------------------------------
# Pydantic schemas
# ------------------------------------------------------------------
class StructuredInput(BaseModel):
    """Direct structured input — every field a farmer might know on day one."""
    pond_area_ha: float = Field(..., gt=0, description="Pond area in hectares")
    stocking_count: int = Field(..., gt=0, description="Number of fingerlings stocked")
    initial_weight_g: float = Field(..., gt=0, description="Average initial weight in grams")
    culture_days: int = Field(..., gt=0, description="Planned culture duration in days")
    mean_temperature_c: float = Field(..., description="Expected mean water temperature (°C)")
    season: str = Field(..., pattern="^(summer|winter|monsoon)$")
    intensity: str = Field(..., pattern="^(extensive|semi-intensive|intensive)$")
    feed_protein_pct: int = Field(..., ge=20, le=50, description="Feed protein %")
    mean_do_mg_l: float = Field(..., gt=0, description="Expected mean dissolved oxygen (mg/L)")
    min_do_mg_l: float = Field(..., gt=0, description="Expected minimum dissolved oxygen (mg/L)")
    mean_ph: float = Field(..., gt=4, lt=11, description="Expected mean pH")
    min_ph: float = Field(..., gt=4, lt=11, description="Expected minimum pH")
    max_temp_c: float = Field(..., description="Expected maximum temperature (°C)")
    min_temp_c: float = Field(..., description="Expected minimum temperature (°C)")

class PredictionOutput(BaseModel):
    point_estimate_kg: float
    lower_bound_kg: float
    upper_bound_kg: float
    interval_coverage: float
    top_factors: list[dict]
    model_version: str
    dataset_version: str

# ------------------------------------------------------------------
# Helper: run prediction through the shared feature pipeline
# ------------------------------------------------------------------
def _predict_from_df(df: pd.DataFrame) -> dict:
    """Internal helper used by both /predict and /extract endpoints."""
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded. Check artifacts.")

    X, _ = build_features(df)

    # The ColumnTransformer in the pipeline handles missing/extra columns
    # via imputation and OHE, so we just pass X through directly.
    try:
        pred = float(MODEL.predict(X)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    hw = INTERVAL.get("half_width", 200.0)

    # Local feature importance via simple permutation
    base_pred = pred
    importances = []
    for col in X.columns:
        if col in ["season", "intensity"]:
            continue
        X_pert = X.copy()
        X_pert[col] = X_pert[col].median()
        try:
            pert_pred = float(MODEL.predict(X_pert)[0])
            importances.append({"feature": col, "impact_kg": round(base_pred - pert_pred, 2)})
        except Exception:
            continue

    importances.sort(key=lambda x: abs(x["impact_kg"]), reverse=True)

    return {
        "point_estimate_kg": round(pred, 1),
        "lower_bound_kg": round(max(pred - hw, 0), 1),
        "upper_bound_kg": round(pred + hw, 1),
        "interval_coverage": INTERVAL.get("coverage", 0.90),
        "top_factors": importances[:5],
        "model_version": "v1.1.0-baseline",
        "dataset_version": INTERVAL.get("fitted_on", "unknown"),
    }

# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL_LOADED,
        "interval_loaded": MODEL_LOADED,
        "artifact_dir": str(ARTIFACT_DIR),
    }

@app.post("/predict", response_model=PredictionOutput)
def predict_structured(payload: StructuredInput):
    """Predict harvest from fully-structured farmer input."""
    df = pd.DataFrame([payload.model_dump()])
    return _predict_from_df(df)

@app.post("/predict/extract")
def predict_from_text(farmer_text: str):
    """Accept free-text farmer description, extract structured params,
    validate, then predict.

    Returns a dict with both the extraction result and the prediction.
    """
    from nlp.extract import extract_farmer_input
    extracted = extract_farmer_input(farmer_text)

    if extracted.get("missing_fields"):
        return {
            "status": "incomplete",
            "missing_fields": extracted["missing_fields"],
            "extracted_so_far": extracted["structured"],
            "follow_up_question": _build_followup(extracted["missing_fields"]),
        }

    df = pd.DataFrame([extracted["structured"]])
    prediction = _predict_from_df(df)
    explanation = _explain_prediction(prediction, extracted["structured"])

    return {
        "status": "complete",
        "extracted": extracted["structured"],
        "prediction": prediction,
        "explanation": explanation,
    }

def _build_followup(missing: list[str]) -> str:
    """Generate a natural-language follow-up question for missing fields."""
    field_labels = {
        "pond_area_ha": "pond area in hectares",
        "stocking_count": "number of fingerlings stocked",
        "initial_weight_g": "initial weight of the fingerlings in grams",
        "culture_days": "how many days you plan to grow them",
        "mean_temperature_c": "expected average water temperature",
        "season": "season (summer, winter, or monsoon)",
        "intensity": "farming intensity (extensive, semi-intensive, or intensive)",
        "feed_protein_pct": "feed protein percentage",
        "mean_do_mg_l": "expected average dissolved oxygen",
        "min_do_mg_l": "expected minimum dissolved oxygen",
        "mean_ph": "expected average pH",
        "min_ph": "expected minimum pH",
        "max_temp_c": "expected maximum temperature",
        "min_temp_c": "expected minimum temperature",
    }
    labels = [field_labels.get(f, f) for f in missing]
    if len(labels) == 1:
        return f"Could you also tell me the {labels[0]}?"
    return "Could you also tell me: " + ", ".join(labels[:-1]) + f", and {labels[-1]}?"

def _explain_prediction(pred: dict, params: dict) -> str:
    """Simple rule-based explanation (placeholder for LLM explanation layer)."""
    factors = pred["top_factors"]
    top = factors[0] if factors else {"feature": "stocking density", "impact_kg": 0}
    direction = "increases" if top.get("impact_kg", 0) > 0 else "reduces"
    return (
        f"Based on your inputs, the model expects a harvest of approximately "
        f"{pred['point_estimate_kg']:.0f} kg "
        f"(range: {pred['lower_bound_kg']:.0f}–{pred['upper_bound_kg']:.0f} kg). "
        f"The biggest driver of this forecast is **{top['feature']}**, which "
        f"{direction} the estimate by about {abs(top.get('impact_kg', 0)):.0f} kg. "
        f"Water quality and stocking density are the next most important factors."
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
