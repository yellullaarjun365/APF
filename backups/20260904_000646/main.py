"""APF V1 -- FastAPI prediction endpoint with auth, chat history & uploads.

Run:  uvicorn src.api.main:app --reload --port 8000
"""
import json
import os
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features.build_features import build_features

from .auth import (
    ACCESS_TOKEN_EXPIRE_DAYS,
    FRONTEND_URL,
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET,
    create_access_token,
    decode_token,
    get_current_user,
    oauth,
    require_user,
)
from .db import ChatMessage, Upload, User, get_db, init_db

# ------------------------------------------------------------------
# Init DB
# ------------------------------------------------------------------
init_db()

# ------------------------------------------------------------------
# Load model artifacts
# ------------------------------------------------------------------
ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "models" / "artifacts"

with open(ARTIFACT_DIR / "model.pkl", "rb") as f:
    MODEL = pickle.load(f)

with open(ARTIFACT_DIR / "interval.json", "r") as f:
    INTERVAL = json.load(f)

EXPECTED_FEATURES = list(MODEL.named_steps["pre"].get_feature_names_out())

UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------
app = FastAPI(title="APF V1 -- Aquaculture Production Forecasting")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your Streamlit URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Pydantic schemas
# ------------------------------------------------------------------
class StructuredInput(BaseModel):
    pond_area_ha: float = Field(..., gt=0)
    stocking_count: int = Field(..., gt=0)
    initial_weight_g: float = Field(..., gt=0)
    culture_days: int = Field(..., gt=0)
    mean_temperature_c: float = Field(...)
    season: str = Field(..., pattern="^(summer|winter|monsoon)$")
    intensity: str = Field(..., pattern="^(extensive|semi-intensive|intensive)$")
    feed_protein_pct: int = Field(..., ge=20, le=50)
    mean_do_mg_l: float = Field(..., gt=0)
    min_do_mg_l: float = Field(..., gt=0)
    mean_ph: float = Field(..., gt=4, lt=11)
    min_ph: float = Field(..., gt=4, lt=11)
    max_temp_c: float = Field(...)
    min_temp_c: float = Field(...)

class PredictionOutput(BaseModel):
    point_estimate_kg: float
    lower_bound_kg: float
    upper_bound_kg: float
    interval_coverage: float
    top_factors: list[dict]
    model_version: str
    dataset_version: str

class ChatMessageOut(BaseModel):
    id: int
    role: str
    content: str
    prediction_json: Optional[str]
    created_at: str

class ChatMessageIn(BaseModel):
    role: str
    content: str
    prediction_json: Optional[str] = None

# ------------------------------------------------------------------
# Prediction helpers (unchanged logic)
# ------------------------------------------------------------------
def _predict_from_df(df: pd.DataFrame) -> dict:
    X, _ = build_features(df)
    pred = float(MODEL.predict(X)[0])
    hw = INTERVAL["half_width"]
    base_pred = pred
    importances = []
    for col in X.columns:
        if col in ["season", "intensity"]:
            continue
        X_pert = X.copy()
        X_pert[col] = X_pert[col].median()
        pert_pred = float(MODEL.predict(X_pert)[0])
        importances.append({"feature": col, "impact_kg": round(base_pred - pert_pred, 2)})
    importances.sort(key=lambda x: abs(x["impact_kg"]), reverse=True)
    return {
        "point_estimate_kg": round(pred, 1),
        "lower_bound_kg": round(max(pred - hw, 0), 1),
        "upper_bound_kg": round(pred + hw, 1),
        "interval_coverage": INTERVAL["coverage"],
        "top_factors": importances[:5],
        "model_version": "v1.1.0-baseline",
        "dataset_version": INTERVAL.get("fitted_on", "unknown"),
    }

def _build_followup(missing: list[str]) -> str:
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
    factors = pred["top_factors"]
    top = factors[0]
    direction = "increases" if top["impact_kg"] > 0 else "reduces"
    return (
        f"Based on your inputs, the model expects a harvest of approximately "
        f"{pred['point_estimate_kg']:.0f} kg "
        f"(range: {pred['lower_bound_kg']:.0f}–{pred['upper_bound_kg']:.0f} kg). "
        f"The biggest driver of this forecast is **{top['feature']}**, which "
        f"{direction} the estimate by about {abs(top['impact_kg']):.0f} kg. "
        f"Water quality and stocking density are the next most important factors."
    )

# ------------------------------------------------------------------
# Auth endpoints
# ------------------------------------------------------------------
@app.get("/auth/google")
async def login_google(request: Request):
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(status_code=500, detail="Google OAuth not configured")
    redirect_uri = request.url_for("auth_callback")
    return await oauth.google.authorize_redirect(request, str(redirect_uri))

@app.get("/auth/callback")
async def auth_callback(request: Request, db: Session = Depends(get_db)):
    try:
        token = await oauth.google.authorize_access_token(request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"OAuth failed: {e}")
    user_info = token.get("userinfo")
    if not user_info:
        raise HTTPException(status_code=400, detail="No user info from Google")

    google_id = user_info["sub"]
    email = user_info["email"]
    name = user_info.get("name", email.split("@")[0])
    picture = user_info.get("picture", "")

    user = db.query(User).filter(User.google_id == google_id).first()
    if not user:
        user = User(google_id=google_id, email=email, name=name, picture=picture)
        db.add(user)
        db.commit()
        db.refresh(user)
    else:
        user.name = name
        user.picture = picture
        db.commit()

    access_token = create_access_token({"sub": str(user.id)})
    # Redirect back to Streamlit with token
    redirect_url = f"{FRONTEND_URL}/?token={access_token}"
    return RedirectResponse(url=redirect_url)

@app.get("/auth/me")
async def auth_me(user: Optional[User] = Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "picture": user.picture,
    }

# ------------------------------------------------------------------
# Prediction endpoints (public — auth optional)
# ------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "interval_loaded": True}

@app.post("/predict", response_model=PredictionOutput)
def predict_structured(payload: StructuredInput, user: Optional[User] = Depends(get_current_user), db: Session = Depends(get_db)):
    df = pd.DataFrame([payload.model_dump()])
    result = _predict_from_df(df)
    # If logged in, save to chat history as an assistant message
    if user:
        msg = ChatMessage(
            user_id=user.id,
            role="assistant",
            content=_explain_prediction(result, payload.model_dump()),
            prediction_json=json.dumps(result),
        )
        db.add(msg)
        db.commit()
    return result

@app.post("/predict/extract")
def predict_from_text(farmer_text: str, user: Optional[User] = Depends(get_current_user), db: Session = Depends(get_db)):
    from nlp.extract import extract_farmer_input
    extracted = extract_farmer_input(farmer_text)

    # Save user message
    if user:
        db.add(ChatMessage(user_id=user.id, role="user", content=farmer_text))
        db.commit()

    if extracted.get("missing_fields"):
        missing = extracted["missing_fields"]
        followup = _build_followup(missing)
        resp = {"status": "incomplete", "missing_fields": missing, "extracted_so_far": extracted["structured"], "follow_up_question": followup}
        if user:
            db.add(ChatMessage(user_id=user.id, role="assistant", content=followup))
            db.commit()
        return resp

    df = pd.DataFrame([extracted["structured"]])
    prediction = _predict_from_df(df)
    explanation = _explain_prediction(prediction, extracted["structured"])

    resp = {
        "status": "complete",
        "extracted": extracted["structured"],
        "prediction": prediction,
        "explanation": explanation,
    }
    if user:
        db.add(ChatMessage(user_id=user.id, role="assistant", content=explanation, prediction_json=json.dumps(prediction)))
        db.commit()
    return resp

# ------------------------------------------------------------------
# Chat history endpoints (auth required)
# ------------------------------------------------------------------
@app.get("/chat/history", response_model=list[ChatMessageOut])
def get_chat_history(user: User = Depends(require_user), db: Session = Depends(get_db)):
    msgs = db.query(ChatMessage).filter(ChatMessage.user_id == user.id).order_by(ChatMessage.created_at.asc()).all()
    return [
        {
            "id": m.id,
            "role": m.role,
            "content": m.content,
            "prediction_json": m.prediction_json,
            "created_at": m.created_at.isoformat() if m.created_at else "",
        }
        for m in msgs
    ]

@app.post("/chat/message")
def add_chat_message(msg: ChatMessageIn, user: User = Depends(require_user), db: Session = Depends(get_db)):
    db_msg = ChatMessage(user_id=user.id, role=msg.role, content=msg.content, prediction_json=msg.prediction_json)
    db.add(db_msg)
    db.commit()
    db.refresh(db_msg)
    return {"id": db_msg.id, "status": "saved"}

@app.delete("/chat/history")
def clear_chat_history(user: User = Depends(require_user), db: Session = Depends(get_db)):
    db.query(ChatMessage).filter(ChatMessage.user_id == user.id).delete()
    db.commit()
    return {"status": "cleared"}

# ------------------------------------------------------------------
# File upload endpoints (auth required)
# ------------------------------------------------------------------
@app.post("/upload")
def upload_file(
    file: UploadFile = File(...),
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    user_dir = UPLOAD_DIR / str(user.id)
    user_dir.mkdir(parents=True, exist_ok=True)
    file_path = user_dir / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    upload = Upload(
        user_id=user.id,
        filename=file.filename,
        file_path=str(file_path),
        file_type=file.content_type or "unknown",
    )
    db.add(upload)
    db.commit()
    db.refresh(upload)
    return {"id": upload.id, "filename": upload.filename, "file_type": upload.file_type}

@app.get("/uploads")
def list_uploads(user: User = Depends(require_user), db: Session = Depends(get_db)):
    uploads = db.query(Upload).filter(Upload.user_id == user.id).order_by(Upload.created_at.desc()).all()
    return [
        {"id": u.id, "filename": u.filename, "file_type": u.file_type, "created_at": u.created_at.isoformat() if u.created_at else ""}
        for u in uploads
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
