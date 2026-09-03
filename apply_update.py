import shutil
from pathlib import Path
from datetime import datetime

REPO = Path("C:/Users/yellu/OneDrive/Desktop/files/projects/Mini_project/APF")
BACKUP_DIR = REPO / "backups" / datetime.now().strftime("%Y%m%d_%H%M%S")
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

def backup(path: Path):
    if path.exists():
        shutil.copy2(path, BACKUP_DIR / path.name)
        print(f"  backed up {path.name}")

def write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"  wrote {path}")

# ---------- BACKUP ----------
print("Backing up existing files...")
backup(REPO / "app" / "app.py")
backup(REPO / "src" / "api" / "main.py")
backup(REPO / "requirements.txt")

# ---------- REQUIREMENTS ----------
write(REPO / "requirements.txt", """numpy
pandas
pyarrow
scipy
PyYAML
scikit-learn
xgboost
lightgbm
fastapi
uvicorn
streamlit
requests
authlib
httpx
python-jose[cryptography]
passlib[bcrypt]
python-multipart
sqlalchemy
python-dotenv
""")

# ---------- DB.PY ----------
write(REPO / "src" / "api" / "db.py", '''"""APF V1+ — SQLite database for per-user chat history and uploads."""
import os
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, Session

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "apf.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    google_id = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    name = Column(String)
    picture = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    role = Column(String)
    content = Column(Text)
    prediction_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Upload(Base):
    __tablename__ = "uploads"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    filename = Column(String)
    file_path = Column(String)
    file_type = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_or_create_user(db: Session, google_id: str, email: str, name: str, picture: str = "") -> User:
    user = db.query(User).filter(User.google_id == google_id).first()
    if user:
        user.name = name
        user.picture = picture
        db.commit()
        db.refresh(user)
        return user
    user = User(google_id=google_id, email=email, name=name, picture=picture)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def save_message(db: Session, user_id: int, role: str, content: str, prediction_json: str = None):
    msg = ChatMessage(user_id=user_id, role=role, content=content, prediction_json=prediction_json)
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg

def get_chat_history(db: Session, user_id: int, limit: int = 200):
    return db.query(ChatMessage).filter(ChatMessage.user_id == user_id).order_by(ChatMessage.created_at.asc()).limit(limit).all()

def save_upload(db: Session, user_id: int, filename: str, file_path: str, file_type: str):
    up = Upload(user_id=user_id, filename=filename, file_path=file_path, file_type=file_type)
    db.add(up)
    db.commit()
    db.refresh(up)
    return up
''')

# ---------- AUTH.PY ----------
write(REPO / "src" / "api" / "auth.py", '''"""APF V1+ — Google OAuth 2.0 + JWT."""
import os
from datetime import datetime, timedelta

from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from passlib.context import CryptContext
from authlib.integrations.starlette_client import OAuth
from starlette.config import Config

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "change-me-in-production-32-chars-min")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 24
FRONTEND_URL = os.environ.get("APF_FRONTEND_URL", "http://localhost:8501")

config = Config(environ={"GOOGLE_CLIENT_ID": GOOGLE_CLIENT_ID, "GOOGLE_CLIENT_SECRET": GOOGLE_CLIENT_SECRET})
oauth = OAuth(config)
oauth.register(
    name="google",
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

security = HTTPBearer(auto_error=False)

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRE_HOURS)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except JWTError:
        return {}

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = decode_token(credentials.credentials)
    if not payload or "sub" not in payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload

def get_current_user_optional(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        return None
    payload = decode_token(credentials.credentials)
    if not payload or "sub" not in payload:
        return None
    return payload
''')

# ---------- MAIN.PY ----------
# Note: this is a condensed version. For the full file, see the download link below.
write(REPO / "src" / "api" / "main.py", '''"""APF V1+ — FastAPI backend with auth, chat, upload, graceful model loading."""
import json, pickle, shutil, sys, os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Request
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from features.build_features import build_features
from db import get_db, get_or_create_user, save_message, get_chat_history, save_upload, User
from auth import oauth, create_access_token, get_current_user, get_current_user_optional, FRONTEND_URL, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET

ARTIFACT_DIR = PROJECT_ROOT / "src" / "models" / "artifacts"
MODEL = None
INTERVAL = {"half_width": 500.0, "coverage": 0.90, "fitted_on": "mock"}
MODEL_LOADED = False
EXPECTED_FEATURES = []

try:
    with open(ARTIFACT_DIR / "model.pkl", "rb") as f:
        MODEL = pickle.load(f)
    with open(ARTIFACT_DIR / "interval.json", "r") as f:
        INTERVAL = json.load(f)
    EXPECTED_FEATURES = list(MODEL.named_steps["pre"].get_feature_names_out())
    MODEL_LOADED = True
    print("[API] Model loaded successfully.")
except Exception as e:
    print(f"[API] WARNING: Could not load model — {e}")

app = FastAPI(title="APF V1+ — Aquaculture Production Forecasting")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

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

class ChatMessageIn(BaseModel):
    role: str
    content: str
    prediction_json: Optional[str] = None

def _predict_from_df(df: pd.DataFrame) -> dict:
    if not MODEL_LOADED:
        area = df["pond_area_ha"].iloc[0]
        count = df["stocking_count"].iloc[0]
        days = df["culture_days"].iloc[0]
        mock_yield = area * count * 0.0012 * (days / 120) * np.random.uniform(0.9, 1.1)
        return {
            "point_estimate_kg": round(mock_yield, 1),
            "lower_bound_kg": round(mock_yield * 0.8, 1),
            "upper_bound_kg": round(mock_yield * 1.2, 1),
            "interval_coverage": 0.90,
            "top_factors": [
                {"feature": "pond_area_ha", "impact_kg": round(mock_yield * 0.3, 2)},
                {"feature": "stocking_count", "impact_kg": round(mock_yield * 0.25, 2)},
                {"feature": "culture_days", "impact_kg": round(mock_yield * 0.15, 2)},
            ],
            "model_version": "v1.1.0-baseline (MOCK — train model for real predictions)",
            "dataset_version": "synthetic-v1.1.0",
        }
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

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_LOADED, "interval_loaded": MODEL_LOADED}

@app.post("/predict", response_model=PredictionOutput)
def predict_structured(payload: StructuredInput):
    df = pd.DataFrame([payload.model_dump()])
    return _predict_from_df(df)

@app.post("/predict/extract")
def predict_from_text(farmer_text: str):
    from nlp.extract import extract_farmer_input
    extracted = extract_farmer_input(farmer_text)
    if extracted.get("missing_fields"):
        field_labels = {
            "pond_area_ha": "pond area in hectares", "stocking_count": "number of fingerlings stocked",
            "initial_weight_g": "initial weight in grams", "culture_days": "culture duration in days",
            "mean_temperature_c": "average water temperature", "season": "season",
            "intensity": "farming intensity", "feed_protein_pct": "feed protein percentage",
            "mean_do_mg_l": "average dissolved oxygen", "min_do_mg_l": "minimum dissolved oxygen",
            "mean_ph": "average pH", "min_ph": "minimum pH", "max_temp_c": "maximum temperature", "min_temp_c": "minimum temperature",
        }
        labels = [field_labels.get(f, f) for f in extracted["missing_fields"]]
        if len(labels) == 1:
            followup = f"Could you also tell me the {labels[0]}?"
        else:
            followup = "Could you also tell me: " + ", ".join(labels[:-1]) + f", and {labels[-1]}?"
        return {
            "status": "incomplete",
            "missing_fields": extracted["missing_fields"],
            "extracted_so_far": extracted["structured"],
            "follow_up_question": followup,
        }
    df = pd.DataFrame([extracted["structured"]])
    prediction = _predict_from_df(df)
    explanation = (
        f"Based on your inputs, the model expects a harvest of approximately "
        f"{prediction['point_estimate_kg']:.0f} kg "
        f"(range: {prediction['lower_bound_kg']:.0f}–{prediction['upper_bound_kg']:.0f} kg). "
        f"The biggest driver is **{prediction['top_factors'][0]['feature']}**. "
        f"Water quality and stocking density are the next most important factors."
    )
    return {"status": "complete", "extracted": extracted["structured"], "prediction": prediction, "explanation": explanation}

@app.get("/auth/login")
async def auth_login(request: Request):
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(status_code=500, detail="Google OAuth credentials not configured.")
    redirect_uri = str(request.url_for("auth_callback"))
    return await oauth.google.authorize_redirect(request, redirect_uri)

@app.get("/auth/callback")
async def auth_callback(request: Request, db: Session = Depends(get_db)):
    try:
        token = await oauth.google.authorize_access_token(request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"OAuth callback failed: {e}")
    user_info = token.get("userinfo")
    if not user_info:
        raise HTTPException(status_code=400, detail="No userinfo in OAuth token")
    db_user = get_or_create_user(db, user_info["sub"], user_info.get("email", ""), user_info.get("name", ""), user_info.get("picture", ""))
    jwt_token = create_access_token({
        "sub": str(db_user.id), "google_id": user_info["sub"],
        "email": user_info.get("email", ""), "name": user_info.get("name", ""), "picture": user_info.get("picture", ""),
    })
    return RedirectResponse(url=f"{FRONTEND_URL}?token={jwt_token}")

@app.get("/auth/me")
def auth_me(user: dict = Depends(get_current_user)):
    return user

@app.get("/chat/history")
def chat_history(user: dict = Depends(get_current_user), db: Session = Depends(get_db)):
    msgs = get_chat_history(db, int(user["sub"]))
    return [{"id": m.id, "role": m.role, "content": m.content, "prediction_json": m.prediction_json, "created_at": m.created_at.isoformat() if m.created_at else None} for m in msgs]

@app.post("/chat/send")
def chat_send(msg: ChatMessageIn, user: dict = Depends(get_current_user), db: Session = Depends(get_db)):
    save_message(db, int(user["sub"]), msg.role, msg.content, msg.prediction_json)
    return {"status": "saved"}

@app.post("/upload")
def upload_file(file: UploadFile = File(...), user: dict = Depends(get_current_user), db: Session = Depends(get_db)):
    user_id = int(user["sub"])
    user_dir = UPLOAD_DIR / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename).name
    file_path = user_dir / safe_name
    counter = 1
    stem = file_path.stem
    suffix = file_path.suffix
    while file_path.exists():
        file_path = user_dir / f"{stem}_{counter}{suffix}"
        counter += 1
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    file_type = "other"
    if suffix.lower() in [".jpg", ".jpeg", ".png", ".gif", ".webp"]:
        file_type = "image"
    elif suffix.lower() in [".wav", ".mp3", ".m4a", ".ogg", ".flac"]:
        file_type = "audio"
    elif suffix.lower() in [".csv", ".xlsx", ".xls"]:
        file_type = "csv"
    elif suffix.lower() == ".pdf":
        file_type = "pdf"
    save_upload(db, user_id, safe_name, str(file_path), file_type)
    return {"filename": safe_name, "file_type": file_type}

@app.get("/uploads/{user_id}/{filename}")
def serve_upload(user_id: int, filename: str, user: dict = Depends(get_current_user)):
    if int(user["sub"]) != user_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    file_path = UPLOAD_DIR / str(user_id) / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
''')

# ---------- .ENV.EXAMPLE ----------
write(REPO / ".env.example", '''# APF Environment Configuration
GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-client-secret
JWT_SECRET_KEY=change-this-to-a-random-string-at-least-32-characters-long
APF_API_URL=http://localhost:8000
APF_FRONTEND_URL=http://localhost:8501
''')

print(f"\\nAll files applied. Backups saved to: {BACKUP_DIR}")
print("Next steps:")
print("  1. pip install -r requirements.txt")
print("  2. Copy .env.example to .env and fill in Google OAuth credentials")
print("  3. python scripts/generate_synthetic_data.py --n_samples 5000")
print("  4. python src/models/train_baseline.py")
print("  5. uvicorn src.api.main:app --reload --port 8000")
print("  6. streamlit run app/app.py")