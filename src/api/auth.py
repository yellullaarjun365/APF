"""APF V1+ — Google OAuth 2.0 + JWT."""
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
