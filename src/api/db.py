"""APF V1+ — SQLite database for per-user chat history and uploads."""
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
