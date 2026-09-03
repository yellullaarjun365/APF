"""APF V1 -- Streamlit Web UI with Google Auth, per-user chat, voice & upload.

Run:  streamlit run app/app.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

API_BASE = os.environ.get("APF_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="AquaPredict AI – Smarter Aquaculture",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================================================================
# DARK THEME CSS
# ==================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    [data-testid="stAppViewContainer"] { background: #0f1117 !important; }
    .block-container { padding: 0 2rem 2rem 2rem !important; max-width: 1000px; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #16181f !important;
        border-right: 1px solid #23262f;
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1rem !important; }

    .nav-active {
        background: linear-gradient(135deg, #0d9488, #0f766e);
        color: #ffffff !important;
        border-radius: 10px;
        padding: 10px 14px;
        font-weight: 600;
        font-size: 14px;
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 6px;
        box-shadow: 0 2px 8px rgba(13, 148, 136, 0.25);
    }
    .nav-item {
        color: #94a3b8;
        padding: 10px 14px;
        border-radius: 10px;
        font-size: 14px;
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 6px;
        cursor: pointer;
        transition: all 0.15s ease;
    }
    .nav-item:hover { background: #1e212b; color: #e2e8f0; }

    /* Chat messages */
    .msg-assistant {
        display: flex;
        gap: 12px;
        margin-bottom: 24px;
        align-items: flex-start;
    }
    .msg-assistant-avatar {
        width: 28px; height: 28px;
        background: linear-gradient(135deg, #0d9488, #0f766e);
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        color: white; font-size: 13px; flex-shrink: 0; margin-top: 4px;
    }
    .msg-assistant-body {
        background: transparent; color: #e2e8f0;
        font-size: 15px; line-height: 1.7;
        max-width: 85%; white-space: pre-wrap;
    }
    .msg-assistant-body strong { color: #ffffff; font-weight: 600; }
    .msg-assistant-body ul { margin: 8px 0; padding-left: 20px; }
    .msg-assistant-body li { margin: 4px 0; }

    .msg-user {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 24px;
    }
    .msg-user-body {
        background: #1e212b;
        border: 1px solid #2a2d3a;
        border-radius: 18px 18px 4px 18px;
        padding: 12px 18px;
        color: #e2e8f0;
        font-size: 15px; line-height: 1.6;
        max-width: 80%; white-space: pre-wrap;
    }

    /* Thinking indicator */
    .thinking {
        display: flex; align-items: center; gap: 8px;
        color: #64748b; font-size: 14px; padding: 8px 0;
    }
    .thinking-dots { display: flex; gap: 4px; }
    .thinking-dots span {
        width: 6px; height: 6px;
        background: #0d9488; border-radius: 50%;
        animation: pulse 1.4s infinite ease-in-out both;
    }
    .thinking-dots span:nth-child(1) { animation-delay: -0.32s; }
    .thinking-dots span:nth-child(2) { animation-delay: -0.16s; }
    @keyframes pulse {
        0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
        40% { transform: scale(1); opacity: 1; }
    }

    /* Forecast inline card */
    .forecast-inline {
        background: #16181f;
        border: 1px solid #23262f;
        border-radius: 14px;
        padding: 20px;
        margin-top: 12px;
        max-width: 420px;
    }
    .forecast-inline .value { font-size: 28px; font-weight: 700; color: #0d9488; }
    .forecast-inline .label { font-size: 12px; color: #64748b; margin-bottom: 4px; }
    .forecast-inline .range { font-size: 13px; color: #94a3b8; margin-top: 4px; }
    .forecast-inline .factors { margin-top: 12px; font-size: 13px; color: #94a3b8; }
    .forecast-inline .factors li { margin: 3px 0; }

    /* Input bar */
    .input-bar-wrapper {
        position: sticky;
        bottom: 0;
        background: #0f1117;
        padding: 16px 0 8px 0;
        border-top: 1px solid #23262f;
        margin-top: 20px;
    }

    /* Welcome suggestions */
    .suggestion-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        margin: 20px 0;
    }
    .suggestion-card {
        background: #16181f;
        border: 1px solid #23262f;
        border-radius: 12px;
        padding: 16px;
        cursor: pointer;
        transition: all 0.15s ease;
    }
    .suggestion-card:hover { border-color: #0d9488; background: #1a1d26; }
    .suggestion-card .icon { font-size: 20px; margin-bottom: 8px; }
    .suggestion-card .title { font-size: 14px; font-weight: 600; color: #e2e8f0; margin-bottom: 4px; }
    .suggestion-card .desc { font-size: 12px; color: #64748b; line-height: 1.4; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0f1117; }
    ::-webkit-scrollbar-thumb { background: #2a2d3a; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #3a3d4a; }

    /* Auth card */
    .auth-card {
        background: #16181f;
        border: 1px solid #23262f;
        border-radius: 16px;
        padding: 40px;
        text-align: center;
        max-width: 400px;
        margin: 80px auto;
    }
    .auth-card h2 { color: #e2e8f0; margin-bottom: 8px; }
    .auth-card p { color: #64748b; font-size: 14px; margin-bottom: 24px; }
    .google-btn {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        background: #ffffff;
        color: #3c4043;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 500;
        font-size: 14px;
        text-decoration: none;
        transition: all 0.15s;
    }
    .google-btn:hover { background: #f1f3f4; }

    /* File upload compact */
    div[data-testid="stFileUploader"] > section {
        background: #1e212b !important;
        border: 1px dashed #2a2d3a !important;
        border-radius: 10px !important;
        padding: 8px 12px !important;
    }
    div[data-testid="stFileUploader"] > section:hover {
        border-color: #0d9488 !important;
    }

    /* Section labels */
    .section-label {
        font-size: 10px; font-weight: 700; color: #475569;
        text-transform: uppercase; letter-spacing: 0.8px;
        margin: 24px 0 10px 0;
    }

    /* User profile in sidebar */
    .user-pill {
        display: flex; align-items: center; gap: 10px;
        padding: 10px 12px;
        background: #1e212b;
        border-radius: 10px;
        border: 1px solid #2a2d3a;
    }
    .user-pill img {
        width: 32px; height: 32px; border-radius: 50%;
    }
</style>
""", unsafe_allow_html=True)

# ==================================================================
# Session state init
# ==================================================================
def init_state():
    defaults = {
        "page": "Chat Assistant",
        "chat_history": [],
        "last_prediction": None,
        "pond_params": {
            "pond_area_ha": 0.5, "stocking_count": 3000, "initial_weight_g": 15.0,
            "culture_days": 120, "mean_temperature_c": 28.0, "season": "summer",
            "intensity": "semi-intensive", "feed_protein_pct": 30,
            "mean_do_mg_l": 7.5, "min_do_mg_l": 5.0, "mean_ph": 7.5,
            "min_ph": 6.8, "max_temp_c": 32.0, "min_temp_c": 24.0,
        },
        "analyzing": False,
        "show_suggestions": True,
        "jwt_token": None,
        "user": None,
        "voice_triggered": False,
        "uploaded_file_info": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ==================================================================
# Auth: Check for token in query params (OAuth callback)
# ==================================================================
if "token" in st.query_params and not st.session_state.jwt_token:
    token = st.query_params["token"]
    st.session_state.jwt_token = token
    # Clear token from URL
    new_params = {k: v for k, v in st.query_params.items() if k != "token"}
    st.query_params.clear()
    for k, v in new_params.items():
        st.query_params[k] = v
    st.rerun()

# ==================================================================
# API helpers
# ==================================================================
def _headers():
    h = {"Content-Type": "application/json"}
    if st.session_state.jwt_token:
        h["Authorization"] = f"Bearer {st.session_state.jwt_token}"
    return h

def _api_get(path: str):
    try:
        r = requests.get(f"{API_BASE}{path}", headers=_headers(), timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def _api_post(path: str, payload: dict):
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, headers=_headers(), timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def _api_upload(file_bytes, filename: str):
    try:
        files = {"file": (filename, file_bytes)}
        headers = {}
        if st.session_state.jwt_token:
            headers["Authorization"] = f"Bearer {st.session_state.jwt_token}"
        r = requests.post(f"{API_BASE}/upload", files=files, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def _api_predict_structured(params: dict) -> dict:
    try:
        r = requests.post(f"{API_BASE}/predict", json=params, headers=_headers(), timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def _api_predict_text(text: str) -> dict:
    try:
        r = requests.post(
            f"{API_BASE}/predict/extract",
            params={"farmer_text": text},
            headers=_headers(),
            timeout=20,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def _api_health() -> dict:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}

def _fetch_user():
    if st.session_state.jwt_token and not st.session_state.user:
        data = _api_get("/auth/me")
        if "error" not in data:
            st.session_state.user = data

def _fetch_chat_history():
    if st.session_state.jwt_token and not st.session_state.chat_history:
        data = _api_get("/chat/history")
        if "error" not in data and isinstance(data, list):
            st.session_state.chat_history = [
                {
                    "role": m["role"],
                    "content": m["content"],
                    "prediction": json.loads(m["prediction_json"]) if m.get("prediction_json") else None,
                    "time": m["created_at"][:16] if m.get("created_at") else "",
                }
                for m in data
            ]
            if st.session_state.chat_history:
                st.session_state.show_suggestions = False

_fetch_user()
_fetch_chat_history()

# ==================================================================
# Sidebar
# ==================================================================
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;padding:8px 0 20px 0;">
        <div style="width:36px;height:36px;background:linear-gradient(135deg,#0d9488,#0f766e);border-radius:50%;display:flex;align-items:center;justify-content:center;color:white;font-size:18px;">🐟</div>
        <div>
            <div style="font-size:16px;font-weight:700;color:#e2e8f0;">AquaPredict AI</div>
            <div style="font-size:12px;color:#64748b;">Smarter Aquaculture</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Main</div>', unsafe_allow_html=True)

    nav_items = [
        ("💬", "Chat Assistant", "Chat Assistant"),
        ("📊", "Production Forecast", "Production Forecast"),
    ]
    for icon, label, page_name in nav_items:
        active = st.session_state.page == page_name
        if active:
            st.markdown(f'<div class="nav-active">{icon} {label}</div>', unsafe_allow_html=True)
        else:
            if st.button(f"{icon} {label}", key=f"nav_{page_name}", use_container_width=True):
                st.session_state.page = page_name
                st.rerun()

    st.markdown('<div class="section-label">Account</div>', unsafe_allow_html=True)

    if st.session_state.user:
        user = st.session_state.user
        picture = user.get("picture", "")
        name = user.get("name", "User")
        email = user.get("email", "")
        img_html = f'<img src="{picture}" style="width:32px;height:32px;border-radius:50%;" onerror="this.style.display=\'none\'" />' if picture else '<div style="width:32px;height:32px;background:#0d9488;border-radius:50%;display:flex;align-items:center;justify-content:center;color:white;font-size:14px;font-weight:600;">' + name[0].upper() + '</div>'
        st.markdown(f"""
        <div class="user-pill">
            {img_html}
            <div>
                <div style="font-size:13px;font-weight:600;color:#e2e8f0;">{name}</div>
                <div style="font-size:11px;color:#64748b;">{email}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚪 Logout", use_container_width=True, key="btn_logout"):
            st.session_state.jwt_token = None
            st.session_state.user = None
            st.session_state.chat_history = []
            st.session_state.show_suggestions = True
            st.rerun()
    else:
        st.markdown("""
        <div style="padding:12px;background:#1e212b;border-radius:10px;border:1px solid #2a2d3a;margin-bottom:12px;">
            <div style="font-size:13px;color:#94a3b8;margin-bottom:8px;">Sign in to save your chat history and uploads.</div>
            <a href="{API_BASE}/auth/google" target="_self" class="google-btn" style="display:inline-flex;align-items:center;gap:8px;background:#fff;color:#3c4043;padding:8px 16px;border-radius:6px;text-decoration:none;font-size:13px;font-weight:500;">
                <svg width="18" height="18" viewBox="0 0 24 24"><path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/><path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/><path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/><path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/></svg>
                Sign in with Google
            </a>
        </div>
        """.replace("{API_BASE}", API_BASE), unsafe_allow_html=True)

    st.markdown('<div class="section-label">Settings</div>', unsafe_allow_html=True)
    if st.button("⚙️ Settings", key="nav_settings", use_container_width=True):
        st.session_state.page = "Settings"
        st.rerun()

    # Backend status
    health = _api_health()
    backend_ok = health.get("status") == "ok"
    status_color = "#34d399" if backend_ok else "#f87171"
    status_text = "Online" if backend_ok else "Offline"
    st.markdown(f"""
    <div style="margin-top:24px;">
        <div style="display:flex;align-items:center;gap:10px;padding:12px;background:#1e212b;border-radius:10px;border:1px solid #2a2d3a;">
            <div style="width:10px;height:10px;background:{status_color};border-radius:50%;box-shadow:0 0 6px {status_color};"></div>
            <div>
                <div style="font-size:12px;font-weight:600;color:#e2e8f0;">API Server</div>
                <div style="font-size:11px;color:#64748b;">{status_text} • {API_BASE}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==================================================================
# Helper: Format assistant response
# ==================================================================
def format_assistant_response(text: str, prediction: dict = None) -> str:
    html = f'<div class="msg-assistant-body">{text}</div>'
    if prediction:
        pe = prediction.get("point_estimate_kg", 0)
        lb = prediction.get("lower_bound_kg", 0)
        ub = prediction.get("upper_bound_kg", 0)
        factors = prediction.get("top_factors", [])
        factors_html = ""
        for f in factors[:4]:
            arrow = "↑" if f["impact_kg"] > 0 else "↓"
            factors_html += f"<li>{f['feature'].replace('_', ' ').title()}: {arrow}</li>"
        html += f"""
        <div class="forecast-inline">
            <div class="label">Estimated Production</div>
            <div class="value">{pe:.1f} <span style="font-size:14px;font-weight:400;color:#64748b;">kg</span></div>
            <div class="range">Range: {lb:.0f} – {ub:.0f} kg (90% CI)</div>
            <div class="factors">
                <strong style="color:#e2e8f0;">Key Factors:</strong>
                <ul style="margin-top:4px;">{factors_html}</ul>
            </div>
        </div>
        """
    return html

# ==================================================================
# PAGE: CHAT ASSISTANT (full width, no right panel)
# ==================================================================
def render_chat_assistant():
    st.markdown("""
    <div style="padding:8px 0 4px 0;">
        <div style="font-size:22px;font-weight:700;color:#e2e8f0;">💬 Chat Assistant</div>
        <div style="font-size:14px;color:#64748b;margin-top:4px;">Ask anything about your pond or get a production forecast.</div>
    </div>
    <hr style='margin:12px 0 20px 0;border:none;border-top:1px solid #23262f;'>
    """, unsafe_allow_html=True)

    # Chat messages
    for msg in st.session_state.chat_history:
        if msg["role"] == "assistant":
            pred = msg.get("prediction")
            body_html = format_assistant_response(msg["content"], pred)
            st.markdown(f"""
            <div class="msg-assistant">
                <div class="msg-assistant-avatar">🐟</div>
                <div>{body_html}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="msg-user">
                <div class="msg-user-body">{msg["content"]}</div>
            </div>
            """, unsafe_allow_html=True)

    # Thinking indicator
    if st.session_state.analyzing:
        st.markdown("""
        <div class="msg-assistant">
            <div class="msg-assistant-avatar">🐟</div>
            <div class="thinking">
                <div class="thinking-dots"><span></span><span></span><span></span></div>
                <span>Analyzing your pond data...</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Welcome suggestions
    if st.session_state.show_suggestions and len(st.session_state.chat_history) == 0:
        st.markdown("<div style='font-size:13px;color:#64748b;margin-bottom:12px;'>Try asking:</div>", unsafe_allow_html=True)
        sugg_col1, sugg_col2 = st.columns(2)
        suggestions = [
            ("📊", "Forecast my production", "Get a harvest estimate from your pond data"),
            ("💧", "Check water quality", "Analyze if your DO, pH, and temp are optimal"),
            ("🍽️", "Feeding recommendation", "Get optimal feed rate and protein %"),
            ("📈", "Growth status", "See how your fish should be progressing"),
        ]
        for i, (icon, title, desc) in enumerate(suggestions):
            col = sugg_col1 if i < 2 else sugg_col2
            with col:
                if st.button(f"{icon} {title}", key=f"sugg_{i}", use_container_width=True):
                    _handle_user_message(title, is_suggestion=True)

    # ---- VOICE INPUT: Check for transcribed text from query params ----
    if "voice_text" in st.query_params:
        voice_text = st.query_params["voice_text"]
        # Clear param
        new_params = {k: v for k, v in st.query_params.items() if k != "voice_text"}
        st.query_params.clear()
        for k, v in new_params.items():
            st.query_params[k] = v
        if voice_text and voice_text.strip():
            st.session_state.voice_text = voice_text.strip()
            st.session_state.voice_triggered = True
            st.rerun()

    # ---- HANDLE VOICE TEXT AS MESSAGE ----
    if st.session_state.get("voice_triggered") and st.session_state.get("voice_text"):
        text = st.session_state.voice_text
        st.session_state.voice_triggered = False
        st.session_state.voice_text = ""
        _handle_user_message(text)
        return  # Stop here; rerun will happen inside _handle_user_message

    # ---- FILE UPLOAD ----
    st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Attach a file",
        type=["jpg", "jpeg", "png", "pdf", "csv", "wav", "mp3", "m4a"],
        key="chat_file_uploader",
        label_visibility="collapsed",
    )
    if uploaded is not None:
        # Only process if this is a new upload
        current_name = st.session_state.get("last_uploaded_name")
        if uploaded.name != current_name:
            st.session_state.last_uploaded_name = uploaded.name
            # Send to backend if logged in
            if st.session_state.jwt_token:
                result = _api_upload(uploaded.getvalue(), uploaded.name)
                if "error" not in result:
                    st.session_state.uploaded_file_info = result
                    file_msg = f"📎 Uploaded: {uploaded.name}"
                else:
                    file_msg = f"📎 {uploaded.name} (upload failed: {result['error']})"
            else:
                file_msg = f"📎 {uploaded.name} (sign in to save uploads)"
            st.session_state.chat_history.append({"role": "user", "content": file_msg, "time": ""})
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"I've received your file **{uploaded.name}**. "
                           f"For V1, I can analyze CSV data with pond parameters. "
                           f"Image-based disease detection is coming in V2. "
                           f"If this is a CSV with pond data, I can process it for a forecast.",
                "time": "",
            })
            st.session_state.show_suggestions = False
            st.rerun()

    # ---- INPUT BAR ----
    st.markdown('<div class="input-bar-wrapper">', unsafe_allow_html=True)

    input_row = st.columns([0.5, 5.5, 0.5, 0.5])

    with input_row[0]:
        # Voice button
        if st.button("🎙️", key="btn_voice", help="Click, then speak"):
            st.session_state.voice_triggered = True
            st.components.v1.html("""
            <script>
            (function() {
                if (!('webkitSpeechRecognition' in window)) {
                    alert('Speech recognition not supported. Use Chrome or Edge.');
                    return;
                }
                const rec = new webkitSpeechRecognition();
                rec.lang = 'en-US';
                rec.interimResults = false;
                rec.maxAlternatives = 1;
                rec.start();
                rec.onresult = function(e) {
                    const text = e.results[0][0].transcript;
                    const url = new URL(window.parent.location.href);
                    url.searchParams.set('voice_text', text);
                    window.parent.history.replaceState({}, '', url);
                    window.parent.location.reload();
                };
                rec.onerror = function(e) {
                    alert('Speech error: ' + e.error);
                };
            })();
            </script>
            """, height=0)

    with input_row[1]:
        user_text = st.text_input("", placeholder="Ask AquaPredict AI...", key="chat_input", label_visibility="collapsed")

    with input_row[2]:
        # Placeholder for model selector or attach (future)
        st.markdown("<div style='width:36px;'></div>", unsafe_allow_html=True)

    with input_row[3]:
        send_clicked = st.button("➤", key="btn_send")

    if send_clicked and user_text and user_text.strip():
        _handle_user_message(user_text.strip())
        return

    st.markdown('</div>', unsafe_allow_html=True)

    # ---- HANDLE ANALYSIS ----
    if st.session_state.analyzing:
        last_user_msg = None
        for msg in reversed(st.session_state.chat_history):
            if msg["role"] == "user" and not msg.get("content", "").startswith("📎"):
                last_user_msg = msg["content"]
                break

        if last_user_msg:
            result = _api_predict_text(last_user_msg)
            if result.get("status") == "incomplete":
                missing = result.get("missing_fields", [])
                followup = result.get("follow_up_question", "Could you provide more details?")
                resp = followup + "\n\nMissing: " + ", ".join(missing)
                st.session_state.chat_history.append({"role": "assistant", "content": resp, "time": ""})
            elif result.get("status") == "complete":
                pred_data = result.get("prediction", {})
                st.session_state.last_prediction = pred_data
                st.session_state.pond_params.update(result.get("extracted", {}))
                explanation = result.get("explanation", "")
                pe = pred_data.get("point_estimate_kg", 0)
                lb = pred_data.get("lower_bound_kg", 0)
                ub = pred_data.get("upper_bound_kg", 0)
                resp = "Here is your production forecast based on the pond details you provided."
                if explanation:
                    resp += f"\n\n{explanation}"
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": resp,
                    "time": "",
                    "prediction": pred_data,
                })
            elif "error" in result:
                resp = "Sorry, I encountered an error connecting to the forecasting engine: " + str(result["error"])
                st.session_state.chat_history.append({"role": "assistant", "content": resp, "time": ""})
            else:
                resp = "I'm not sure how to process that. Try describing your pond with area, stocking count, days, temperature, DO, and pH."
                st.session_state.chat_history.append({"role": "assistant", "content": resp, "time": ""})

        st.session_state.analyzing = False
        st.rerun()


def _handle_user_message(text: str, is_suggestion: bool = False):
    """Add user message, trigger analysis, and rerun."""
    st.session_state.chat_history.append({"role": "user", "content": text, "time": ""})
    st.session_state.analyzing = True
    st.session_state.show_suggestions = False
    st.rerun()

# ==================================================================
# PAGE: PRODUCTION FORECAST
# ==================================================================
def render_forecast():
    st.markdown("""
    <div style="padding:8px 0 4px 0;">
        <div style="font-size:22px;font-weight:700;color:#e2e8f0;">📊 Production Forecast</div>
        <div style="font-size:14px;color:#64748b;margin-top:4px;">Enter your pond parameters for a detailed forecast.</div>
    </div>
    <hr style='margin:12px 0 20px 0;border:none;border-top:1px solid #23262f;'>
    """, unsafe_allow_html=True)

    pred = st.session_state.last_prediction
    st.markdown("<div style='background:#16181f;border:1px solid #23262f;border-radius:14px;padding:20px;margin-bottom:20px;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:16px;font-weight:600;color:#e2e8f0;margin-bottom:16px;'>Forecast Parameters</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("Pond Area (ha)", value=st.session_state.pond_params["pond_area_ha"], key="fc_area")
        st.number_input("Stocking Count", value=st.session_state.pond_params["stocking_count"], key="fc_count")
        st.number_input("Initial Weight (g)", value=st.session_state.pond_params["initial_weight_g"], key="fc_weight")
    with c2:
        st.number_input("Culture Days", value=st.session_state.pond_params["culture_days"], key="fc_days")
        st.number_input("Mean Temp (°C)", value=st.session_state.pond_params["mean_temperature_c"], key="fc_temp")
        st.number_input("Mean DO (mg/L)", value=st.session_state.pond_params["mean_do_mg_l"], key="fc_do")
    with c3:
        st.number_input("Mean pH", value=st.session_state.pond_params["mean_ph"], key="fc_ph")
        st.selectbox("Season", ["summer", "winter", "monsoon"], index=0, key="fc_season")
        st.selectbox("Intensity", ["extensive", "semi-intensive", "intensive"], index=1, key="fc_intensity")

    if st.button("🔄 Run Forecast", use_container_width=True, key="btn_run_forecast"):
        params = {
            "pond_area_ha": st.session_state.fc_area,
            "stocking_count": int(st.session_state.fc_count),
            "initial_weight_g": st.session_state.fc_weight,
            "culture_days": int(st.session_state.fc_days),
            "mean_temperature_c": st.session_state.fc_temp,
            "season": st.session_state.fc_season,
            "intensity": st.session_state.fc_intensity,
            "feed_protein_pct": 30,
            "mean_do_mg_l": st.session_state.fc_do,
            "min_do_mg_l": st.session_state.fc_do - 1.5,
            "mean_ph": st.session_state.fc_ph,
            "min_ph": st.session_state.fc_ph - 0.3,
            "max_temp_c": st.session_state.fc_temp + 3,
            "min_temp_c": st.session_state.fc_temp - 3,
        }
        result = _api_predict_structured(params)
        if "error" not in result:
            st.session_state.last_prediction = result
            st.session_state.pond_params.update(params)
            st.success("Forecast updated!")
            st.rerun()
        else:
            st.error("Error: " + str(result["error"]))
    st.markdown("</div>", unsafe_allow_html=True)

    if pred:
        st.markdown("<div style='background:#16181f;border:1px solid #23262f;border-radius:14px;padding:20px;'>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-size:32px;font-weight:700;color:#0d9488;'>"
            f"{pred['point_estimate_kg']:.1f} <span style='font-size:14px;font-weight:400;color:#64748b;'>kg</span></div>"
            f"<div style='font-size:11px;color:#64748b;margin-bottom:8px;font-weight:500;'>Point Estimate</div>"
            f"<div style='font-size:14px;color:#e2e8f0;font-weight:600;'>"
            f"{pred['lower_bound_kg']:.0f} – {pred['upper_bound_kg']:.0f} kg (90% CI)</div>",
            unsafe_allow_html=True,
        )
        days = np.arange(0, st.session_state.pond_params["culture_days"] + 1, 5)
        t_norm = days / days[-1]
        biomass = pred["point_estimate_kg"] * (3 * t_norm**2 - 2 * t_norm**3)
        upper = biomass * (pred["upper_bound_kg"] / pred["point_estimate_kg"])
        lower = biomass * (pred["lower_bound_kg"] / pred["point_estimate_kg"])
        chart_df = pd.DataFrame({"Day": days, "Forecast": biomass, "Upper": upper, "Lower": lower})
        st.line_chart(
            chart_df.set_index("Day")[["Forecast", "Upper", "Lower"]],
            color=["#0d9488", "#2a2d3a", "#2a2d3a"],
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

# ==================================================================
# PAGE: SETTINGS
# ==================================================================
def render_settings():
    st.markdown("""
    <div style="padding:8px 0 4px 0;">
        <div style="font-size:22px;font-weight:700;color:#e2e8f0;">⚙️ Settings</div>
    </div>
    <hr style='margin:12px 0 20px 0;border:none;border-top:1px solid #23262f;'>
    """, unsafe_allow_html=True)

    st.markdown("<div style='background:#16181f;border:1px solid #23262f;border-radius:14px;padding:20px;margin-bottom:16px;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:15px;font-weight:600;color:#e2e8f0;margin-bottom:12px;'>API Configuration</div>", unsafe_allow_html=True)
    st.text_input("API Base URL", value=API_BASE, key="settings_api_url", disabled=True)
    st.markdown("<div style='font-size:12px;color:#64748b;margin-top:4px;'>Set via APF_API_URL environment variable. Restart required to apply.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='background:#16181f;border:1px solid #23262f;border-radius:14px;padding:20px;margin-bottom:16px;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:15px;font-weight:600;color:#e2e8f0;margin-bottom:12px;'>Account</div>", unsafe_allow_html=True)
    if st.session_state.user:
        st.markdown(f"<div style='font-size:13px;color:#94a3b8;'>Signed in as <strong style='color:#e2e8f0;'>{st.session_state.user['name']}</strong> ({st.session_state.user['email']})</div>", unsafe_allow_html=True)
        if st.button("🚪 Sign Out", key="btn_settings_logout"):
            st.session_state.jwt_token = None
            st.session_state.user = None
            st.session_state.chat_history = []
            st.session_state.show_suggestions = True
            st.rerun()
    else:
        st.markdown("<div style='font-size:13px;color:#94a3b8;'>Not signed in. Your chat history and uploads are not persisted.</div>", unsafe_allow_html=True)
        st.markdown(f'<a href="{API_BASE}/auth/google" target="_self" style="display:inline-flex;align-items:center;gap:8px;background:#fff;color:#3c4043;padding:10px 20px;border-radius:6px;text-decoration:none;font-size:13px;font-weight:500;margin-top:12px;">Sign in with Google</a>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='background:#16181f;border:1px solid #23262f;border-radius:14px;padding:20px;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:15px;font-weight:600;color:#e2e8f0;margin-bottom:12px;'>About</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:13px;color:#94a3b8;line-height:1.6;">
    <b>AquaPredict AI</b> v1.0 — Aquaculture Production Forecasting<br>
    Built for Nile Tilapia culture systems.<br><br>
    <b>Stack:</b> FastAPI + XGBoost/LightGBM + Streamlit<br>
    <b>Model:</b> v1.1.0-baseline with conformal prediction intervals<br>
    <b>Data:</b> Mechanistic synthetic generator with literature validation
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ==================================================================
# Router
# ==================================================================
page = st.session_state.page
if page == "Chat Assistant":
    render_chat_assistant()
elif page == "Production Forecast":
    render_forecast()
elif page == "Settings":
    render_settings()
else:
    render_chat_assistant()
