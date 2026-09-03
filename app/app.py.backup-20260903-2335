"""APF V1 -- Streamlit Web UI (Real AI Chat Interface).
ChatGPT/Claude-style chat interface with unified bottom input bar,
working speech input, dark theme, and V1-scoped sidebar.

Run:  streamlit run app/app.py
"""
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

from features.build_features import build_features

API_BASE = os.environ.get("APF_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="AquaPredict AI – Smarter Aquaculture",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================================================================
# DARK THEME CSS — Real AI Chat Interface
# ==================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Dark background */
    [data-testid="stAppViewContainer"] { background: #0f1117 !important; }
    .block-container { padding: 0 2rem 2rem 2rem !important; max-width: 1400px; }

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

    /* Chat container */
    .chat-scroll {
        max-height: calc(100vh - 280px);
        overflow-y: auto;
        padding-right: 8px;
    }

    /* Assistant message */
    .msg-assistant {
        display: flex;
        gap: 12px;
        margin-bottom: 24px;
        align-items: flex-start;
    }
    .msg-assistant-avatar {
        width: 28px;
        height: 28px;
        background: linear-gradient(135deg, #0d9488, #0f766e);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 13px;
        flex-shrink: 0;
        margin-top: 4px;
    }
    .msg-assistant-body {
        background: transparent;
        color: #e2e8f0;
        font-size: 15px;
        line-height: 1.7;
        max-width: 85%;
        white-space: pre-wrap;
    }
    .msg-assistant-body strong {
        color: #ffffff;
        font-weight: 600;
    }
    .msg-assistant-body ul {
        margin: 8px 0;
        padding-left: 20px;
    }
    .msg-assistant-body li {
        margin: 4px 0;
    }

    /* User message */
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
        font-size: 15px;
        line-height: 1.6;
        max-width: 80%;
        white-space: pre-wrap;
    }

    /* Thinking / loading */
    .thinking {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #64748b;
        font-size: 14px;
        padding: 8px 0;
    }
    .thinking-dots {
        display: flex;
        gap: 4px;
    }
    .thinking-dots span {
        width: 6px;
        height: 6px;
        background: #0d9488;
        border-radius: 50%;
        animation: pulse 1.4s infinite ease-in-out both;
    }
    .thinking-dots span:nth-child(1) { animation-delay: -0.32s; }
    .thinking-dots span:nth-child(2) { animation-delay: -0.16s; }
    @keyframes pulse {
        0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
        40% { transform: scale(1); opacity: 1; }
    }

    /* Forecast card inside chat */
    .forecast-inline {
        background: #16181f;
        border: 1px solid #23262f;
        border-radius: 14px;
        padding: 20px;
        margin-top: 12px;
        max-width: 400px;
    }
    .forecast-inline .value {
        font-size: 28px;
        font-weight: 700;
        color: #0d9488;
    }
    .forecast-inline .label {
        font-size: 12px;
        color: #64748b;
        margin-bottom: 4px;
    }
    .forecast-inline .range {
        font-size: 13px;
        color: #94a3b8;
        margin-top: 4px;
    }
    .forecast-inline .factors {
        margin-top: 12px;
        font-size: 13px;
        color: #94a3b8;
    }
    .forecast-inline .factors li {
        margin: 3px 0;
    }

    /* Info cards (right panel) */
    .info-card {
        background: #16181f;
        border: 1px solid #23262f;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .info-card h4 {
        margin: 0 0 16px 0;
        font-size: 15px;
        font-weight: 600;
        color: #e2e8f0;
    }

    /* Water quality rows */
    .wq-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 0;
        border-bottom: 1px solid #23262f;
    }
    .wq-row:last-child { border-bottom: none; }

    /* Badges */
    .badge-good {
        background: #064e3b;
        color: #34d399;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
    }
    .badge-medium {
        background: #451a03;
        color: #fbbf24;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
    }

    /* Quick actions */
    .quick-action {
        background: #1e212b;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 10px;
        color: #0d9488;
        font-size: 14px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.15s ease;
        border: 1px solid #2a2d3a;
    }
    .quick-action:hover {
        background: #0d9488;
        color: #ffffff;
        border-color: #0d9488;
    }

    /* Pond selector */
    .pond-selector {
        background: #16181f;
        border: 1px solid #23262f;
        border-radius: 10px;
        padding: 10px 14px;
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 14px;
        color: #e2e8f0;
    }

    /* Section labels */
    .section-label {
        font-size: 10px;
        font-weight: 700;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin: 24px 0 10px 0;
    }

    /* Input bar container */
    .input-bar-wrapper {
        position: sticky;
        bottom: 0;
        background: #0f1117;
        padding: 16px 0 8px 0;
        border-top: 1px solid #23262f;
        margin-top: 20px;
    }

    /* Hide default Streamlit file uploader */
    div[data-testid="stFileUploader"] { display: none; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0f1117; }
    ::-webkit-scrollbar-thumb { background: #2a2d3a; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #3a3d4a; }

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
    .suggestion-card:hover {
        border-color: #0d9488;
        background: #1a1d26;
    }
    .suggestion-card .icon {
        font-size: 20px;
        margin-bottom: 8px;
    }
    .suggestion-card .title {
        font-size: 14px;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 4px;
    }
    .suggestion-card .desc {
        font-size: 12px;
        color: #64748b;
        line-height: 1.4;
    }

    /* Streamlit native input override for dark theme */
    div[data-testid="stTextInput"] input {
        background: #1e212b !important;
        border: 1px solid #2a2d3a !important;
        border-radius: 24px !important;
        color: #e2e8f0 !important;
        padding: 14px 20px !important;
        font-size: 15px !important;
        width: 100% !important;
    }
    div[data-testid="stTextInput"] input::placeholder {
        color: #64748b !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #0d9488 !important;
        box-shadow: 0 0 0 2px rgba(13, 148, 136, 0.2) !important;
        outline: none !important;
    }

    button[kind="primary"] {
        background: linear-gradient(135deg, #0d9488, #0f766e) !important;
        border-radius: 50% !important;
        width: 40px !important;
        height: 40px !important;
        border: none !important;
        color: white !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
</style>
""", unsafe_allow_html=True)

# ==================================================================
# Session state
# ==================================================================
def init_state():
    if "page" not in st.session_state:
        st.session_state.page = "Chat Assistant"
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": "Hello! I'm your AquaPredict AI assistant.\n\nI can help you forecast tilapia production, analyze water quality, and answer aquaculture questions. Just describe your pond setup or ask me anything.",
                "time": "10:30 AM"
            }
        ]
    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None
    if "pond_params" not in st.session_state:
        st.session_state.pond_params = {
            "pond_area_ha": 0.5,
            "stocking_count": 3000,
            "initial_weight_g": 15.0,
            "culture_days": 120,
            "mean_temperature_c": 28.0,
            "season": "summer",
            "intensity": "semi-intensive",
            "feed_protein_pct": 30,
            "mean_do_mg_l": 7.5,
            "min_do_mg_l": 5.0,
            "mean_ph": 7.5,
            "min_ph": 6.8,
            "max_temp_c": 32.0,
            "min_temp_c": 24.0,
        }
    if "analyzing" not in st.session_state:
        st.session_state.analyzing = False
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    if "show_suggestions" not in st.session_state:
        st.session_state.show_suggestions = True

init_state()

# ==================================================================
# API helpers
# ==================================================================
def _api_predict_structured(params: dict) -> dict:
    try:
        r = requests.post(f"{API_BASE}/predict", json=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def _api_predict_text(text: str) -> dict:
    try:
        r = requests.post(
            f"{API_BASE}/predict/extract",
            params={"farmer_text": text},
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

# ==================================================================
# Sidebar navigation (V1 scope only)
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
    <div style="margin-top:40px;">
        <div style="display:flex;align-items:center;gap:10px;padding:12px;background:#1e212b;border-radius:10px;border:1px solid #2a2d3a;">
            <div style="width:10px;height:10px;background:{status_color};border-radius:50%;box-shadow:0 0 6px {status_color};"></div>
            <div>
                <div style="font-size:12px;font-weight:600;color:#e2e8f0;">API Server</div>
                <div style="font-size:11px;color:#64748b;">{status_text} • {API_BASE}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:16px;">
        <div style="display:flex;align-items:center;gap:10px;padding:12px;background:#1e212b;border-radius:10px;border:1px solid #2a2d3a;">
            <div style="width:32px;height:32px;background:#0d9488;border-radius:50%;display:flex;align-items:center;justify-content:center;color:white;font-size:14px;font-weight:600;">F</div>
            <div>
                <div style="font-size:13px;font-weight:600;color:#e2e8f0;">Farmer</div>
                <div style="font-size:11px;color:#64748b;">Basic Plan</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==================================================================
# Top bar
# ==================================================================
top_col1, top_col2 = st.columns([3, 1])
with top_col1:
    greeting = "Good morning" if st.session_state.page == "Chat Assistant" else "Production Forecast"
    sub = "Ask anything about your pond or get a production forecast." if st.session_state.page == "Chat Assistant" else "Enter your pond parameters for a detailed forecast."
    st.markdown(f"""
    <div style="padding:8px 0 4px 0;">
        <div style="font-size:22px;font-weight:700;color:#e2e8f0;">{greeting}, Farmer 👋</div>
        <div style="font-size:14px;color:#64748b;margin-top:4px;">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

with top_col2:
    params = st.session_state.pond_params
    days_str = f"{params['culture_days']} days"
    st.markdown(f"""
    <div style="display:flex;justify-content:flex-end;padding-top:8px;">
        <div class="pond-selector">
            <span style="font-size:18px;">🧪</span>
            <div>
                <div style="font-weight:600;color:#e2e8f0;">Pond 1</div>
                <div style="font-size:12px;color:#64748b;">Nile Tilapia • {days_str}</div>
            </div>
            <span style="margin-left:8px;color:#64748b;">▼</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='margin:12px 0 20px 0;border:none;border-top:1px solid #23262f;'>", unsafe_allow_html=True)

# ==================================================================
# Helper: Format assistant response with optional forecast card
# ==================================================================
def format_assistant_response(text: str, prediction: dict = None) -> str:
    """Format assistant text. If prediction provided, append inline forecast card."""
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
# PAGE: CHAT ASSISTANT (Real AI Interface)
# ==================================================================
def render_chat_assistant():
    center_col, right_col = st.columns([2.2, 1])

    with center_col:
        # Chat messages scroll area
        st.markdown('<div class="chat-scroll">', unsafe_allow_html=True)

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

        st.markdown('</div>', unsafe_allow_html=True)

        # Welcome suggestions (only show on first load / empty chat beyond welcome)
        if st.session_state.show_suggestions and len(st.session_state.chat_history) <= 1:
            st.markdown("<div style='margin:8px 0 16px 40px;'><div style='font-size:13px;color:#64748b;margin-bottom:12px;'>Try asking:</div></div>", unsafe_allow_html=True)
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
                        st.session_state.chat_history.append({
                            "role": "user",
                            "content": title,
                            "time": "10:32 AM"
                        })
                        st.session_state.show_suggestions = False
                        # Handle suggestion as a quick response
                        params = st.session_state.pond_params
                        if "Forecast" in title:
                            if st.session_state.last_prediction:
                                pred = st.session_state.last_prediction
                                pe = pred.get("point_estimate_kg", 0)
                                lb = pred.get("lower_bound_kg", 0)
                                ub = pred.get("upper_bound_kg", 0)
                                resp = f"Based on your latest pond data, here is the production forecast."
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": resp,
                                    "time": "10:32 AM",
                                    "prediction": pred
                                })
                            else:
                                resp = "I don't have enough data to forecast yet. Please describe your pond — area, stocking count, culture days, temperature, DO, and pH — and I'll generate a forecast."
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": resp,
                                    "time": "10:32 AM"
                                })
                        elif "Water quality" in title:
                            resp = f"Your current water quality looks good. Dissolved oxygen is at {params['mean_do_mg_l']:.1f} mg/L, pH is {params['mean_ph']:.1f}, and temperature is {params['mean_temperature_c']:.1f}°C. Keep monitoring DO — it should stay above 5 mg/L for optimal tilapia growth."
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": resp,
                                "time": "10:32 AM"
                            })
                        elif "Feeding" in title:
                            resp = "For semi-intensive Nile tilapia culture, feed 28–32% protein pellets at 3–5% body weight daily for the first month, then reduce to 2–3% as fish grow. Adjust based on water temperature — feeding efficiency drops below 24°C and above 32°C."
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": resp,
                                "time": "10:32 AM"
                            })
                        else:
                            pct = min(95, 15 + params["culture_days"] * 0.7)
                            resp = f"Your culture is at day {params['culture_days']}. At this stage, tilapia should be approaching {pct:.0f}% of final harvest weight. Continue monitoring water quality and feed conversion ratio."
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": resp,
                                "time": "10:32 AM"
                            })
                        st.rerun()

        # Unified bottom input bar
        st.markdown('<div class="input-bar-wrapper">', unsafe_allow_html=True)

        # Hidden file uploader
        uploaded = st.file_uploader(
            "",
            type=["jpg", "jpeg", "png", "pdf", "csv"],
            key="chat_file_uploader",
            label_visibility="collapsed"
        )
        if uploaded is not None and uploaded != st.session_state.get("last_uploaded"):
            st.session_state.last_uploaded = uploaded
            st.session_state.uploaded_file = uploaded
            st.session_state.chat_history.append({
                "role": "user",
                "content": f"📎 {uploaded.name}",
                "time": "10:31 AM"
            })
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"I've received your file **{uploaded.name}**. For V1, I can analyze CSV data with pond parameters. Image-based disease detection is coming in V2. If this is a CSV with pond data, I can process it for a forecast.",
                "time": "10:32 AM"
            })
            st.rerun()

        # Input row: [+ attach] [text input] [mic] [send]
        input_row = st.columns([0.6, 5.5, 0.6, 0.6])

        with input_row[0]:
            # Plus/attach button
            st.markdown("""
            <div style="width:40px;height:40px;border-radius:50%;border:1px solid #2a2d3a;background:#1e212b;display:flex;align-items:center;justify-content:center;cursor:pointer;margin-top:4px;transition:all 0.15s;"
                 onmouseover="this.style.background='#2a2d3a';"
                 onmouseout="this.style.background='#1e212b';"
                 onclick="
                    const uploaders = document.querySelectorAll('input[type=\'file\']');
                    for (let u of uploaders) { if (u.offsetParent !== null) u.click(); }
                 ">
                <span style="font-size:20px;color:#94a3b8;">+</span>
            </div>
            """, unsafe_allow_html=True)

        with input_row[1]:
            user_text = st.text_input("", placeholder="Ask AquaPredict AI...", key="chat_input", label_visibility="collapsed")

        with input_row[2]:
            # Microphone button with Web Speech API
            st.markdown("""
            <div style="width:40px;height:40px;border-radius:50%;border:1px solid #2a2d3a;background:#1e212b;display:flex;align-items:center;justify-content:center;cursor:pointer;margin-top:4px;transition:all 0.15s;"
                 id="mic-btn"
                 onmouseover="this.style.background='#2a2d3a';"
                 onmouseout="this.style.background='#1e212b';"
                 onclick="
                    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
                        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                        const rec = new SpeechRecognition();
                        rec.lang = 'en-US';
                        rec.interimResults = false;
                        rec.maxAlternatives = 1;
                        rec.start();
                        this.style.background='#0d9488';
                        this.style.borderColor='#0d9488';
                        this.innerHTML='<span style=\'font-size:18px;color:white;\'>🔴</span>';
                        rec.onresult = function(e) {
                            const txt = e.results[0][0].transcript;
                            const inputs = document.querySelectorAll('input[type=\'text\']');
                            for (let inp of inputs) {
                                if (inp.placeholder && inp.placeholder.includes('Ask AquaPredict')) {
                                    inp.value = txt;
                                    inp.dispatchEvent(new Event('input', { bubbles: true }));
                                    break;
                                }
                            }
                        };
                        rec.onend = function() {
                            const btn = document.getElementById('mic-btn');
                            if (btn) {
                                btn.style.background='#1e212b';
                                btn.style.borderColor='#2a2d3a';
                                btn.innerHTML='<span style=\'font-size:18px;color:#94a3b8;\'>🎙️</span>';
                            }
                        };
                    } else {
                        alert('Speech recognition not supported. Use Chrome or Edge.');
                    }
                 ">
                <span style="font-size:18px;color:#94a3b8;">🎙️</span>
            </div>
            """, unsafe_allow_html=True)

        with input_row[3]:
            send_clicked = st.button("➤", key="btn_send")

        if send_clicked and user_text and user_text.strip():
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_text.strip(),
                "time": "10:31 AM"
            })
            st.session_state.analyzing = True
            st.session_state.show_suggestions = False
            st.rerun()

        # Handle analysis after rerun
        if st.session_state.analyzing:
            last_user_msg = None
            for msg in reversed(st.session_state.chat_history):
                if msg["role"] == "user":
                    last_user_msg = msg["content"]
                    break

            if last_user_msg:
                result = _api_predict_text(last_user_msg)
                if result.get("status") == "incomplete":
                    missing = result.get("missing_fields", [])
                    followup = result.get("follow_up_question", "Could you provide more details?")
                    resp = followup + "\n\nMissing: " + ", ".join(missing)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": resp,
                        "time": "10:32 AM"
                    })
                elif result.get("status") == "complete":
                    pred_data = result.get("prediction", {})
                    st.session_state.last_prediction = pred_data
                    st.session_state.pond_params.update(result.get("extracted", {}))
                    explanation = result.get("explanation", "")
                    pe = pred_data.get("point_estimate_kg", 0)
                    lb = pred_data.get("lower_bound_kg", 0)
                    ub = pred_data.get("upper_bound_kg", 0)
                    resp = (
                        "Here is your production forecast based on the pond details you provided."
                    )
                    if explanation:
                        resp += f"\n\n{explanation}"
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": resp,
                        "time": "10:32 AM",
                        "prediction": pred_data
                    })
                elif "error" in result:
                    resp = "Sorry, I encountered an error connecting to the forecasting engine: " + str(result["error"])
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": resp,
                        "time": "10:32 AM"
                    })
                else:
                    resp = "I'm not sure how to process that. Try describing your pond with area, stocking count, days, temperature, DO, and pH."
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": resp,
                        "time": "10:32 AM"
                    })

            st.session_state.analyzing = False
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        # Pond Overview
        st.markdown("""
        <div class="info-card">
            <h4>Pond Overview</h4>
        """, unsafe_allow_html=True)

        p = st.session_state.pond_params
        overview_items = [
            ("🐟", "Species", "Nile Tilapia"),
            ("🧪", "Pond Area", f"{p['pond_area_ha']:.2f} hectares"),
            ("📅", "Culture Day", f"{p['culture_days']} days"),
            ("👥", "Stocking", f"{p['stocking_count']:,}"),
            ("🕐", "Last Updated", "10:25 AM, Today"),
        ]
        for icon, label, value in overview_items:
            st.markdown(f"""
            <div class="wq-row">
                <div style="display:flex;align-items:center;gap:10px;">
                    <span style="font-size:16px;">{icon}</span>
                    <span style="font-size:13px;color:#94a3b8;">{label}</span>
                </div>
                <span style="font-size:13px;font-weight:600;color:#e2e8f0;">{value}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Water Quality
        st.markdown("""
        <div class="info-card">
            <h4>Water Quality <span style="font-size:12px;color:#64748b;font-weight:400;">(Latest)</span></h4>
        """, unsafe_allow_html=True)

        do_status = "Good" if p["mean_do_mg_l"] >= 5 else "Medium"
        do_badge = "badge-good" if do_status == "Good" else "badge-medium"

        wq_items = [
            ("🌡️", "Temperature", f"{p['mean_temperature_c']:.1f} °C", "Good", "badge-good"),
            ("⚗️", "pH", f"{p['mean_ph']:.1f}", "Good", "badge-good"),
            ("💨", "Dissolved Oxygen", f"{p['mean_do_mg_l']:.1f} mg/L", do_status, do_badge),
            ("🧂", "Salinity", "0.5 ppt", "Good", "badge-good"),
            ("☠️", "Ammonia (NH₃)", "0.12 mg/L", "Good", "badge-good"),
        ]
        for icon, label, value, status, badge_class in wq_items:
            st.markdown(f"""
            <div class="wq-row">
                <div style="display:flex;align-items:center;gap:10px;">
                    <span style="font-size:16px;">{icon}</span>
                    <span style="font-size:13px;color:#94a3b8;">{label}</span>
                </div>
                <div style="display:flex;align-items:center;gap:10px;">
                    <span style="font-size:13px;font-weight:500;color:#e2e8f0;">{value}</span>
                    <span class="{badge_class}">{status}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
            <div style="text-align:right;margin-top:12px;">
                <a href="#" style="font-size:12px;color:#0d9488;text-decoration:none;font-weight:500;">View all measurements →</a>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Quick Actions
        st.markdown("""
        <div class="info-card">
            <h4>Quick Actions</h4>
        """, unsafe_allow_html=True)

        qa_items = [
            ("📝", "Add Measurement"),
            ("📊", "View Forecasts"),
        ]
        for icon, label in qa_items:
            st.markdown(f"""
            <div class="quick-action">
                <span style="font-size:16px;">{icon}</span>
                <span>{label}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ==================================================================
# PAGE: PRODUCTION FORECAST (structured form)
# ==================================================================
def render_forecast():
    st.markdown("<div style='font-size:24px;font-weight:700;color:#e2e8f0;margin-bottom:1rem;'>📊 Production Forecast</div>", unsafe_allow_html=True)
    pred = st.session_state.last_prediction

    st.markdown("<div class='info-card'>", unsafe_allow_html=True)
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
        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='info-card'>", unsafe_allow_html=True)
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
        chart_df = pd.DataFrame({
            "Day": days,
            "Forecast": biomass,
            "Upper": upper,
            "Lower": lower,
        })
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
    st.markdown("<div style='font-size:24px;font-weight:700;color:#e2e8f0;margin-bottom:1rem;'>⚙️ Settings</div>", unsafe_allow_html=True)
    st.markdown("<div class='info-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:15px;font-weight:600;color:#e2e8f0;margin-bottom:12px;'>API Configuration</div>", unsafe_allow_html=True)
    st.text_input("API Base URL", value=API_BASE, key="settings_api_url", disabled=True)
    st.markdown("<div style='font-size:12px;color:#64748b;margin-top:4px;'>Set via APF_API_URL environment variable. Restart required to apply.</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:20px;font-size:15px;font-weight:600;color:#e2e8f0;margin-bottom:12px;'>Speech Input</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:13px;color:#94a3b8;line-height:1.6;'>Speech recognition uses your browser's built-in Web Speech API (Chrome/Edge). No audio is sent to any server — everything happens locally in your browser.</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:20px;font-size:15px;font-weight:600;color:#e2e8f0;margin-bottom:12px;'>About</div>", unsafe_allow_html=True)
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