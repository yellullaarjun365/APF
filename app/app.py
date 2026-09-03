
fixed_app_py = '''"""APF V1 -- Streamlit Web UI (fixed for actual API endpoints).

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

    /* Section labels */
    .section-label {
        font-size: 10px; font-weight: 700; color: #475569;
        text-transform: uppercase; letter-spacing: 0.8px;
        margin: 24px 0 10px 0;
    }

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
        "voice_triggered": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ==================================================================
# API helpers
# ==================================================================
def _api_post(path: str, payload: dict):
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, headers={"Content-Type": "application/json"}, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def _api_predict_structured(params: dict) -> dict:
    return _api_post("/predict", params)

def _api_predict_text(text: str) -> dict:
    """FIXED: use json= instead of params= so the API receives the body correctly."""
    return _api_post("/predict/extract", {"farmer_text": text})

def _api_health() -> dict:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}

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

    st.markdown('<div class="section-label">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="padding:12px;background:#1e212b;border-radius:10px;border:1px solid #2a2d3a;font-size:12px;color:#94a3b8;line-height:1.5;">
        <b>AquaPredict AI</b> v1.0<br>
        Nile Tilapia production forecasting.<br><br>
        <b>Stack:</b> FastAPI + LightGBM + Streamlit<br>
        <b>Model:</b> v1.1.0-baseline<br>
        <b>Data:</b> Mechanistic synthetic generator
    </div>
    """, unsafe_allow_html=True)

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
            # FIXED: API returns "importance", not "impact_kg"
            imp = f.get("importance", 0)
            factors_html += f"<li>{f['feature'].replace('_', ' ').title()}: {imp:.3f}</li>"
        html += f"""
        <div class="forecast-inline">
            <div class="label">Estimated Production</div>
            <div class="value">{pe:.1f} <span style="font-size:14px;font-weight:400;color:#64748b;">kg</span></div>
            <div class="range">Range: {lb:.0f} – {ub:.0f} kg (90% CI)</div>
            <div class="factors">
                <strong style="color:#e2e8f0;">Top Factors:</strong>
                <ul style="margin-top:4px;">{factors_html}</ul>
            </div>
        </div>
        """
    return html

# ==================================================================
# PAGE: CHAT ASSISTANT
# ==================================================================
def render_chat_assistant():
    st.markdown("""
    <div style="padding:8px 0 4px 0;">
        <div style="font-size:22px;font-weight:700;color:#e2e8f0;">💬 Chat Assistant</div>
        <div style="font-size:14px;color:#64748b;margin-top:4px;">Describe your pond in plain text and get a production forecast.</div>
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
        return

    # ---- INPUT BAR ----
    st.markdown('<div class="input-bar-wrapper">', unsafe_allow_html=True)

    input_row = st.columns([0.5, 5.5, 0.5, 0.5])

    with input_row[0]:
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
            if msg["role"] == "user":
                last_user_msg = msg["content"]
                break

        if last_user_msg:
            result = _api_predict_text(last_user_msg)
            if result.get("status") == "incomplete":
                missing = result.get("missing_fields", [])
                followup = result.get("follow_up_question", "Could you provide more details?")
                resp = followup + "\\n\\nMissing: " + ", ".join(missing)
                st.session_state.chat_history.append({"role": "assistant", "content": resp, "time": ""})
            elif result.get("status") == "complete":
                pred_data = result.get("prediction", {})
                st.session_state.last_prediction = pred_data
                st.session_state.pond_params.update(result.get("extracted", {}))
                explanation = result.get("explanation", "")
                resp = "Here is your production forecast based on the pond details you provided."
                if explanation:
                    resp += f"\\n\\n{explanation}"
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
# Router
# ==================================================================
page = st.session_state.page
if page == "Chat Assistant":
    render_chat_assistant()
elif page == "Production Forecast":
    render_forecast()
else:
    render_chat_assistant()
'''

with open("/mnt/agents/output/app.py", "w") as f:
    f.write(fixed_app_py)

print("app.py written")
