"""APF V1 -- Streamlit Web UI.

Run:  streamlit run app/app.py

Requires: app/assets/bg_base64.txt (background image, base64-encoded).
If that file is missing, the app falls back to a plain dark background
instead of crashing.
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

API_BASE = os.environ.get("APF_API_URL", "http://localhost:8000")

# ==================================================================
# Temporary storage for anything the user uploads via chat.
# Not committed to git (see .gitignore note in the deployment steps) --
# this is scratch space, not a permanent data store.
# ==================================================================
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="AquaPredict AI -- Smarter Aquaculture",
    page_icon="\U0001F41F",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================================================================
# Background image (base64, so no static-file-serving config needed)
# ==================================================================
_BG_FILE = APP_DIR / "assets" / "bg_base64.txt"
_BG_B64 = _BG_FILE.read_text().strip() if _BG_FILE.exists() else ""

if _BG_B64:
    _bg_css = f"""
    [data-testid="stAppViewContainer"] {{
        background:
            linear-gradient(rgba(6,10,16,0.45), rgba(6,10,16,0.55)),
            url('data:image/jpeg;base64,{_BG_B64}') no-repeat center center fixed;
        background-size: cover;
    }}
    """
else:
    _bg_css = """
    [data-testid="stAppViewContainer"] { background: #0f1117 !important; }
    """

# ==================================================================
# CSS
# ==================================================================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    /* NOTE: header is kept (not hidden) -- fully hiding it also hides the
       sidebar collapse/expand control on some Streamlit versions, which is
       what made the sidebar disappear with no way to bring it back. We just
       make it transparent so it blends in instead. */
    header {{ background: transparent !important; box-shadow: none !important; }}

    {_bg_css}

    .block-container {{ padding: 0 2rem 140px 2rem !important; max-width: 1000px; }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: rgba(22, 24, 31, 0.92) !important;
        border-right: 1px solid #23262f;
    }}
    [data-testid="stSidebar"] > div:first-child {{ padding-top: 1rem !important; }}

    .nav-active {{
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
    }}
    .nav-item {{
        color: #94a3b8;
        padding: 10px 14px;
        border-radius: 10px;
        font-size: 14px;
    }}

    /* Chat messages */
    .msg-assistant {{
        display: flex;
        gap: 12px;
        margin-bottom: 24px;
        align-items: flex-start;
    }}
    .msg-assistant-avatar {{
        width: 28px; height: 28px;
        background: linear-gradient(135deg, #0d9488, #0f766e);
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        color: white; font-size: 13px; flex-shrink: 0; margin-top: 4px;
    }}
    .msg-assistant-body {{
        background: rgba(15, 17, 23, 0.78);
        border-radius: 14px;
        padding: 12px 16px;
        color: #e2e8f0;
        font-size: 15px; line-height: 1.7;
        max-width: 85%; white-space: pre-wrap;
    }}
    .msg-assistant-body strong {{ color: #ffffff; font-weight: 600; }}

    .msg-user {{
        display: flex;
        justify-content: flex-end;
        margin-bottom: 24px;
    }}
    .msg-user-body {{
        background: rgba(30, 33, 43, 0.9);
        border: 1px solid #2a2d3a;
        border-radius: 18px 18px 4px 18px;
        padding: 12px 18px;
        color: #e2e8f0;
        font-size: 15px; line-height: 1.6;
        max-width: 80%; white-space: pre-wrap;
    }}

    /* Thinking indicator */
    .thinking {{ display: flex; align-items: center; gap: 8px; color: #94a3b8; font-size: 14px; padding: 8px 0; }}
    .thinking-dots {{ display: flex; gap: 4px; }}
    .thinking-dots span {{
        width: 6px; height: 6px; background: #0d9488; border-radius: 50%;
        animation: pulse 1.4s infinite ease-in-out both;
    }}
    .thinking-dots span:nth-child(1) {{ animation-delay: -0.32s; }}
    .thinking-dots span:nth-child(2) {{ animation-delay: -0.16s; }}
    @keyframes pulse {{
        0%, 80%, 100% {{ transform: scale(0.6); opacity: 0.4; }}
        40% {{ transform: scale(1); opacity: 1; }}
    }}

    /* Forecast inline card */
    .forecast-inline {{
        background: rgba(22, 24, 31, 0.85);
        border: 1px solid #23262f;
        border-radius: 14px;
        padding: 20px;
        margin-top: 12px;
        max-width: 420px;
    }}
    .forecast-inline .value {{ font-size: 28px; font-weight: 700; color: #0d9488; }}
    .forecast-inline .label {{ font-size: 12px; color: #94a3b8; margin-bottom: 4px; }}
    .forecast-inline .range {{ font-size: 13px; color: #cbd5e1; margin-top: 4px; }}
    .forecast-inline .factors {{ margin-top: 12px; font-size: 13px; color: #cbd5e1; }}

    /* ---- Fixed bottom input bar (pinned to viewport bottom, ChatGPT-style) ---- */
    .input-bar-fixed {{
        position: fixed;
        bottom: 0; left: 0; right: 0;
        z-index: 999;
        background: linear-gradient(to top, rgba(6,10,16,0.97) 60%, rgba(6,10,16,0));
        padding: 24px 0 18px 0;
    }}
    .input-bar-inner {{
        max-width: 1000px;
        margin: 0 auto;
        padding: 0 2rem;
    }}
    div[data-testid="stTextInput"] input {{
        background: #1a1d26 !important;
        border: 1px solid #2a2d3a !important;
        border-radius: 24px !important;
        color: #e2e8f0 !important;
        padding: 12px 18px !important;
    }}
    div[data-testid="stTextInput"] input:focus {{
        border-color: #0d9488 !important;
        box-shadow: 0 0 0 1px #0d9488 !important;
    }}
    div[data-testid="stForm"] {{ border: none !important; padding: 0 !important; }}

    /* Compact file-attach control -- shrink Streamlit's default drag-drop
       box down to something that reads as an "attach" row, not a whole
       upload panel. */
    div[data-testid="stFileUploader"] {{
        background: #1a1d26;
        border: 1px dashed #2a2d3a;
        border-radius: 14px;
        padding: 6px 10px;
        margin-bottom: 8px;
    }}
    div[data-testid="stFileUploaderDropzone"] {{
        background: transparent !important;
        padding: 4px !important;
        min-height: 0 !important;
    }}
    div[data-testid="stFileUploaderDropzoneInstructions"] span {{ font-size: 12px !important; }}
    div[data-testid="stFileUploaderDropzoneInstructions"] small {{ font-size: 10px !important; }}

    /* Section labels */
    .section-label {{
        font-size: 10px; font-weight: 700; color: #94a3b8;
        text-transform: uppercase; letter-spacing: 0.8px;
        margin: 24px 0 10px 0;
    }}
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
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ==================================================================
# Voice transcript handoff -- MUST run before the chat_input widget is
# created below. This is what fixes the "auto-sends without me clicking
# send" bug: we only ever populate the text box, never trigger a send.
# ==================================================================
if "voice_text" in st.query_params:
    _vtext = st.query_params.get("voice_text", "")
    st.query_params.clear()
    if _vtext and _vtext.strip():
        st.session_state["chat_input"] = _vtext.strip()

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
        <div style="width:36px;height:36px;background:linear-gradient(135deg,#0d9488,#0f766e);border-radius:50%;display:flex;align-items:center;justify-content:center;color:white;font-size:18px;">\U0001F41F</div>
        <div>
            <div style="font-size:16px;font-weight:700;color:#e2e8f0;">AquaPredict AI</div>
            <div style="font-size:12px;color:#94a3b8;">Smarter Aquaculture</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Main</div>', unsafe_allow_html=True)

    nav_items = [
        ("\U0001F4AC", "Chat Assistant", "Chat Assistant"),
        ("\U0001F4CA", "Production Forecast", "Production Forecast"),
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
    <div style="padding:12px;background:rgba(30,33,43,0.8);border-radius:10px;border:1px solid #2a2d3a;font-size:12px;color:#94a3b8;line-height:1.5;">
        <b>AquaPredict AI</b> v1.0<br>
        Nile Tilapia production forecasting.<br><br>
        <b>Stack:</b> FastAPI + LightGBM + Streamlit<br>
        <b>Model:</b> v1.1.0-baseline
    </div>
    """, unsafe_allow_html=True)

    health = _api_health()
    backend_ok = health.get("status") == "ok"
    status_color = "#34d399" if backend_ok else "#f87171"
    status_text = "Online" if backend_ok else "Offline"
    st.markdown(f"""
    <div style="margin-top:24px;">
        <div style="display:flex;align-items:center;gap:10px;padding:12px;background:rgba(30,33,43,0.8);border-radius:10px;border:1px solid #2a2d3a;">
            <div style="width:10px;height:10px;background:{status_color};border-radius:50%;box-shadow:0 0 6px {status_color};"></div>
            <div>
                <div style="font-size:12px;font-weight:600;color:#e2e8f0;">API Server</div>
                <div style="font-size:11px;color:#94a3b8;">{status_text} &bull; {API_BASE}</div>
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
            imp = f.get("importance", 0)
            factors_html += f"<li>{f['feature'].replace('_', ' ').title()}: {imp:.3f}</li>"
        html += f"""
        <div class="forecast-inline">
            <div class="label">Estimated Production</div>
            <div class="value">{pe:.1f} <span style="font-size:14px;font-weight:400;color:#94a3b8;">kg</span></div>
            <div class="range">Range: {lb:.0f} - {ub:.0f} kg (90% CI)</div>
            <div class="factors">
                <strong style="color:#e2e8f0;">Top Factors:</strong>
                <ul style="margin-top:4px;">{factors_html}</ul>
            </div>
        </div>
        """
    return html

def _handle_user_message(text: str):
    """Add user message, trigger analysis, and rerun. This is the ONLY
    place a message is actually sent -- voice input never calls this
    directly, it only fills the text box (see query-param handling above)."""
    st.session_state.chat_history.append({"role": "user", "content": text, "time": ""})
    st.session_state.analyzing = True
    st.session_state.show_suggestions = False
    st.rerun()

# ==================================================================
# PAGE: CHAT ASSISTANT
# ==================================================================
def render_chat_assistant():
    st.markdown("""
    <div style="padding:8px 0 4px 0;">
        <div style="font-size:22px;font-weight:700;color:#e2e8f0;">\U0001F4AC Chat Assistant</div>
        <div style="font-size:14px;color:#94a3b8;margin-top:4px;">Ask anything about your pond or get a production forecast.</div>
    </div>
    <hr style='margin:12px 0 20px 0;border:none;border-top:1px solid #23262f;'>
    """, unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        if msg["role"] == "assistant":
            pred = msg.get("prediction")
            body_html = format_assistant_response(msg["content"], pred)
            st.markdown(f"""
            <div class="msg-assistant">
                <div class="msg-assistant-avatar">\U0001F41F</div>
                <div>{body_html}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="msg-user">
                <div class="msg-user-body">{msg["content"]}</div>
            </div>
            """, unsafe_allow_html=True)

    if st.session_state.analyzing:
        st.markdown("""
        <div class="msg-assistant">
            <div class="msg-assistant-avatar">\U0001F41F</div>
            <div class="thinking">
                <div class="thinking-dots"><span></span><span></span><span></span></div>
                <span>Analyzing your pond data...</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.show_suggestions and len(st.session_state.chat_history) == 0:
        st.markdown("<div style='font-size:13px;color:#94a3b8;margin-bottom:12px;'>Try asking:</div>", unsafe_allow_html=True)
        sugg_col1, sugg_col2 = st.columns(2)
        suggestions = [
            ("\U0001F4CA", "Forecast my production"),
            ("\U0001F4A7", "Check water quality"),
            ("\U0001F37D", "Feeding recommendation"),
            ("\U0001F4C8", "Growth status"),
        ]
        for i, (icon, title) in enumerate(suggestions):
            col = sugg_col1 if i < 2 else sugg_col2
            with col:
                if st.button(f"{icon} {title}", key=f"sugg_{i}", use_container_width=True):
                    _handle_user_message(title)

    # ---- FIXED BOTTOM INPUT BAR ----
    st.markdown('<div class="input-bar-fixed"><div class="input-bar-inner">', unsafe_allow_html=True)

    outer = st.columns([0.6, 5.4])

    with outer[0]:
        if st.button("\U0001F3A4", key="btn_voice", help="Click, then speak (Chrome/Edge only -- Brave blocks this by default)"):
            st.components.v1.html("""
            <script>
            (function() {
                if (!('webkitSpeechRecognition' in window)) {
                    alert('Speech recognition not supported in this browser. Use Chrome or Edge.');
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
                    window.parent.location.href = url.toString();
                };
                rec.onerror = function(e) {
                    if (e.error === 'network') {
                        alert('Speech error: network.\\n\\nThis usually means your browser blocks the speech-recognition service (Brave does this by default for privacy). Try Chrome or Edge instead.');
                    } else {
                        alert('Speech error: ' + e.error);
                    }
                };
            })();
            </script>
            """, height=0)

    with outer[1]:
        # Attach control -- compact file uploader. Whatever's picked shows
        # as Streamlit's own filename/size chip (with a built-in remove
        # "x") right above the text box, exactly like the reference UI,
        # until the message is actually sent.
        attached_file = st.file_uploader(
            "Attach", type=["jpg", "jpeg", "png", "pdf", "csv", "wav", "mp3", "m4a"],
            key="chat_attachment", label_visibility="collapsed",
            accept_multiple_files=False,
        )

        # st.form gives us two things we didn't have before: pressing
        # Enter submits (not just clicking the arrow), and clear_on_submit
        # empties the text box + attachment automatically after sending --
        # this is what was making "send" feel broken (it worked, but the
        # box never cleared and Enter did nothing).
        with st.form(key="chat_form", clear_on_submit=True):
            form_cols = st.columns([5.4, 0.6])
            with form_cols[0]:
                user_text = st.text_input(
                    "", placeholder="Ask AquaPredict AI...", key="chat_input", label_visibility="collapsed",
                )
            with form_cols[1]:
                send_clicked = st.form_submit_button("\u27A4")

    st.markdown('</div></div>', unsafe_allow_html=True)

    if send_clicked and ((user_text and user_text.strip()) or attached_file is not None):
        display_text = user_text.strip() if user_text else ""
        saved_path = None
        if attached_file is not None:
            import time
            ts = time.strftime("%Y%m%d_%H%M%S")
            saved_path = UPLOAD_DIR / f"{ts}_{attached_file.name}"
            saved_path.write_bytes(attached_file.getvalue())
            attach_note = f"\U0001F4CE {attached_file.name}"
            display_text = f"{display_text}\n\n{attach_note}" if display_text else attach_note

        st.session_state.chat_history.append({
            "role": "user", "content": display_text, "time": "",
            "attachment_path": str(saved_path) if saved_path else None,
        })

        if user_text and user_text.strip():
            # There's real text -- run it through the actual pipeline.
            st.session_state.analyzing = True
        else:
            # File only, no text: V1's model only understands typed pond
            # descriptions (see PROJECT_MANUAL.md V1 scope). Say so
            # honestly instead of pretending to analyze it.
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": (
                    f"Saved **{attached_file.name}** to temporary storage "
                    f"(`data/uploads/`). V1's forecasting model only reads "
                    f"typed pond descriptions right now -- file analysis "
                    f"(images, CSVs) is planned for V2. Feel free to "
                    f"describe your pond in text and I'll forecast it."
                ),
                "time": "",
            })

        st.session_state.show_suggestions = False
        st.rerun()

    # ---- REAL PIPELINE CALL: text -> /predict/extract (extraction -> model -> explanation) ----
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
                st.session_state.chat_history.append({"role": "assistant", "content": resp, "time": ""})
            elif result.get("status") == "complete":
                pred_data = result.get("prediction", {})
                st.session_state.last_prediction = pred_data
                st.session_state.pond_params.update(result.get("extracted", {}))
                explanation = result.get("explanation", "")
                resp = "Here is your production forecast based on the pond details you provided."
                if explanation:
                    resp += f"\n\n{explanation}"
                st.session_state.chat_history.append({
                    "role": "assistant", "content": resp, "time": "", "prediction": pred_data,
                })
            elif "error" in result:
                resp = "Sorry, I couldn't reach the forecasting engine: " + str(result["error"]) + "\n\n(Is the API running? `uvicorn src.api.main:app --reload --port 8000`)"
                st.session_state.chat_history.append({"role": "assistant", "content": resp, "time": ""})
            else:
                resp = "I'm not sure how to process that. Try describing your pond with area, stocking count, culture days, temperature, DO, and pH."
                st.session_state.chat_history.append({"role": "assistant", "content": resp, "time": ""})

        st.session_state.analyzing = False
        st.rerun()

# ==================================================================
# PAGE: PRODUCTION FORECAST
# ==================================================================
def render_forecast():
    st.markdown("""
    <div style="padding:8px 0 4px 0;">
        <div style="font-size:22px;font-weight:700;color:#e2e8f0;">\U0001F4CA Production Forecast</div>
        <div style="font-size:14px;color:#94a3b8;margin-top:4px;">Enter your pond parameters for a detailed forecast.</div>
    </div>
    <hr style='margin:12px 0 20px 0;border:none;border-top:1px solid #23262f;'>
    """, unsafe_allow_html=True)

    pred = st.session_state.last_prediction
    st.markdown("<div style='background:rgba(22,24,31,0.85);border:1px solid #23262f;border-radius:14px;padding:20px;margin-bottom:20px;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:16px;font-weight:600;color:#e2e8f0;margin-bottom:16px;'>Forecast Parameters</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("Pond Area (ha)", value=st.session_state.pond_params["pond_area_ha"], key="fc_area")
        st.number_input("Stocking Count", value=st.session_state.pond_params["stocking_count"], key="fc_count")
        st.number_input("Initial Weight (g)", value=st.session_state.pond_params["initial_weight_g"], key="fc_weight")
    with c2:
        st.number_input("Culture Days", value=st.session_state.pond_params["culture_days"], key="fc_days")
        st.number_input("Mean Temp (\u00b0C)", value=st.session_state.pond_params["mean_temperature_c"], key="fc_temp")
        st.number_input("Mean DO (mg/L)", value=st.session_state.pond_params["mean_do_mg_l"], key="fc_do")
    with c3:
        st.number_input("Mean pH", value=st.session_state.pond_params["mean_ph"], key="fc_ph")
        st.selectbox("Season", ["summer", "winter", "monsoon"], index=0, key="fc_season")
        st.selectbox("Intensity", ["extensive", "semi-intensive", "intensive"], index=1, key="fc_intensity")

    if st.button("\U0001F504 Run Forecast", use_container_width=True, key="btn_run_forecast"):
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
        st.markdown("<div style='background:rgba(22,24,31,0.85);border:1px solid #23262f;border-radius:14px;padding:20px;'>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-size:32px;font-weight:700;color:#0d9488;'>"
            f"{pred['point_estimate_kg']:.1f} <span style='font-size:14px;font-weight:400;color:#94a3b8;'>kg</span></div>"
            f"<div style='font-size:11px;color:#94a3b8;margin-bottom:8px;font-weight:500;'>Point Estimate</div>"
            f"<div style='font-size:14px;color:#e2e8f0;font-weight:600;'>"
            f"{pred['lower_bound_kg']:.0f} - {pred['upper_bound_kg']:.0f} kg (90% CI)</div>",
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
