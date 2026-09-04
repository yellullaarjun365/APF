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

@st.cache_resource
def _clear_uploads_on_server_start():
    """Runs exactly once per server process (not per rerun/click, thanks
    to cache_resource), so a fresh `streamlit run` always starts with an
    empty uploads folder, but uploads survive normal reruns within the
    same running session."""
    for f in UPLOAD_DIR.glob("*"):
        f.unlink(missing_ok=True)
    return True

_clear_uploads_on_server_start()

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

    .block-container {{
        padding: 0 2rem 2rem 2rem !important;
        max-width: 1000px;
        min-height: 100vh !important;
        display: flex !important;
        flex-direction: column !important;
    }}
    /* block-container has exactly ONE direct child in Streamlit's DOM --
       the page's root div[data-testid="stVerticalBlock"] -- so the flex
       column above needs to apply *inside* that child, not at
       block-container's own level (a lone flex item doesn't grow to fill
       its parent's height unless told to). This is the actual container
       every top-level st.markdown()/st.container() call becomes a child
       of, and therefore the real flex parent for pinning the composer. */
    .block-container > div[data-testid="stVerticalBlock"] {{
        flex: 1 !important;
        min-height: 0 !important;
        display: flex !important;
        flex-direction: column !important;
    }}

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

    /* ---- Bottom input bar, pinned to the bottom of the page via flexbox
       instead of `position: fixed` -- Streamlit applies a CSS transform
       to an ancestor container, which resets the containing block for
       fixed-position children and silently breaks true viewport pinning.

       Previous bug: this used to be a raw HTML <div> opened in one
       st.markdown() call and closed in a LATER, separate st.markdown()
       call, with the buttons/form in between. That doesn't work --
       each st.markdown() call is its own independent HTML fragment as
       far as the browser's parser is concerned, so the unclosed <div>
       from the first call was auto-closed at the end of THAT fragment.
       It never actually wrapped the composer row below it; it just
       rendered as an empty strip wherever it fell in normal document
       flow (right under the header, since the chat was empty), and
       margin-top:auto did nothing because it wasn't a real flex child
       of anything.

       Fix: the composer is now a genuine st.container(key="composer")
       in the Python below, which is ONE real DOM node Streamlit tags
       with the class "st-key-composer" -- so it can actually be
       selected and pinned. The :has() rule below also covers Streamlit
       versions that nest that class one level deeper than expected,
       so this keeps working even if that DOM detail shifts. ---- */
    div[data-testid="stVerticalBlock"] > div:has(> .st-key-composer),
    div[data-testid="stVerticalBlock"] > div.st-key-composer {{
        margin-top: auto !important;
    }}
    .st-key-composer {{
        /* Previous attempt used a translucent dark gradient + blur to
           "merge" with the background -- but the bottom of the whale
           photo is already near-black deep water, so any dark overlay
           on top of already-near-black pixels still reads as solid
           black. There's nothing to blend into visually. Real fix: no
           panel at all -- let the photo show through untouched. The
           text input and buttons already carry their own #1a1d26 chip
           backgrounds (see rules below), so they stay legible floating
           directly over the water instead of sitting on a colored bar. */
        background: transparent;
        padding: 24px 0 18px 0;
        position: sticky;
        bottom: 0;
        z-index: 10;
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

    /* Attach control -- compact, full width of its own row (revealed only
       when the "+" toggle is clicked), styled dark instead of Streamlit's
       default light drag-drop panel. Trying to squeeze this into a narrow
       inline column breaks it (the native button keeps its natural width
       and overlaps neighboring widgets), so it gets its own row instead. */
    div[data-testid="stFileUploader"] {{
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        margin: 0 0 8px 0 !important;
    }}
    div[data-testid="stFileUploaderDropzone"] {{
        background: #1a1d26 !important;
        border: 1px dashed #2a2d3a !important;
        border-radius: 12px !important;
        padding: 6px 10px !important;
        min-height: 0 !important;
    }}
    div[data-testid="stFileUploaderDropzoneInstructions"] svg {{ width: 20px !important; height: 20px !important; }}
    div[data-testid="stFileUploaderDropzoneInstructions"] span {{ font-size: 12px !important; }}
    div[data-testid="stFileUploaderDropzoneInstructions"] small {{ font-size: 10px !important; }}
    /* Once a file is picked, Streamlit shows a separate file-list item
       below the dropzone (name/size chip + remove x) -- restyle that to
       match the dark theme instead of its default light card. */
    div[data-testid="stFileUploaderFile"] {{
        background: #1a1d26 !important;
        border: 1px solid #2a2d3a !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        margin-top: 8px !important;
    }}
    div[data-testid="stFileUploaderFile"] small {{ color: #94a3b8 !important; }}

    /* ---- Forecast page: number inputs, selects, buttons -- match the
       chat page's dark/teal theme instead of Streamlit's default light
       controls. ---- */
    div[data-testid="stNumberInput"] input,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
        background: #1a1d26 !important;
        border: 1px solid #2a2d3a !important;
        color: #e2e8f0 !important;
        border-radius: 10px !important;
    }}
    div[data-testid="stNumberInput"] button {{
        background: #23262f !important;
        border: 1px solid #2a2d3a !important;
    }}
    div[data-testid="stSelectbox"] svg {{ fill: #94a3b8 !important; }}
    div[data-testid="stWidgetLabel"] label p {{
        color: #94a3b8 !important; font-size: 12px !important; font-weight: 500 !important;
    }}
    div[data-testid="stButton"] button {{
        background: linear-gradient(135deg, #0d9488, #0f766e) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }}
    div[data-testid="stButton"] button:hover {{
        box-shadow: 0 2px 12px rgba(13, 148, 136, 0.4) !important;
    }}
    .fc-card {{
        background: rgba(15, 17, 23, 0.82);
        border: 1px solid #23262f;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
    }}
    .fc-hero {{
        background: linear-gradient(135deg, rgba(13,148,136,0.16), rgba(15,17,23,0.85));
        border: 1px solid rgba(13,148,136,0.35);
        border-radius: 16px;
        padding: 28px;
        box-shadow: 0 0 30px rgba(13,148,136,0.08);
    }}
    .fc-range-track {{
        position: relative;
        height: 6px;
        background: #23262f;
        border-radius: 3px;
        margin: 14px 0 6px 0;
    }}
    .fc-range-fill {{
        position: absolute;
        top: 0; bottom: 0;
        background: linear-gradient(90deg, #0d9488, #2dd4bf);
        border-radius: 3px;
    }}
    .fc-range-dot {{
        position: absolute;
        top: -4px;
        width: 14px; height: 14px;
        background: #ffffff;
        border: 3px solid #0d9488;
        border-radius: 50%;
        transform: translateX(-50%);
    }}


    /* Section labels */
    .section-label {{
        font-size: 10px; font-weight: 700; color: #94a3b8;
        text-transform: uppercase; letter-spacing: 0.8px;
        margin: 24px 0 10px 0;
    }}

    /* ---- Kill the rerun dim-flash ----
       Every button click (Run Forecast, send, attach toggle, etc.) triggers
       a full Streamlit script rerun. While that rerun is in flight,
       Streamlit tags the existing DOM with data-stale="true" and fades it
       to ~60% opacity as a "this content may be outdated" indicator -- that
       fade is the dimming you're seeing, not a bug in the app code. It's
       intentional default behavior, but with our own custom-styled UI it
       just reads as a flicker, so turn it off. */
    [data-stale="true"] {{
        opacity: 1 !important;
        transition: none !important;
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
        "show_attach": False,
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

    # ---- Temporary Storage review panel ----
    st.markdown('<div class="section-label">Temporary Storage</div>', unsafe_allow_html=True)
    uploads = sorted(UPLOAD_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True) if UPLOAD_DIR.exists() else []

    if not uploads:
        st.markdown("""
        <div style="padding:12px;background:rgba(30,33,43,0.8);border-radius:10px;border:1px solid #2a2d3a;font-size:12px;color:#64748b;">
            No files uploaded yet. Anything you attach in chat will appear here.
        </div>
        """, unsafe_allow_html=True)
    else:
        with st.expander(f"\U0001F4C1 {len(uploads)} file(s)", expanded=False):
            for f in uploads:
                size_kb = f.stat().st_size / 1024
                st.markdown(
                    f"<div style='font-size:12px;color:#e2e8f0;font-weight:500;margin-top:8px;'>{f.name}</div>"
                    f"<div style='font-size:11px;color:#64748b;margin-bottom:4px;'>{size_kb:.0f} KB</div>",
                    unsafe_allow_html=True,
                )
                if f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    st.image(str(f), use_container_width=True)
                elif f.suffix.lower() == ".csv":
                    try:
                        st.dataframe(pd.read_csv(f).head(5), use_container_width=True, height=150)
                    except Exception:
                        st.caption("(couldn't preview this CSV)")
                st.markdown("<hr style='margin:8px 0;border-color:#2a2d3a;'>", unsafe_allow_html=True)

            if st.button("\U0001F5D1 Clear all", key="btn_clear_uploads", use_container_width=True):
                for f in uploads:
                    f.unlink(missing_ok=True)
                st.rerun()

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
        <div style="font-size:22px;font-weight:700;color:#e2e8f0;">\U0001F4AC AquaLife Assistant</div>
        <div style="font-size:14px;color:#94a3b8;margin-top:4px;">Your intelligent companion for smarter aquaculture decisions.</div>
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

    # ---- FIXED BOTTOM INPUT BAR ----
    # A real st.container(key=...) -- ONE actual DOM node Streamlit tags
    # with class "st-key-composer" -- instead of a <div> opened in one
    # st.markdown() call and closed in a later, separate one (see the CSS
    # comment above for why that never worked).
    with st.container(key="composer"):
        outer = st.columns([0.6, 0.6, 4.8])

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
            if st.button("\u2795", key="btn_attach_toggle", help="Attach a file"):
                st.session_state.show_attach = not st.session_state.get("show_attach", False)

        with outer[2]:
            # Attach control (only rendered when toggled on -- its own row,
            # full width, so nothing overlaps) + text box + send button are
            # ALL inside the same form, so clear_on_submit empties every one
            # of them together after sending.
            with st.form(key="chat_form", clear_on_submit=True):
                attached_file = None
                if st.session_state.get("show_attach", False):
                    attached_file = st.file_uploader(
                        "Attach", type=["jpg", "jpeg", "png", "pdf", "csv", "wav", "mp3", "m4a"],
                        key="chat_attachment", label_visibility="collapsed",
                        accept_multiple_files=False,
                    )
                form_cols = st.columns([5.4, 0.6])
                with form_cols[0]:
                    user_text = st.text_input(
                        "", placeholder="Ask AquaPredict AI...", key="chat_input", label_visibility="collapsed",
                    )
                with form_cols[1]:
                    send_clicked = st.form_submit_button("\u27A4")

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
        st.session_state.show_attach = False
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

    st.markdown('<div class="fc-card">', unsafe_allow_html=True)
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

    st.markdown("<div style='margin-top:8px;'>", unsafe_allow_html=True)
    run_clicked = st.button("\U0001F504 Run Forecast", use_container_width=True, key="btn_run_forecast")
    st.markdown("</div>", unsafe_allow_html=True)

    if run_clicked:
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
            st.rerun()
        else:
            st.error("Error: " + str(result["error"]))
    st.markdown("</div>", unsafe_allow_html=True)

    if pred:
        pe, lb, ub = pred["point_estimate_kg"], pred["lower_bound_kg"], pred["upper_bound_kg"]
        dot_pct = 0 if ub == lb else (pe - lb) / (ub - lb) * 100
        fill_pct = 0 if ub == lb else (pe - lb) / (ub - lb) * 100

        st.markdown('<div class="fc-hero">', unsafe_allow_html=True)
        hc1, hc2 = st.columns([1.3, 1])
        with hc1:
            st.markdown(
                f"<div style='font-size:12px;color:#5eead4;font-weight:600;letter-spacing:0.5px;text-transform:uppercase;'>Point Estimate</div>"
                f"<div style='font-size:42px;font-weight:700;color:#ffffff;margin-top:4px;'>"
                f"{pe:.1f} <span style='font-size:16px;font-weight:400;color:#94a3b8;'>kg</span></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='fc-range-track'>"
                f"<div class='fc-range-fill' style='left:0%;width:{fill_pct:.1f}%;'></div>"
                f"<div class='fc-range-dot' style='left:{dot_pct:.1f}%;'></div>"
                f"</div>"
                f"<div style='display:flex;justify-content:space-between;font-size:12px;color:#94a3b8;'>"
                f"<span>{lb:.0f} kg</span><span>{ub:.0f} kg</span></div>"
                f"<div style='font-size:11px;color:#64748b;margin-top:4px;'>90% confidence interval</div>",
                unsafe_allow_html=True,
            )
        with hc2:
            factors = pred.get("top_factors", [])
            if factors:
                factors_html = "<div style='font-size:12px;color:#94a3b8;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:10px;'>Top Factors</div>"
                for f in factors[:5]:
                    imp = f.get("importance", 0)
                    bar_w = min(100, abs(imp) * 400)
                    factors_html += (
                        f"<div style='margin-bottom:8px;'>"
                        f"<div style='font-size:12px;color:#e2e8f0;'>{f['feature'].replace('_', ' ').title()}</div>"
                        f"<div style='height:4px;background:#23262f;border-radius:2px;margin-top:3px;'>"
                        f"<div style='height:4px;width:{bar_w:.0f}%;background:linear-gradient(90deg,#0d9488,#2dd4bf);border-radius:2px;'></div>"
                        f"</div></div>"
                    )
                st.markdown(factors_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="fc-card" style="margin-top:20px;">', unsafe_allow_html=True)
        st.markdown("<div style='font-size:14px;font-weight:600;color:#e2e8f0;margin-bottom:12px;'>Projected Growth Curve</div>", unsafe_allow_html=True)
        days = np.arange(0, st.session_state.pond_params["culture_days"] + 1, 5)
        t_norm = days / days[-1]
        biomass = pe * (3 * t_norm**2 - 2 * t_norm**3)
        upper = biomass * (ub / pe) if pe else biomass
        lower = biomass * (lb / pe) if pe else biomass
        chart_df = pd.DataFrame({"Day": days, "Forecast": biomass, "Upper": upper, "Lower": lower})
        st.line_chart(
            chart_df.set_index("Day")[["Forecast", "Upper", "Lower"]],
            color=["#2dd4bf", "#3a3f4d", "#3a3f4d"],
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
