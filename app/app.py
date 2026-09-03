"""APF V1 -- Streamlit Web UI (M4).
Light, warm color palette matching the reference design.

Run:  streamlit run app/app.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from features.build_features import build_features

# ------------------------------------------------------------------
# API config
# ------------------------------------------------------------------
API_BASE = os.environ.get("APF_API_URL", "http://localhost:8000")

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="AquaPredict AI – Smarter Aquaculture",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Custom CSS — warm light palette
# ------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif !important;
}

/* Light warm background */
[data-testid="stAppViewContainer"] {
    background: #f0f4f3 !important;
}

.block-container {
    padding: 1rem 2rem !important;
    max-width: 1400px;
}

/* Sidebar — clean white with warm border */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e0e8e6;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 1.5rem 1rem !important;
}

/* KPI cards — white with soft shadow */
.kpi-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 1.25rem;
    text-align: center;
    border: 1px solid #e8eeec;
    box-shadow: 0 2px 8px rgba(13, 138, 138, 0.06);
}

/* Chat bubbles */
.chat-bot {
    background: #f0f4f3;
    border-radius: 14px 14px 14px 4px;
    padding: 0.85rem 1rem;
    color: #2d3e3c;
    font-size: 13px;
    line-height: 1.5;
    margin-bottom: 0.5rem;
    border: 1px solid #e0e8e6;
}
.chat-user {
    background: #0d8a8a;
    border-radius: 14px 14px 4px 14px;
    padding: 0.85rem 1rem;
    color: #ffffff;
    font-size: 13px;
    line-height: 1.5;
    margin-bottom: 0.5rem;
    margin-left: auto;
}

/* Status badges */
.badge-good {
    background: #e6f4ea;
    color: #1e7e34;
    padding: 3px 12px;
    border-radius: 12px;
    font-size: 10px;
    font-weight: 600;
}
.badge-medium {
    background: #fff4e1;
    color: #b8860b;
    padding: 3px 12px;
    border-radius: 12px;
    font-size: 10px;
    font-weight: 600;
}
.badge-bad {
    background: #fce8e6;
    color: #c62828;
    padding: 3px 12px;
    border-radius: 12px;
    font-size: 10px;
    font-weight: 600;
}

/* Panel cards */
.panel-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 1.25rem;
    border: 1px solid #e8eeec;
    box-shadow: 0 2px 8px rgba(13, 138, 138, 0.05);
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Button styling */
button[kind="secondary"] {
    background: #f0f4f3 !important;
    border: 1px solid #d0ddd9 !important;
    border-radius: 20px !important;
    color: #2d3e3c !important;
    font-size: 12px !important;
}
button[kind="secondary"]:hover {
    background: #e0e8e6 !important;
}

/* Primary button */
button[kind="primary"] {
    background: #0d8a8a !important;
    border-radius: 20px !important;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------
def init_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": "Hello! I'm your Aqua AI Assistant. You can ask me anything about your pond, production forecast, water quality, feeding, and more. How can I help you today?",
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

init_state()

# ------------------------------------------------------------------
# API helpers
# ------------------------------------------------------------------
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


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("<div style='font-size:28px;'>🐟</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div style='font-size:16px; font-weight:700; color:#1a2e35;'>AquaPredict AI</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:10px; color:#6b7c93;'>Smarter Aquaculture</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

    st.markdown("<div style='font-size:10px; font-weight:700; color:#6b7c93; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:8px;'>Farm Management</div>", unsafe_allow_html=True)

    nav_items = [
        ("🏠", "Dashboard", True),
        ("💧", "Ponds", False),
        ("🔄", "Culture Cycles", False),
        ("📏", "Measurements", False),
        ("🍽️", "Feed Management", False),
    ]
    for icon, label, active in nav_items:
        bg = "#0d8a8a" if active else "transparent"
        color = "#ffffff" if active else "#4a5568"
        weight = "600" if active else "400"
        html = (
            "<div style='padding:8px 10px; border-radius:8px; background:" + bg +
            "; color:" + color + "; font-size:13px; font-weight:" + weight +
            "; margin-bottom:2px;'>" + icon + " " + label + "</div>"
        )
        st.markdown(html, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

    st.markdown("<div style='font-size:10px; font-weight:700; color:#6b7c93; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:8px;'>AI Tools</div>", unsafe_allow_html=True)

    ai_items = [
        ("🤖", "AI Assistant", True, "New"),
        ("📊", "Production Forecast", False, None),
        ("🔧", "What-if Simulator", False, None),
        ("⚠️", "Risk Assessment", False, None),
    ]
    for icon, label, active, badge in ai_items:
        bg = "#0d8a8a" if active else "transparent"
        color = "#ffffff" if active else "#4a5568"
        weight = "600" if active else "400"
        badge_html = ""
        if badge:
            badge_html = "<span style='background:#e6f4f4; color:#0d8a8a; font-size:9px; padding:2px 8px; border-radius:10px; margin-left:auto;'>New</span>"
        html = (
            "<div style='padding:8px 10px; border-radius:8px; background:" + bg +
            "; color:" + color + "; font-size:13px; font-weight:" + weight +
            "; margin-bottom:2px; display:flex; justify-content:space-between; align-items:center;'>"
            "<span>" + icon + " " + label + "</span>" + badge_html + "</div>"
        )
        st.markdown(html, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:auto; padding-top:2rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='display:flex; align-items:center; gap:8px; padding:10px; background:#f5f9f8; border-radius:10px;'>"
        "<span style='font-size:18px;'>🎧</span>"
        "<div><div style='font-size:12px; font-weight:600; color:#1a2e35;'>Need help?</div>"
        "<div style='font-size:10px; color:#6b7c93;'>Contact Support</div></div></div>",
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------
# Top bar
# ------------------------------------------------------------------
top_col1, top_col2 = st.columns([3, 1])
with top_col1:
    st.markdown(
        "<div style='font-size:22px; font-weight:700; color:#1a2e35;'>Good morning, Farmer! 👋</div>"
        "<div style='font-size:13px; color:#6b7c93; margin-top:4px;'>Here is the overview of your farm today.</div>",
        unsafe_allow_html=True,
    )
with top_col2:
    st.markdown(
        "<div style='display:flex; justify-content:flex-end; align-items:center; gap:16px;'>"
        "<div style='text-align:center;'>"
        "<div style='font-size:22px;'>🌤️</div>"
        "<div style='font-size:12px; font-weight:600; color:#1a2e35;'>28°C</div>"
        "<div style='font-size:10px; color:#6b7c93;'>Partly Cloudy</div></div>"
        "<div style='display:flex; align-items:center; gap:8px;'>"
        "<div style='width:32px; height:32px; background:#e6f4f4; border-radius:50%; display:flex; align-items:center; justify-content:center;'>👤</div>"
        "<div><div style='font-size:12px; font-weight:600; color:#1a2e35;'>Farmer</div>"
        "<div style='font-size:10px; color:#6b7c93;'>Tilapia Farm</div></div></div></div>",
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Main layout
# ------------------------------------------------------------------
left_col, center_col, right_col = st.columns([1.2, 2.5, 1.3])

# ==================== LEFT ====================
with left_col:
    pass

# ==================== CENTER ====================
with center_col:
    # --- KPI Cards ---
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    pred = st.session_state.last_prediction

    with kpi1:
        st.markdown(
            "<div class='kpi-card'>"
            "<div style='font-size:28px; margin-bottom:6px;'>💧</div>"
            "<div style='font-size:11px; color:#6b7c93; font-weight:500;'>Total Ponds</div>"
            "<div style='font-size:24px; font-weight:700; color:#1a2e35; margin-top:4px;'>8</div>"
            "<div style='font-size:11px; color:#0d8a8a; font-weight:500;'>Active: 6</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with kpi2:
        st.markdown(
            "<div class='kpi-card'>"
            "<div style='font-size:28px; margin-bottom:6px;'>🌱</div>"
            "<div style='font-size:11px; color:#6b7c93; font-weight:500;'>Active Cultures</div>"
            "<div style='font-size:24px; font-weight:700; color:#1a2e35; margin-top:4px;'>6</div>"
            "<div style='font-size:11px; color:#0d8a8a; font-weight:500;'>In progress</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with kpi3:
        est_val = "{:.1f}".format(pred["point_estimate_kg"]) if pred else "—"
        est_unit = "kg" if pred else ""
        st.markdown(
            "<div class='kpi-card'>"
            "<div style='font-size:28px; margin-bottom:6px;'>⚖️</div>"
            "<div style='font-size:11px; color:#6b7c93; font-weight:500;'>Est. Total Production</div>"
            "<div style='font-size:24px; font-weight:700; color:#1a2e35; margin-top:4px;'>" + est_val +
            " <span style='font-size:12px; font-weight:400; color:#6b7c93;'>" + est_unit + "</span></div>"
            "<div style='font-size:11px; color:#6b7c93; font-weight:500;'>All active ponds</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with kpi4:
        st.markdown(
            "<div class='kpi-card'>"
            "<div style='font-size:28px; margin-bottom:6px;'>📅</div>"
            "<div style='font-size:11px; color:#6b7c93; font-weight:500;'>Avg. Culture Day</div>"
            "<div style='font-size:24px; font-weight:700; color:#1a2e35; margin-top:4px;'>72</div>"
            "<div style='font-size:11px; color:#6b7c93; font-weight:500;'>days</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)

    # --- Production Forecast Panel ---
    st.markdown("<div class='panel-card'>", unsafe_allow_html=True)

    fcol1, fcol2 = st.columns([1, 2])
    with fcol1:
        if pred:
            lower = "{:.0f}".format(pred["lower_bound_kg"])
            upper = "{:.0f}".format(pred["upper_bound_kg"])
            st.markdown(
                "<div style='font-size:32px; font-weight:700; color:#0d8a8a;'>" +
                "{:.1f}".format(pred["point_estimate_kg"]) +
                " <span style='font-size:14px; font-weight:400; color:#6b7c93;'>kg</span></div>"
                "<div style='font-size:11px; color:#6b7c93; margin-bottom:8px; font-weight:500;'>Total Estimated Production</div>"
                "<div style='font-size:14px; color:#1a2e35; font-weight:600;'>" + lower + " – " + upper + " kg</div>"
                "<div style='font-size:10px; color:#6b7c93; margin-bottom:8px; font-weight:500;'>Prediction Range</div>"
                "<div style='display:flex; align-items:center; gap:6px;'>"
                "<span style='font-size:14px;'>⚡</span>"
                "<span style='font-size:12px; color:#c9a227; font-weight:600;'>Medium Confidence</span></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='font-size:32px; font-weight:700; color:#0d8a8a;'>— <span style='font-size:14px; font-weight:400; color:#6b7c93;'>kg</span></div>"
                "<div style='font-size:11px; color:#6b7c93; margin-bottom:8px; font-weight:500;'>Run a forecast to see results</div>",
                unsafe_allow_html=True,
            )

    with fcol2:
        if pred:
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
                color=["#0d8a8a", "#d0e8e8", "#d0e8e8"],
                use_container_width=True,
            )
        else:
            st.markdown(
                "<div style='background:#f5f9f8; border-radius:10px; padding:2rem; text-align:center; color:#6b7c93; font-size:13px;'>"
                "Enter pond details in the AI Assistant panel and click <b>Forecast</b> to see the production curve."
                "</div>",
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)

    # --- Pond Overview Table ---
    st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
    st.markdown(
        "<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;'>"
        "<div style='font-size:15px; font-weight:700; color:#1a2e35;'>Pond Overview</div>"
        "<div style='font-size:11px; color:#0d8a8a; font-weight:500; cursor:pointer;'>View all ponds →</div></div>",
        unsafe_allow_html=True,
    )

    if pred:
        est = "{:.1f}".format(pred["point_estimate_kg"])
        low = "{:.0f}".format(pred["lower_bound_kg"])
        up = "{:.0f}".format(pred["upper_bound_kg"])
        days_str = str(st.session_state.pond_params["culture_days"]) + " days"
        pond_data = {
            "Pond ID": ["🐟 Pond 1"],
            "Species": ["Nile Tilapia"],
            "Culture Day": [days_str],
            "Est. Production": [est + " kg\n(" + low + " – " + up + ")"],
            "Status": ['<span class="badge-good">Good</span>'],
            "Action": ["View →"],
        }
        df_pond = pd.DataFrame(pond_data)
        st.markdown(
            df_pond.to_html(escape=False, index=False),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='text-align:center; padding:2rem; color:#6b7c93; font-size:13px;'>No forecast data yet. Use the AI Assistant to generate a prediction.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ==================== RIGHT ====================
with right_col:
    # --- AI Assistant Chat ---
    st.markdown(
        "<div class='panel-card'>"
        "<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;'>"
        "<div style='font-size:15px; font-weight:700; color:#1a2e35;'>AI Assistant</div>"
        "<div style='display:flex; gap:8px;'>"
        "<span style='font-size:11px; background:#f0f4f3; padding:3px 10px; border-radius:10px; color:#4a5568; font-weight:500; cursor:pointer;'>New Chat</span>"
        "<span style='font-size:14px; color:#6b7c93; cursor:pointer;'>✕</span></div></div>",
        unsafe_allow_html=True,
    )

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "assistant":
                st.markdown(
                    "<div style='display:flex; gap:8px; align-items:flex-start; margin-bottom:8px;'>"
                    "<div style='width:28px; height:28px; background:#e6f4f4; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:12px; flex-shrink:0;'>🤖</div>"
                    "<div class='chat-bot'>" + msg["content"] + "</div></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div style='display:flex; gap:8px; align-items:flex-start; margin-bottom:8px; flex-direction:row-reverse;'>"
                    "<div style='width:28px; height:28px; background:#0d8a8a; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:12px; flex-shrink:0; color:white;'>👤</div>"
                    "<div class='chat-user'>" + msg["content"] + "</div></div>",
                    unsafe_allow_html=True,
                )

    # Quick action pills
    pill_col1, pill_col2, pill_col3 = st.columns(3)
    with pill_col1:
        if st.button("Explain more", key="pill_explain", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": "Explain more"})
            if pred:
                factors = pred.get("top_factors", [])
                if factors:
                    top = factors[0]
                    dir_text = "increases" if top["impact_kg"] > 0 else "reduces"
                    resp = (
                        "The biggest driver is <b>" + top["feature"] + "</b>, which " + dir_text +
                        " the estimate by about " + str(abs(int(top["impact_kg"]))) + " kg. "
                        "Water quality (DO, pH, temperature) and stocking density are the next most important factors."
                    )
                else:
                    resp = "The model weighs pond area, stocking density, culture duration, and water quality parameters to arrive at this forecast."
            else:
                resp = "Please run a forecast first so I can explain the specific factors driving your prediction."
            st.session_state.chat_history.append({"role": "assistant", "content": resp})
            st.rerun()
    with pill_col2:
        if st.button("What can I improve?", key="pill_improve", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": "What can I improve?"})
            resp = (
                "Based on typical tilapia production: 1) Keep DO above 5 mg/L — hypoxia is the #1 yield killer. "
                "2) Maintain temperature between 26–30°C for optimal growth. 3) Use 28–32% protein feed for semi-intensive systems. "
                "4) Monitor pH weekly — keep it between 6.8–8.2."
            )
            st.session_state.chat_history.append({"role": "assistant", "content": resp})
            st.rerun()
    with pill_col3:
        if st.button("Show graph", key="pill_graph", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": "Show graph"})
            resp = "The production forecast graph is displayed in the center panel above. It shows the projected biomass curve over your culture period."
            st.session_state.chat_history.append({"role": "assistant", "content": resp})
            st.rerun()

    # Text input + Send
    user_text = st.text_input("", placeholder="Type your message...", key="chat_input", label_visibility="collapsed")

    col_send, col_speak = st.columns([4, 1])
    with col_send:
        if st.button("➤ Send", use_container_width=True, key="btn_send"):
            if user_text.strip():
                st.session_state.chat_history.append({"role": "user", "content": user_text.strip()})
                result = _api_predict_text(user_text.strip())
                if result.get("status") == "incomplete":
                    missing = result.get("missing_fields", [])
                    followup = result.get("follow_up_question", "Could you provide more details?")
                    resp = followup + "<br><br><span style='font-size:11px; color:#6b7c93;'>Missing: " + ", ".join(missing) + "</span>"
                elif result.get("status") == "complete":
                    pred_data = result.get("prediction", {})
                    st.session_state.last_prediction = pred_data
                    st.session_state.pond_params.update(result.get("extracted", {}))
                    pe = pred_data.get("point_estimate_kg", 0)
                    lb = pred_data.get("lower_bound_kg", 0)
                    ub = pred_data.get("upper_bound_kg", 0)
                    resp = (
                        "Sure! Based on your inputs, here is the production forecast."
                        "<div style='background:#ffffff; border-radius:10px; padding:14px; margin-top:10px; border:1px solid #e0e8e6;'>"
                        "<div style='font-size:11px; color:#6b7c93; margin-bottom:4px; font-weight:500;'>Estimated Production</div>"
                        "<div style='font-size:26px; font-weight:700; color:#1a2e35;'>" + "{:.1f}".format(pe) +
                        " <span style='font-size:12px; font-weight:400;'>kg</span></div>"
                        "<div style='font-size:12px; color:#6b7c93; margin:4px 0; font-weight:500;'>Range: " +
                        "{:.0f}".format(lb) + " – " + "{:.0f}".format(ub) + " kg</div>"
                        "<div style='font-size:12px; color:#6b7c93; font-weight:500;'>Confidence: <span style='color:#c9a227; font-weight:600;'>Medium</span></div>"
                        "<div style='font-size:12px; color:#1a2e35; margin-top:10px; font-weight:600;'>Key Factors:</div>"
                        "<ul style='margin:4px 0; padding-left:16px; font-size:11px; color:#4a5568;'>"
                    )
                    factors = pred_data.get("top_factors", [])
                    for f in factors[:4]:
                        arrow = "↑" if f["impact_kg"] > 0 else "↓"
                        resp += "<li>" + f["feature"].replace("_", " ").title() + ": " + arrow + "</li>"
                    resp += "</ul></div>"
                elif "error" in result:
                    resp = "Sorry, I encountered an error: " + str(result["error"])
                else:
                    resp = "I'm not sure how to process that. Try describing your pond with area, stocking count, days, temperature, DO, and pH."
                st.session_state.chat_history.append({"role": "assistant", "content": resp})
                st.rerun()

    with col_speak:
        st.markdown("<div style='text-align:center; margin-top:4px;'><span style='font-size:10px; color:#6b7c93;'>🎤 Speak</span></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # --- Water Quality Panel ---
    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='panel-card'>"
        "<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;'>"
        "<div style='font-size:15px; font-weight:700; color:#1a2e35;'>Water Quality <span style='font-size:10px; color:#6b7c93; font-weight:400;'>(Live)</span></div>"
        "<div style='background:#f0f4f3; border-radius:8px; padding:3px 10px; font-size:11px; color:#4a5568; font-weight:500;'>Pond 1 ▼</div></div>",
        unsafe_allow_html=True,
    )

    params = st.session_state.pond_params
    wq_items = [
        ("🌡️", "Temperature", "{:.1f}".format(params["mean_temperature_c"]) + " °C", "Good"),
        ("⚗️", "pH", "{:.1f}".format(params["mean_ph"]), "Good"),
        ("💨", "Dissolved Oxygen", "{:.1f}".format(params["mean_do_mg_l"]) + " mg/L", "Good"),
    ]
    for icon, label, value, status in wq_items:
        badge_class = "badge-good" if status == "Good" else ("badge-medium" if status == "Medium" else "badge-bad")
        st.markdown(
            "<div style='display:flex; justify-content:space-between; align-items:center; padding:8px 0; border-bottom:1px solid #f0f4f3;'>"
            "<div style='display:flex; align-items:center; gap:10px;'>"
            "<span style='font-size:16px;'>" + icon + "</span><span style='font-size:13px; color:#4a5568; font-weight:500;'>" + label + "</span></div>"
            "<div style='display:flex; align-items:center; gap:10px;'>"
            "<span style='font-size:13px; font-weight:600; color:#1a2e35;'>" + value + "</span>"
            "<span class='" + badge_class + "'>" + status + "</span></div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # --- Risk Assessment ---
    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='panel-card' style='text-align:center;'>"
        "<div style='font-size:15px; font-weight:700; color:#1a2e35; margin-bottom:14px; text-align:left;'>Risk Assessment</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <svg width="160" height="90" viewBox="0 0 160 90" style="margin:0 auto; display:block;">
          <path d="M 15 78 A 65 65 0 0 1 145 78" fill="none" stroke="#e0e8e6" stroke-width="14" stroke-linecap="round"/>
          <path d="M 15 78 A 65 65 0 0 1 58 20" fill="none" stroke="#4caf50" stroke-width="14" stroke-linecap="round"/>
          <path d="M 58 20 A 65 65 0 0 1 102 20" fill="none" stroke="#ffc107" stroke-width="14" stroke-linecap="round"/>
          <path d="M 102 20 A 65 65 0 0 1 145 78" fill="none" stroke="#f44336" stroke-width="14" stroke-linecap="round"/>
          <line x1="80" y1="78" x2="40" y2="38" stroke="#1a2e35" stroke-width="3" stroke-linecap="round"/>
          <circle cx="80" cy="78" r="6" fill="#1a2e35"/>
        </svg>
        <div style="font-size:18px; font-weight:700; color:#2e7d32; margin-top:10px;">Low Risk</div>
        <div style="font-size:12px; color:#6b7c93; font-weight:500;">Overall Pond Risk</div>
        <div style="font-size:12px; color:#0d8a8a; margin-top:10px; font-weight:500; cursor:pointer;">View risk details →</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.markdown(
    "<div style='text-align:center; margin-top:2rem; padding-top:1rem; border-top:1px solid #e0e8e6;'>"
    "<span style='font-size:10px; color:#6b7c93; font-weight:500;'>All times are in your local timezone.</span></div>",
    unsafe_allow_html=True,
)
