"""APF V1 -- Streamlit Web UI (M4).
Single-page dashboard matching the reference design.
Compatible with Streamlit 1.63.0 and Python 3.10.

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
# Custom CSS
# ------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif !important;
}

.block-container {
    padding: 1rem 2rem !important;
    max-width: 1400px;
}

[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e8edf2;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 1.5rem 1rem !important;
}

.kpi-card {
    background: white;
    border-radius: 12px;
    padding: 1.25rem;
    text-align: center;
    border: 1px solid #f0f4f8;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

.chat-bot {
    background: #f0f4f8;
    border-radius: 12px 12px 12px 4px;
    padding: 0.75rem 1rem;
    color: #4a5568;
    font-size: 13px;
    line-height: 1.5;
    margin-bottom: 0.5rem;
}
.chat-user {
    background: #0d7377;
    border-radius: 12px 12px 4px 12px;
    padding: 0.75rem 1rem;
    color: white;
    font-size: 13px;
    line-height: 1.5;
    margin-bottom: 0.5rem;
    margin-left: auto;
}

.badge-good {
    background: #e8f5e9;
    color: #2e7d32;
    padding: 2px 10px;
    border-radius: 10px;
    font-size: 10px;
    font-weight: 600;
}
.badge-medium {
    background: #fff8e1;
    color: #c9a227;
    padding: 2px 10px;
    border-radius: 10px;
    font-size: 10px;
    font-weight: 600;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
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
        bg = "#0d7377" if active else "transparent"
        color = "white" if active else "#4a5568"
        weight = "500" if active else "400"
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
        bg = "#0d7377" if active else "transparent"
        color = "white" if active else "#4a5568"
        weight = "500" if active else "400"
        badge_html = ""
        if badge:
            badge_html = "<span style='background:#e8f0e8; color:#0d7377; font-size:9px; padding:2px 6px; border-radius:10px; margin-left:auto;'>New</span>"
        html = (
            "<div style='padding:8px 10px; border-radius:8px; background:" + bg +
            "; color:" + color + "; font-size:13px; font-weight:" + weight +
            "; margin-bottom:2px; display:flex; justify-content:space-between; align-items:center;'>"
            "<span>" + icon + " " + label + "</span>" + badge_html + "</div>"
        )
        st.markdown(html, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:auto; padding-top:2rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='display:flex; align-items:center; gap:8px; padding:10px; background:#f8fafc; border-radius:8px;'>"
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
        "<div style='font-size:20px; font-weight:700; color:#1a2e35;'>Good morning, Farmer! 👋</div>"
        "<div style='font-size:13px; color:#6b7c93; margin-top:2px;'>Here's the overview of your farm today.</div>",
        unsafe_allow_html=True,
    )
with top_col2:
    st.markdown(
        "<div style='display:flex; justify-content:flex-end; align-items:center; gap:16px;'>"
        "<div style='text-align:center;'>"
        "<div style='font-size:20px;'>🌤️</div>"
        "<div style='font-size:12px; font-weight:600; color:#1a2e35;'>28°C</div>"
        "<div style='font-size:10px; color:#6b7c93;'>Partly Cloudy</div></div>"
        "<div style='display:flex; align-items:center; gap:8px;'>"
        "<div style='width:32px; height:32px; background:#e8f0e8; border-radius:50%; display:flex; align-items:center; justify-content:center;'>👤</div>"
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
            "<div style='font-size:24px; margin-bottom:4px;'>💧</div>"
            "<div style='font-size:11px; color:#6b7c93;'>Total Ponds</div>"
            "<div style='font-size:22px; font-weight:700; color:#1a2e35;'>8</div>"
            "<div style='font-size:10px; color:#0d7377;'>Active: 6</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with kpi2:
        st.markdown(
            "<div class='kpi-card'>"
            "<div style='font-size:24px; margin-bottom:4px;'>🌱</div>"
            "<div style='font-size:11px; color:#6b7c93;'>Active Cultures</div>"
            "<div style='font-size:22px; font-weight:700; color:#1a2e35;'>6</div>"
            "<div style='font-size:10px; color:#0d7377;'>In progress</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with kpi3:
        est_val = "{:.1f}".format(pred["point_estimate_kg"]) if pred else "—"
        est_unit = "kg" if pred else ""
        st.markdown(
            "<div class='kpi-card'>"
            "<div style='font-size:24px; margin-bottom:4px;'>⚖️</div>"
            "<div style='font-size:11px; color:#6b7c93;'>Est. Total Production</div>"
            "<div style='font-size:22px; font-weight:700; color:#1a2e35;'>" + est_val +
            " <span style='font-size:12px; font-weight:400; color:#6b7c93;'>" + est_unit + "</span></div>"
            "<div style='font-size:10px; color:#6b7c93;'>All active ponds</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with kpi4:
        st.markdown(
            "<div class='kpi-card'>"
            "<div style='font-size:24px; margin-bottom:4px;'>📅</div>"
            "<div style='font-size:11px; color:#6b7c93;'>Avg. Culture Day</div>"
            "<div style='font-size:22px; font-weight:700; color:#1a2e35;'>72</div>"
            "<div style='font-size:10px; color:#6b7c93;'>days</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)

    # --- Production Forecast Panel ---
    st.markdown("<div style='background:white; border-radius:12px; padding:1.5rem; border:1px solid #f0f4f8;'>", unsafe_allow_html=True)

    fcol1, fcol2 = st.columns([1, 2])
    with fcol1:
        if pred:
            lower = "{:.0f}".format(pred["lower_bound_kg"])
            upper = "{:.0f}".format(pred["upper_bound_kg"])
            st.markdown(
                "<div style='font-size:32px; font-weight:700; color:#0d7377;'>" +
                "{:.1f}".format(pred["point_estimate_kg"]) +
                " <span style='font-size:14px; font-weight:400; color:#6b7c93;'>kg</span></div>"
                "<div style='font-size:11px; color:#6b7c93; margin-bottom:8px;'>Total Estimated Production</div>"
                "<div style='font-size:13px; color:#1a2e35; font-weight:600;'>" + lower + " – " + upper + " kg</div>"
                "<div style='font-size:10px; color:#6b7c93; margin-bottom:8px;'>Prediction Range</div>"
                "<div style='display:flex; align-items:center; gap:4px;'>"
                "<span style='font-size:12px;'>⚡</span>"
                "<span style='font-size:11px; color:#c9a227; font-weight:600;'>Medium Confidence</span></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='font-size:32px; font-weight:700; color:#0d7377;'>— <span style='font-size:14px; font-weight:400; color:#6b7c93;'>kg</span></div>"
                "<div style='font-size:11px; color:#6b7c93; margin-bottom:8px;'>Run a forecast to see results</div>",
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
                color=["#0d7377", "#e6f3f3", "#e6f3f3"],
                use_container_width=True,
            )
        else:
            st.markdown(
                "<div style='background:#f8fafc; border-radius:8px; padding:2rem; text-align:center; color:#6b7c93; font-size:13px;'>"
                "Enter pond details in the AI Assistant panel and click <b>Forecast</b> to see the production curve."
                "</div>",
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)

    # --- Pond Overview Table ---
    st.markdown("<div style='background:white; border-radius:12px; padding:1.25rem; border:1px solid #f0f4f8;'>", unsafe_allow_html=True)
    st.markdown(
        "<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;'>"
        "<div style='font-size:14px; font-weight:700; color:#1a2e35;'>Pond Overview</div>"
        "<div style='font-size:11px; color:#0d7377;'>View all ponds →</div></div>",
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
        "<div style='background:white; border-radius:12px; padding:1rem; border:1px solid #f0f4f8;'>"
        "<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;'>"
        "<div style='font-size:14px; font-weight:700; color:#1a2e35;'>AI Assistant</div>"
        "<div style='display:flex; gap:8px;'>"
        "<span style='font-size:11px; background:#f0f4f8; padding:2px 8px; border-radius:10px; color:#4a5568;'>New Chat</span>"
        "<span style='font-size:14px; color:#6b7c93;'>✕</span></div></div>",
        unsafe_allow_html=True,
    )

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "assistant":
                st.markdown(
                    "<div style='display:flex; gap:8px; align-items:flex-start; margin-bottom:8px;'>"
                    "<div style='width:28px; height:28px; background:#e8f0e8; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:12px; flex-shrink:0;'>🤖</div>"
                    "<div class='chat-bot'>" + msg["content"] + "</div></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div style='display:flex; gap:8px; align-items:flex-start; margin-bottom:8px; flex-direction:row-reverse;'>"
                    "<div style='width:28px; height:28px; background:#0d7377; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:12px; flex-shrink:0; color:white;'>👤</div>"
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
                        "<div style='background:white; border-radius:8px; padding:12px; margin-top:8px; border:1px solid #e2e8f0;'>"
                        "<div style='font-size:11px; color:#6b7c93; margin-bottom:4px;'>Estimated Production</div>"
                        "<div style='font-size:24px; font-weight:700; color:#1a2e35;'>" + "{:.1f}".format(pe) +
                        " <span style='font-size:12px; font-weight:400;'>kg</span></div>"
                        "<div style='font-size:11px; color:#6b7c93; margin:4px 0;'>Range: " +
                        "{:.0f}".format(lb) + " – " + "{:.0f}".format(ub) + " kg</div>"
                        "<div style='font-size:11px; color:#6b7c93;'>Confidence: <span style='color:#c9a227; font-weight:600;'>Medium</span></div>"
                        "<div style='font-size:11px; color:#1a2e35; margin-top:8px; font-weight:600;'>Key Factors:</div>"
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

    # --- Water Quality Panel ---
    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='background:white; border-radius:12px; padding:1rem; border:1px solid #f0f4f8;'>"
        "<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;'>"
        "<div style='font-size:14px; font-weight:700; color:#1a2e35;'>Water Quality <span style='font-size:10px; color:#6b7c93; font-weight:400;'>(Live)</span></div>"
        "<div style='background:#f0f4f8; border-radius:6px; padding:2px 8px; font-size:11px; color:#4a5568;'>Pond 1 ▼</div></div>"
        "</div>",
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
            "<div style='display:flex; justify-content:space-between; align-items:center; padding:6px 0; border-bottom:1px solid #f8fafc;'>"
            "<div style='display:flex; align-items:center; gap:8px;'>"
            "<span style='font-size:14px;'>" + icon + "</span><span style='font-size:12px; color:#4a5568;'>" + label + "</span></div>"
            "<div style='display:flex; align-items:center; gap:8px;'>"
            "<span style='font-size:12px; font-weight:600; color:#1a2e35;'>" + value + "</span>"
            "<span class='" + badge_class + "'>" + status + "</span></div></div>",
            unsafe_allow_html=True,
        )

    # --- Risk Assessment ---
    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='background:white; border-radius:12px; padding:1rem; border:1px solid #f0f4f8; text-align:center;'>"
        "<div style='font-size:14px; font-weight:700; color:#1a2e35; margin-bottom:12px; text-align:left;'>Risk Assessment</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <svg width="140" height="80" viewBox="0 0 140 80" style="margin:0 auto; display:block;">
          <path d="M 10 70 A 60 60 0 0 1 130 70" fill="none" stroke="#e2e8f0" stroke-width="12" stroke-linecap="round"/>
          <path d="M 10 70 A 60 60 0 0 1 50 18" fill="none" stroke="#4caf50" stroke-width="12" stroke-linecap="round"/>
          <path d="M 50 18 A 60 60 0 0 1 90 18" fill="none" stroke="#ffc107" stroke-width="12" stroke-linecap="round"/>
          <path d="M 90 18 A 60 60 0 0 1 130 70" fill="none" stroke="#f44336" stroke-width="12" stroke-linecap="round"/>
          <line x1="70" y1="70" x2="35" y2="35" stroke="#1a2e35" stroke-width="3" stroke-linecap="round"/>
          <circle cx="70" cy="70" r="5" fill="#1a2e35"/>
        </svg>
        <div style="font-size:16px; font-weight:700; color:#2e7d32; margin-top:8px;">Low Risk</div>
        <div style="font-size:11px; color:#6b7c93;">Overall Pond Risk</div>
        <div style="font-size:11px; color:#0d7377; margin-top:8px;">View risk details →</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.markdown(
    "<div style='text-align:center; margin-top:2rem; padding-top:1rem; border-top:1px solid #f0f4f8;'>"
    "<span style='font-size:10px; color:#6b7c93;'>All times are in your local timezone.</span></div>",
    unsafe_allow_html=True,
)
