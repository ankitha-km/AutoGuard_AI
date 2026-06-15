"""
AutoGuard AI: Vehicle Health & Predictive Maintenance Dashboard
A Streamlit application for real-time vehicle health monitoring,
failure risk prediction, RUL estimation, and explainable AI insights.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_preprocessing import preprocess_pipeline, FEATURE_COLUMNS
from predict_utils import (
    load_artifacts,
    predict_single,
    predict_batch,
    compute_health_score,
)

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="AutoGuard AI | Predictive Maintenance",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# CUSTOM CSS - MODERN DASHBOARD STYLING
# ----------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        color: #9aa5b1;
        font-size: 1.05rem;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e2530 0%, #2a3340 100%);
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        border: 1px solid #3a4555;
        box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    }
    .metric-title {
        color: #9aa5b1;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 2.0rem;
        font-weight: 800;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.9rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.03em;
    }
    .badge-critical { background: #ff4b4b; color: white; }
    .badge-high { background: #ff9f43; color: white; }
    .badge-moderate { background: #ffd43b; color: #333; }
    .badge-low { background: #51cf66; color: #003311; }

    .recommendation-box {
        background: #1e2530;
        border-left: 4px solid #0072ff;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
    }
    .issue-box {
        background: #2a1e1e;
        border-left: 4px solid #ff4b4b;
        border-radius: 8px;
        padding: 0.7rem 1.1rem;
        margin-bottom: 0.4rem;
        font-size: 0.95rem;
    }
    .action-box {
        background: #1e2a24;
        border-left: 4px solid #51cf66;
        border-radius: 8px;
        padding: 0.7rem 1.1rem;
        margin-bottom: 0.4rem;
        font-size: 0.95rem;
    }
    section[data-testid="stSidebar"] {
        background: #161a23;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# LOAD MODEL ARTIFACTS (cached)
# ----------------------------------------------------------------------
@st.cache_resource
def get_artifacts():
    return load_artifacts()


@st.cache_data
def get_sample_data():
    path = os.path.join(os.path.dirname(__file__), "data", "vehicle_sensor_data.csv")
    return pd.read_csv(path)


try:
    model, scaler, metadata = get_artifacts()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    LOAD_ERROR = str(e)

# ----------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------
st.markdown('<p class="main-header">🚗 AutoGuard AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Vehicle Health & Predictive Maintenance Dashboard — AI-powered failure risk, RUL & maintenance insights</p>', unsafe_allow_html=True)

if not MODEL_LOADED:
    st.error(f"⚠️ Model artifacts not found. Please run `python src/train_models.py` first.\n\nError: {LOAD_ERROR}")
    st.stop()

# ----------------------------------------------------------------------
# SIDEBAR - INPUT MODE
# ----------------------------------------------------------------------
st.sidebar.markdown("## ⚙️ Input Mode")
input_mode = st.sidebar.radio(
    "Choose how to provide vehicle data:",
    ["📁 Upload CSV", "✍️ Manual Entry", "🎲 Try Sample Vehicle"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("## 🧠 Model Info")
st.sidebar.markdown(f"**Active Model:** `{metadata['best_model']}`")
best_metrics = metadata["metrics"][metadata["best_model"]]
st.sidebar.markdown(f"**Accuracy:** {best_metrics['Accuracy']:.2%}")
st.sidebar.markdown(f"**F1 Score:** {best_metrics['F1_Score']:.2%}")
st.sidebar.markdown(f"**ROC-AUC:** {best_metrics['ROC_AUC']:.2%}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Comparison")
comp_df = pd.DataFrame(metadata["metrics"]).T.round(3)
st.sidebar.dataframe(comp_df, use_container_width=True)


# ----------------------------------------------------------------------
# HELPER: HEALTH GAUGE
# ----------------------------------------------------------------------
def render_gauge(value, title, max_val=100, color_ranges=None):
    if color_ranges is None:
        color_ranges = [
            {"range": [0, 40], "color": "#ff4b4b"},
            {"range": [40, 70], "color": "#ffd43b"},
            {"range": [70, 100], "color": "#51cf66"},
        ]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, max_val]},
            "bar": {"color": "#0072ff"},
            "steps": [{"range": cr["range"], "color": cr["color"]} for cr in color_ranges],
            "threshold": {"line": {"color": "white", "width": 3}, "thickness": 0.8, "value": value},
        },
    ))
    fig.update_layout(height=260, margin=dict(t=50, b=10, l=20, r=20),
                       paper_bgcolor="rgba(0,0,0,0)", font={"color": "#e6e6e6"})
    return fig


def urgency_badge(urgency):
    classes = {
        "CRITICAL": "badge-critical",
        "HIGH": "badge-high",
        "MODERATE": "badge-moderate",
        "LOW": "badge-low",
    }
    icons = {"CRITICAL": "🔴", "HIGH": "🟠", "MODERATE": "🟡", "LOW": "🟢"}
    cls = classes.get(urgency, "badge-low")
    icon = icons.get(urgency, "")
    return f'<span class="badge {cls}">{icon} {urgency} RISK</span>'


def render_result(result, vehicle_label="Vehicle"):
    """Render a full prediction result block."""
    st.markdown(f"### 🔍 Prediction Results — {vehicle_label}")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.plotly_chart(render_gauge(result["health_score"], "Vehicle Health Score"), use_container_width=True)

    with col2:
        risk_pct = result["failure_probability"] * 100
        st.plotly_chart(render_gauge(
            risk_pct, "Failure Risk (%)",
            color_ranges=[
                {"range": [0, 30], "color": "#51cf66"},
                {"range": [30, 60], "color": "#ffd43b"},
                {"range": [60, 100], "color": "#ff4b4b"},
            ]
        ), use_container_width=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-title">Remaining Useful Life</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{result["rul"]["rul_days"]:.0f} days</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="color:#9aa5b1; margin-top:4px;">≈ {result["rul"]["rul_km"]:.0f} km</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-title">Predicted Status</div>', unsafe_allow_html=True)
        status_text = "⚠️ Failure Likely" if result["failure_prediction"] == 1 else "✅ Healthy"
        status_color = "#ff4b4b" if result["failure_prediction"] == 1 else "#51cf66"
        st.markdown(f'<div class="metric-value" style="color:{status_color}; font-size:1.4rem;">{status_text}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        rec = result["recommendation"]
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-title">Maintenance Urgency</div>', unsafe_allow_html=True)
        st.markdown(urgency_badge(rec["urgency"]), unsafe_allow_html=True)
        st.markdown(f'<div style="margin-top:10px; color:#cdd5dd; font-size:0.9rem;">{rec["urgency_message"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Recommendations & Explainability
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("#### 🛠️ Maintenance Recommendations")
        st.markdown("**Detected Issues:**")
        for issue in rec["issues"]:
            st.markdown(f'<div class="issue-box">⚠️ {issue}</div>', unsafe_allow_html=True)
        st.markdown("**Recommended Actions:**")
        for action in rec["actions"]:
            st.markdown(f'<div class="action-box">✅ {action}</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown("#### 🧩 Explainable AI — Why This Prediction?")
        st.markdown("Top factors influencing this prediction (importance × deviation from baseline):")
        exp_df = pd.DataFrame(result["explanations"])
        fig = px.bar(
            exp_df.sort_values("contribution_score"),
            x="contribution_score", y="feature", orientation="h",
            color="contribution_score", color_continuous_scale="OrRd",
            labels={"contribution_score": "Contribution Score", "feature": ""},
        )
        fig.update_layout(height=320, margin=dict(t=10, b=10, l=10, r=10),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font={"color": "#e6e6e6"}, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📋 View raw sensor values used"):
            sensor_data = {k: result["processed_row"][k] for k in
                            ["Engine_Temperature", "Vibration", "Battery_Voltage", "Engine_Load",
                             "RPM", "Oil_Quality_Index", "Mileage_km", "Vehicle_Age_Years"]}
            st.json({k: round(float(v), 2) for k, v in sensor_data.items()})


# ----------------------------------------------------------------------
# MODE 1: CSV UPLOAD
# ----------------------------------------------------------------------
if input_mode == "📁 Upload CSV":
    st.markdown("### 📁 Batch Prediction via CSV Upload")
    st.markdown(
        "Upload a CSV with columns: `Engine_Temperature`, `Vibration`, `Battery_Voltage`, "
        "`Engine_Load`, and optionally `RPM`, `Oil_Quality_Index`, `Mileage_km`, `Vehicle_Age_Years`."
    )

    uploaded_file = st.file_uploader("Upload sensor data CSV", type=["csv"])

    sample_csv = get_sample_data().drop(columns=["Failure_Status"]).head(20)
    st.download_button(
        "⬇️ Download Sample CSV Template",
        data=sample_csv.to_csv(index=False),
        file_name="sample_vehicle_data.csv",
        mime="text/csv",
    )

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            required = ["Engine_Temperature", "Vibration", "Battery_Voltage", "Engine_Load"]
            missing = [c for c in required if c not in input_df.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                results_df = predict_batch(input_df, model, scaler, metadata)

                st.success(f"✅ Processed {len(results_df)} vehicle records.")

                # Summary metrics
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Vehicles", len(results_df))
                c2.metric("At Risk (Failure Predicted)", int(results_df["Failure_Prediction"].sum()))
                c3.metric("Avg Health Score", f"{results_df['Health_Score'].mean():.1f}")
                c4.metric("Avg RUL (days)", f"{results_df['RUL_Days'].mean():.1f}")

                st.markdown("#### 📋 Prediction Results Table")
                display_cols = (["Vehicle_ID"] if "Vehicle_ID" in results_df.columns else []) + [
                    "Engine_Temperature", "Vibration", "Battery_Voltage", "Engine_Load",
                    "Health_Score", "Failure_Probability", "Failure_Prediction", "RUL_Days", "RUL_Km"
                ]
                st.dataframe(results_df[display_cols], use_container_width=True, height=350)

                # Charts
                col_a, col_b = st.columns(2)
                with col_a:
                    fig = px.histogram(results_df, x="Health_Score", nbins=20,
                                        title="Health Score Distribution",
                                        color_discrete_sequence=["#0072ff"])
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#e6e6e6"})
                    st.plotly_chart(fig, use_container_width=True)
                with col_b:
                    fig2 = px.scatter(results_df, x="Engine_Temperature", y="Vibration",
                                       color="Failure_Probability", size="Health_Score",
                                       color_continuous_scale="RdYlGn_r",
                                       title="Temperature vs Vibration (colored by Risk)")
                    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#e6e6e6"})
                    st.plotly_chart(fig2, use_container_width=True)

                st.download_button(
                    "⬇️ Download Predictions CSV",
                    data=results_df.to_csv(index=False),
                    file_name="autoguard_predictions.csv",
                    mime="text/csv",
                )

                # Detailed view for a selected vehicle
                st.markdown("---")
                st.markdown("### 🔎 Inspect Individual Vehicle")
                if "Vehicle_ID" in results_df.columns:
                    selected_id = st.selectbox("Select Vehicle ID", results_df["Vehicle_ID"].tolist())
                    selected_idx = results_df[results_df["Vehicle_ID"] == selected_id].index[0]
                else:
                    selected_idx = st.number_input("Select Row Index", 0, len(results_df) - 1, 0)

                selected_row = input_df.iloc[selected_idx].to_dict()
                single_result = predict_single(selected_row, model, scaler, metadata)
                render_result(single_result, vehicle_label=str(selected_id) if "Vehicle_ID" in results_df.columns else f"Row {selected_idx}")

        except Exception as e:
            st.error(f"Error processing file: {e}")

# ----------------------------------------------------------------------
# MODE 2: MANUAL ENTRY
# ----------------------------------------------------------------------
elif input_mode == "✍️ Manual Entry":
    st.markdown("### ✍️ Manual Sensor Data Entry")
    st.markdown("Enter current vehicle sensor readings to get instant predictions.")

    with st.form("manual_entry_form"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            temp = st.slider("Engine Temperature (°C)", 50.0, 140.0, 90.0, 0.5)
            rpm = st.slider("Engine RPM", 600, 6000, 2200, 50)
        with col2:
            vibration = st.slider("Vibration (mm/s RMS)", 0.0, 15.0, 2.5, 0.1)
            oil_quality = st.slider("Oil Quality Index", 0, 100, 75, 1)
        with col3:
            voltage = st.slider("Battery Voltage (V)", 9.0, 15.0, 13.0, 0.1)
            mileage = st.number_input("Mileage (km)", 0, 400000, 40000, 1000)
        with col4:
            load = st.slider("Engine Load (%)", 0, 100, 50, 1)
            age = st.slider("Vehicle Age (years)", 0.0, 20.0, 3.0, 0.5)

        submitted = st.form_submit_button("🔍 Analyze Vehicle Health", use_container_width=True)

    if submitted:
        input_dict = {
            "Engine_Temperature": temp,
            "Vibration": vibration,
            "Battery_Voltage": voltage,
            "Engine_Load": load,
            "RPM": rpm,
            "Oil_Quality_Index": oil_quality,
            "Mileage_km": mileage,
            "Vehicle_Age_Years": age,
        }
        result = predict_single(input_dict, model, scaler, metadata)
        render_result(result, vehicle_label="Manual Entry Vehicle")

# ----------------------------------------------------------------------
# MODE 3: SAMPLE VEHICLE
# ----------------------------------------------------------------------
else:
    st.markdown("### 🎲 Try a Sample Vehicle")
    st.markdown("Select a preset scenario to instantly see AutoGuard AI in action.")

    presets = {
        "✅ Healthy Vehicle": {
            "Engine_Temperature": 86, "Vibration": 1.9, "Battery_Voltage": 13.3,
            "Engine_Load": 45, "RPM": 2100, "Oil_Quality_Index": 88,
            "Mileage_km": 22000, "Vehicle_Age_Years": 1.5
        },
        "🟡 Moderate Wear": {
            "Engine_Temperature": 95, "Vibration": 3.2, "Battery_Voltage": 12.4,
            "Engine_Load": 65, "RPM": 2500, "Oil_Quality_Index": 60,
            "Mileage_km": 85000, "Vehicle_Age_Years": 4.5
        },
        "🟠 High Risk": {
            "Engine_Temperature": 102, "Vibration": 4.8, "Battery_Voltage": 11.7,
            "Engine_Load": 80, "RPM": 2900, "Oil_Quality_Index": 42,
            "Mileage_km": 130000, "Vehicle_Age_Years": 7
        },
        "🔴 Critical / Imminent Failure": {
            "Engine_Temperature": 112, "Vibration": 7.0, "Battery_Voltage": 11.0,
            "Engine_Load": 92, "RPM": 3400, "Oil_Quality_Index": 28,
            "Mileage_km": 195000, "Vehicle_Age_Years": 10
        },
    }

    selected_preset = st.selectbox("Select a scenario:", list(presets.keys()))

    with st.expander("📋 View preset sensor values"):
        st.json(presets[selected_preset])

    if st.button("🔍 Run Prediction", use_container_width=True):
        result = predict_single(presets[selected_preset], model, scaler, metadata)
        render_result(result, vehicle_label=selected_preset)

# ----------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------
st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#6b7785; font-size:0.85rem;">'
    'AutoGuard AI — Hackathon Proof of Concept | Built with Streamlit, Scikit-learn & XGBoost'
    '</div>',
    unsafe_allow_html=True,
)
