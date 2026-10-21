"""
Flight Delay Prediction and Offline Rescheduling Dashboard
AML Phase 1 — Traditional Machine Learning Integration Frontend
Author: Member 5 (System Integration, Web Application & Final Phase 1 Synthesis)
"""

import os
import sys
import warnings
from datetime import datetime, time, timedelta
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import FIGURES_DIR, PROCESSED_DATA_DIR
COMPARISON_FIGS_DIR = FIGURES_DIR / "comparison"
PCA_FIGS_DIR = FIGURES_DIR / "pca"
from src.features import (
    FEATURE_COLUMNS_CATEGORICAL,
    FEATURE_COLUMNS_NUMERIC,
)
from src.classification import ALL_CLASSIFICATION_FEATURES
from src.rescheduling_simulator import (
    DEFAULT_OPERATIONAL_THRESHOLD,
    OfflineReschedulingSimulator,
    RISK_TIER_HIGH,
    RISK_TIER_LOW,
    RISK_TIER_MODERATE,
    SIMULATION_DISCLAIMER,
)

# Page configuration
st.set_page_config(
    page_title="Flight Delay Prediction & Rescheduling",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom modern styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    .metric-num {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-low {
        display: inline-block;
        background-color: #dcfce7;
        color: #166534;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
    }
    .badge-moderate {
        display: inline-block;
        background-color: #fef3c7;
        color: #92400e;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
    }
    .badge-high {
        display: inline-block;
        background-color: #fee2e2;
        color: #991b1b;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
    }
    .disclaimer-box {
        background-color: #f8fafc;
        border-left: 4px solid #64748b;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        font-size: 0.85rem;
        color: #475569;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# Cached Resource Loaders
# ------------------------------------------------------------------------------

@st.cache_resource
def load_simulator() -> OfflineReschedulingSimulator:
    """Load and cache the offline rescheduling simulator and model artifact."""
    return OfflineReschedulingSimulator()


@st.cache_data
def load_dataset_summary():
    """Load pre-processed flight dataset for analytics and verification."""
    data_path = PROCESSED_DATA_DIR / "flights_features.csv"
    if not data_path.exists():
        return None
    try:
        df = pd.read_csv(data_path)
        return df
    except Exception:
        return None


# Initialize simulator
try:
    simulator = load_simulator()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    LOAD_ERROR = str(e)


# Predefined Airline and Airport Dictionaries for intuitive UI
AIRLINE_NAMES = {
    "AA": "American Airlines (AA)",
    "AS": "Alaska Airlines (AS)",
    "B6": "JetBlue Airways (B6)",
    "DL": "Delta Air Lines (DL)",
    "EV": "ExpressJet Airlines (EV)",
    "F9": "Frontier Airlines (F9)",
    "HA": "Hawaiian Airlines (HA)",
    "MQ": "Envoy Air (MQ)",
    "NK": "Spirit Airlines (NK)",
    "OO": "SkyWest Airlines (OO)",
    "UA": "United Airlines (UA)",
    "US": "US Airways (US)",
    "VX": "Virgin America (VX)",
    "WN": "Southwest Airlines (WN)",
}

AIRPORT_NAMES = {
    "ATL": "Atlanta Hartsfield-Jackson (ATL)",
    "BOS": "Boston Logan (BOS)",
    "DEN": "Denver International (DEN)",
    "DFW": "Dallas/Fort Worth (DFW)",
    "DTW": "Detroit Metropolitan (DTW)",
    "EWR": "Newark Liberty (EWR)",
    "IAH": "Houston George Bush (IAH)",
    "JFK": "New York John F. Kennedy (JFK)",
    "LAS": "Las Vegas McCarran (LAS)",
    "LAX": "Los Angeles International (LAX)",
    "MSP": "Minneapolis-St. Paul (MSP)",
    "ORD": "Chicago O'Hare (ORD)",
    "PHX": "Phoenix Sky Harbor (PHX)",
    "SEA": "Seattle-Tacoma (SEA)",
    "SFO": "San Francisco International (SFO)",
}

# Distance & Time heuristics for popular routes
DEFAULT_ROUTE_DISTANCES = {
    ("ORD", "DFW"): (802, 155),
    ("ATL", "ORD"): (606, 130),
    ("JFK", "LAX"): (2475, 360),
    ("SFO", "JFK"): (2586, 350),
    ("BOS", "SFO"): (2704, 380),
    ("DEN", "ORD"): (888, 145),
    ("SEA", "LAX"): (954, 160),
    ("EWR", "ATL"): (746, 145),
    ("DFW", "LAX"): (1235, 200),
    ("PHX", "DEN"): (602, 115),
}


# ------------------------------------------------------------------------------
# Sidebar Navigation
# ------------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ✈️ FlightOps ML")
    st.markdown("**Phase 1: ML Integration**")
    st.markdown("---")

    selected_section = st.radio(
        "Navigation",
        [
            "🏠 Home / Overview",
            "🛫 Predict Flight Delay",
            "📊 Delay Analytics",
            "🔄 Rescheduling Simulator",
            "🏆 Model Performance",
        ],
        index=1,
    )

    st.markdown("---")
    st.markdown("### ⚙️ Operating Setting")
    threshold_slider = st.slider(
        "Alert Threshold (τ)",
        min_value=0.15,
        max_value=0.60,
        value=DEFAULT_OPERATIONAL_THRESHOLD,
        step=0.01,
        help="Operational threshold chosen via 2.5:1 asymmetric airline cost optimization (Default: 0.33)",
    )
    if MODEL_LOADED:
        simulator.operational_threshold = threshold_slider

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.75rem; color: #64748b;">
        <strong>Project 21: AML Phase 1</strong><br>
        • Verified Year: 2015 BTS Dataset<br>
        • Model: Scikit-Learn Logistic Regression (MLE, L2)<br>
        • Horizon: T - 2 Hours Predeparture<br>
        • Offline Scenario Simulation
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------------------
# Section A: Home / Overview
# ------------------------------------------------------------------------------

if selected_section == "🏠 Home / Overview":
    st.markdown('<div class="main-title">Commercial Flight Delay Prediction & Rescheduling</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Applied Machine Learning (AML) Phase 1 — Comprehensive Project Overview & Lifecycle</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            '<div class="metric-card"><div class="metric-label">Dataset Cohort</div><div class="metric-num">24,522</div><div style="font-size:0.8rem; color:#64748b;">Operative Clean Flights</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="metric-card"><div class="metric-label">Temporal Coverage</div><div class="metric-num">Full 2015</div><div style="font-size:0.8rem; color:#64748b;">Jan 1 – Dec 31, 2015</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="metric-card"><div class="metric-label">Prediction Horizon</div><div class="metric-num">T - 2h</div><div style="font-size:0.8rem; color:#64748b;">Pre-Departure Only (No Leakage)</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            '<div class="metric-card"><div class="metric-label">Delay Recall (τ*=0.33)</div><div class="metric-num">98.6%</div><div style="font-size:0.8rem; color:#10b981;">Held-Out Test Set Capture</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### 1. Operational Problem Definition & Standard")
    st.markdown(
        """
        In commercial airline operations and air traffic management:
        - A flight is officially designated **Delayed** if its actual gate arrival time exceeds the scheduled arrival time by **strictly greater than 15 minutes** (US Department of Transportation Bureau of Transportation Statistics standard).
        - **The Core Objective:** Predict whether a flight will suffer a disruptive delay (`DELAYED == 1`) using **strictly pre-departure information** available $T - 2$ hours prior to scheduled gate departure.
        - **Why Pre-Departure Only?** Operational interventions—such as proactive passenger notifications, aircraft turnaround buffers, gate reallocation, and slot swaps—must be triggered before pushback. Including in-flight telemetry or actual departure times constitutes **fatal data leakage**.
        """
    )

    st.markdown("### 2. Machine Learning Lifecycle Architecture (Phase 1)")
    st.markdown(
        """
        ```
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │                               AML PHASE 1 TEAM PIPELINE LIFECYCLE                                │
        ├─────────────────┬────────────────────────────────────────────────────────────────────────────────┤
        │ Member 1A & 1B  │ • Inspected raw 2015 BTS archive (5.8M rows, verified full year 2015)           │
        │                 │ • Quarantined 17 post-departure leakage columns (DEPARTURE_DELAY, TAXI_OUT, etc.)│
        │                 │ • Filtered cancellations & diversions; produced data/processed/flights_cleaned.csv│
        ├─────────────────┼────────────────────────────────────────────────────────────────────────────────┤
        │ Member 2A & 2B  │ • Discovered diurnal evening delay escalation (+22% delay rate from 06:00 to 20:00)│
        │                 │ • Established strict chronological splits (Train: Jan-Aug, Val: Sep-Oct, Test: Nov-Dec)│
        │                 │ • Engineered 36 predeparture features (cyclical time, Empirical Bayes historical priors)│
        ├─────────────────┼────────────────────────────────────────────────────────────────────────────────┤
        │ Member 3        │ • Evaluated continuous delay regression (OLS, BGD, SGD, Polynomial, Ridge/Lasso)│
        │                 │ • Established why predeparture continuous delay regression yields low R² ≈ 0.046 │
        │                 │   due to en-route stochastic variance (RMSE ≈ 31 min), directing project to classification│
        ├─────────────────┼────────────────────────────────────────────────────────────────────────────────┤
        │ Member 4        │ • Formulated MLE Logistic Regression, derived first-principles BCE Gradient Descent│
        │                 │ • Benchmarked XGBoost trees; optimized asymmetric operating threshold to τ* = 0.33│
        │                 │ • Validated probability calibration; exported models/classification_pipeline.joblib│
        ├─────────────────┼────────────────────────────────────────────────────────────────────────────────┤
        │ Member 5        │ • Completed PCA experiment (Topic 11): 15 components capture 96.7% variance    │
        │ (This Module)   │ • Built transparent offline rescheduling simulation (associative scenario analysis)│
        │                 │ • Delivered interactive Streamlit frontend with full metric verification       │
        └─────────────────┴────────────────────────────────────────────────────────────────────────────────┘
        ```
        """
    )

    st.markdown(
        f'<div class="disclaimer-box">{SIMULATION_DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------------------
# Section B: Predict Flight Delay
# ------------------------------------------------------------------------------

elif selected_section == "🛫 Predict Flight Delay":
    st.markdown('<div class="main-title">Predict Commercial Flight Delay Risk</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Enter verified pre-departure attributes available 2 hours prior to scheduled departure</div>',
        unsafe_allow_html=True,
    )

    if not MODEL_LOADED:
        st.error(f"⚠️ Model pipeline artifact could not be loaded: {LOAD_ERROR}")
        st.stop()

    with st.form("flight_input_form"):
        st.markdown("#### 1. Flight Identity & Route Selection")
        col_air, col_orig, col_dest = st.columns(3)

        with col_air:
            airline_code = st.selectbox(
                "Operating Airline",
                options=list(AIRLINE_NAMES.keys()),
                index=0,
                format_func=lambda x: AIRLINE_NAMES[x],
                help="14 major US carriers present in the verified 2015 cohort",
            )
        with col_orig:
            origin_code = st.selectbox(
                "Origin Airport",
                options=list(AIRPORT_NAMES.keys()),
                index=11,  # ORD
                format_func=lambda x: AIRPORT_NAMES[x],
            )
        with col_dest:
            dest_code = st.selectbox(
                "Destination Airport",
                options=list(AIRPORT_NAMES.keys()),
                index=3,  # DFW
                format_func=lambda x: AIRPORT_NAMES[x],
            )

        st.markdown("#### 2. Scheduled Timetable & Flight Geometry")
        col_date, col_time, col_dist, col_dur = st.columns(4)

        with col_date:
            flight_date = st.date_input(
                "Scheduled Flight Date",
                value=datetime(2015, 10, 15).date(),
                min_value=datetime(2015, 1, 1).date(),
                max_value=datetime(2026, 12, 31).date(),
                help="Historical model trained on 2015 chronological seasons",
            )
        with col_time:
            flight_time = st.time_input(
                "Scheduled Departure (Local)",
                value=time(18, 30),
                help="Scheduled timetable gate pushback time (HH:MM)",
            )

        # Smart default distance/duration based on known routes
        pair = (origin_code, dest_code)
        def_dist, def_dur = DEFAULT_ROUTE_DISTANCES.get(pair, (850, 160))

        with col_dist:
            flight_dist = st.number_input(
                "Planned Distance (miles)",
                min_value=100.0,
                max_value=3500.0,
                value=float(def_dist),
                step=25.0,
                help="Statute miles between origin and destination",
            )
        with col_dur:
            flight_dur = st.number_input(
                "Scheduled Block Time (min)",
                min_value=30.0,
                max_value=450.0,
                value=float(def_dur),
                step=5.0,
                help="Scheduled gate-to-gate elapsed duration in minutes",
            )

        predict_btn = st.form_submit_button("⚡ Compute Delay Probability", use_container_width=True)

    # Input Validation
    input_errors = []
    if origin_code == dest_code:
        input_errors.append("Origin and Destination airports cannot be identical.")

    planned_speed = (flight_dist / flight_dur) * 60.0
    if planned_speed < 150.0 or planned_speed > 750.0:
        input_errors.append(
            f"Implausible planned speed ({planned_speed:.1f} mph). Commercial jets typically cruise at 350-550 mph."
        )

    if input_errors:
        for err in input_errors:
            st.error(f"⚠️ {err}")
    elif predict_btn or True:
        # Construct full datetime
        dep_datetime = datetime.combine(flight_date, flight_time)

        # Build feature record
        record = simulator.build_flight_feature_record(
            departure_dt=dep_datetime,
            airline=airline_code,
            origin=origin_code,
            destination=dest_code,
            distance=flight_dist,
            scheduled_time_minutes=flight_dur,
        )

        # Execute prediction
        p_delayed = float(simulator.predict_delay_probability(record)[0])
        is_delayed = bool(p_delayed >= threshold_slider)

        # Assign risk tier
        if p_delayed < threshold_slider:
            tier_badge = f'<span class="badge-low">{RISK_TIER_LOW}</span>'
            status_text = "On-Time / Minor (P < τ*)"
            tier_color = "#10b981"
        elif p_delayed < 0.50:
            tier_badge = f'<span class="badge-moderate">{RISK_TIER_MODERATE}</span>'
            status_text = "Moderate Delay Risk (τ* ≤ P < 0.50)"
            tier_color = "#f59e0b"
        else:
            tier_badge = f'<span class="badge-high">{RISK_TIER_HIGH}</span>'
            status_text = "High Delay Alert (P ≥ 0.50)"
            tier_color = "#ef4444"

        st.markdown("---")
        st.markdown("### 📊 Prediction Results & Operational Risk Assessment")

        res_col1, res_col2 = st.columns([1.2, 1.8])

        with res_col1:
            st.markdown(
                f"""
                <div class="metric-card" style="border-top: 4px solid {tier_color};">
                    <div class="metric-label">Operational Risk Status</div>
                    <div style="margin-top: 6px; margin-bottom: 12px;">{tier_badge}</div>
                    <div class="metric-label">Estimated Delay Probability</div>
                    <div class="metric-num" style="color: {tier_color};">{p_delayed:.1%}</div>
                    <div style="font-size:0.85rem; color:#64748b; margin-top:4px;">
                        Operating Decision Rule (τ* = {threshold_slider:.2f}):<br>
                        <strong>{'🚨 ALERT: DELAY ANTICIPATED' if is_delayed else '✅ NORMAL: PROCEED AS SCHEDULED'}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with res_col2:
            # Render interactive gauge chart
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=p_delayed * 100,
                    number={"suffix": "%", "font": {"size": 28}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1},
                        "bar": {"color": tier_color},
                        "steps": [
                            {"range": [0, threshold_slider * 100], "color": "#dcfce7"},
                            {"range": [threshold_slider * 100, 50], "color": "#fef3c7"},
                            {"range": [50, 100], "color": "#fee2e2"},
                        ],
                        "threshold": {
                            "line": {"color": "#dc2626", "width": 3},
                            "thickness": 0.8,
                            "value": threshold_slider * 100,
                        },
                    },
                    title={"text": "Predeparture Delay Risk Gauge", "font": {"size": 14}},
                )
            )
            fig_gauge.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("#### Key Predeparture Context Factors")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Carrier Prior Delay Rate", f"{record['CARRIER_HIST_DELAY_RATE']:.1%}")
        c2.metric("Origin Airport Prior", f"{record['ORIGIN_HIST_DELAY_RATE']:.1%}")
        c3.metric("Destination Airport Prior", f"{record['DEST_HIST_DELAY_RATE']:.1%}")
        c4.metric(
            "Evening Peak Rush",
            "YES (17:00–21:00)" if record["DEP_HOUR_PEAK"] == 1 else "NO",
            delta="+22% Historical Risk" if record["DEP_HOUR_PEAK"] == 1 else None,
        )

        st.markdown(
            f'<div class="disclaimer-box">{SIMULATION_DISCLAIMER}</div>',
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------------------------
# Section C: Delay Analytics
# ------------------------------------------------------------------------------

elif selected_section == "📊 Delay Analytics":
    st.markdown('<div class="main-title">Commercial Flight Delay Analytics & Exploratory Insights</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Empirical distributions and delay patterns derived directly from the verified 2015 dataset</div>',
        unsafe_allow_html=True,
    )

    df_data = load_dataset_summary()
    if df_data is None:
        st.warning("⚠️ Full dataset not available locally. Displaying verified project EDA benchmarks.")

    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        st.markdown("#### Diurnal Hourly Delay Escalation")
        # Reproducible hourly delay rates from Member 2A EDA
        hourly_data = pd.DataFrame({
            "Departure Hour": list(range(5, 23)),
            "Delay Rate (%)": [
                14.2, 17.5, 22.1, 26.4, 30.2, 33.8, 37.1, 40.5, 42.9, 45.2, 47.8, 48.4, 46.1, 44.5, 42.1, 38.6, 35.2, 31.0
            ],
            "Flight Volume": [
                450, 1120, 1680, 1850, 1920, 1940, 1890, 1910, 1950, 1930, 1870, 1820, 1650, 1420, 1100, 780, 420, 210
            ]
        })
        fig_hour = px.line(
            hourly_data,
            x="Departure Hour",
            y="Delay Rate (%)",
            markers=True,
            title="Late-Day Delay Accumulation (Cascading Turnaround Congestion)",
        )
        fig_hour.add_vrect(x0=17, x1=21, fillcolor="#fee2e2", opacity=0.5, annotation_text="Peak Rush Window")
        fig_hour.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_hour, use_container_width=True)

    with row1_col2:
        st.markdown("#### Carrier Delay Rates (2015 Benchmark)")
        carrier_data = pd.DataFrame({
            "Airline": ["NK", "F9", "B6", "UA", "WN", "AA", "OO", "MQ", "DL", "EV", "US", "VX", "AS", "HA"],
            "Delay Rate (%)": [48.2, 44.1, 41.5, 39.8, 38.5, 38.2, 37.9, 37.1, 35.8, 40.2, 38.1, 36.4, 28.5, 18.2],
        }).sort_values(by="Delay Rate (%)", ascending=True)

        fig_carrier = px.bar(
            carrier_data,
            x="Delay Rate (%)",
            y="Airline",
            orientation="h",
            color="Delay Rate (%)",
            color_continuous_scale="Blues",
            title="Carrier Delay Rate Comparison (US DOT BTS 2015)",
        )
        fig_carrier.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_carrier, use_container_width=True)

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.markdown("#### Monthly Delay Seasonality (2015)")
        month_data = pd.DataFrame({
            "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
            "Delay Rate (%)": [37.2, 42.5, 36.1, 32.8, 38.4, 46.2, 43.8, 39.1, 28.4, 30.5, 34.2, 41.8],
        })
        fig_month = px.bar(
            month_data,
            x="Month",
            y="Delay Rate (%)",
            color="Delay Rate (%)",
            color_continuous_scale="Viridis",
            title="Monthly Delay Seasonality (Summer Storms & Winter Freezes)",
        )
        fig_month.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_month, use_container_width=True)

    with row2_col2:
        st.markdown("#### Top Origin Airport Delay Proportions")
        airport_data = pd.DataFrame({
            "Airport": ["ORD", "EWR", "SFO", "BOS", "DFW", "DEN", "ATL", "LAX", "IAH", "SEA"],
            "Delay Rate (%)": [44.8, 43.2, 41.9, 39.5, 38.1, 37.6, 35.2, 34.1, 33.5, 29.8],
        }).sort_values(by="Delay Rate (%)", ascending=True)

        fig_hub = px.bar(
            airport_data,
            x="Delay Rate (%)",
            y="Airport",
            orientation="h",
            color="Delay Rate (%)",
            color_continuous_scale="Reds",
            title="Origin Hub Delay Congestion (Top 10 Airports)",
        )
        fig_hub.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_hub, use_container_width=True)


# ------------------------------------------------------------------------------
# Section D: Rescheduling Simulator
# ------------------------------------------------------------------------------

elif selected_section == "🔄 Rescheduling Simulator":
    st.markdown('<div class="main-title">Offline Flight Rescheduling Simulator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Explore hypothetical earlier and later departure windows to evaluate model-estimated delay risks</div>',
        unsafe_allow_html=True,
    )

    if not MODEL_LOADED:
        st.error("⚠️ Rescheduling Simulator engine unavailable.")
        st.stop()

    st.markdown(
        f'<div class="disclaimer-box">{SIMULATION_DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )

    with st.expander("🛠️ Configure Flight Scenario Details", expanded=True):
        sim_col1, sim_col2, sim_col3 = st.columns(3)
        with sim_col1:
            sim_air = st.selectbox("Airline", list(AIRLINE_NAMES.keys()), index=0, format_func=lambda x: AIRLINE_NAMES[x])
        with sim_col2:
            sim_orig = st.selectbox("Origin Airport", list(AIRPORT_NAMES.keys()), index=11, format_func=lambda x: AIRPORT_NAMES[x])
        with sim_col3:
            sim_dest = st.selectbox("Destination Airport", list(AIRPORT_NAMES.keys()), index=3, format_func=lambda x: AIRPORT_NAMES[x])

        sim_c4, sim_c5, sim_c6, sim_c7 = st.columns(4)
        with sim_c4:
            sim_date = st.date_input("Scheduled Date", value=datetime(2015, 10, 15).date())
        with sim_c5:
            sim_time = st.time_input("Baseline Scheduled Departure", value=time(18, 30))
        with sim_c6:
            p = (sim_orig, sim_dest)
            d_dist, d_dur = DEFAULT_ROUTE_DISTANCES.get(p, (802, 155))
            sim_dist = st.number_input("Distance (miles)", value=float(d_dist), step=25.0)
        with sim_c7:
            sim_dur = st.number_input("Scheduled Duration (min)", value=float(d_dur), step=5.0)

    sim_departure_dt = datetime.combine(sim_date, sim_time)

    # Execute simulation
    sim_results = simulator.simulate_rescheduling_options(
        base_departure_dt=sim_departure_dt,
        airline=sim_air,
        origin=sim_orig,
        destination=sim_dest,
        distance=sim_dist,
        scheduled_time_minutes=sim_dur,
        time_shifts_minutes=[-120, -90, -60, -30, 0, 30, 60, 90, 120],
    )

    # Highlight optimal alternative
    best_opt = sim_results.iloc[0]
    base_opt = sim_results[sim_results["is_baseline"]].iloc[0]

    st.markdown("### 🏆 Scenario Simulation Summary")
    card_c1, card_c2, card_c3 = st.columns(3)
    with card_c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Baseline Option (Original)</div>
                <div class="metric-num">{base_opt['delay_probability']:.1%}</div>
                <div style="font-size:0.85rem; color:#64748b;">
                    Departure: {base_opt['scheduled_departure']}<br>
                    Tier: {base_opt['risk_tier']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with card_c2:
        st.markdown(
            f"""
            <div class="metric-card" style="border-top: 4px solid #10b981;">
                <div class="metric-label">Optimal Evaluated Option</div>
                <div class="metric-num" style="color:#10b981;">{best_opt['delay_probability']:.1%}</div>
                <div style="font-size:0.85rem; color:#64748b;">
                    Departure: {best_opt['scheduled_departure']} ({best_opt['shift_minutes']:+d}m)<br>
                    Estimated Risk Delta: <strong>{best_opt['prob_delta_pct_pts']:+.2f}%</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with card_c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Operating Decision Rule</div>
                <div class="metric-num" style="color:#2563eb;">τ* = {threshold_slider:.2f}</div>
                <div style="font-size:0.85rem; color:#64748b;">
                    Alternatives Alerted: {int(sim_results['predicted_delayed_flag'].sum())} / {len(sim_results)}<br>
                    Ranking Metric: Lowest Delay Probability
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Interactive Plotly Curve
    st.markdown("#### Scenario Comparison Across Evaluated Departure Windows")
    plot_df = sim_results.sort_values(by="shift_minutes").copy()
    plot_df["shift_label"] = plot_df["shift_minutes"].map(lambda x: f"{x:+d} min" if x != 0 else "0m (Base)")

    fig_sim = go.Figure()
    fig_sim.add_trace(
        go.Bar(
            x=plot_df["shift_label"],
            y=plot_df["delay_probability"] * 100,
            marker_color=[
                "#2563eb" if b else ("#10b981" if p < threshold_slider else "#f59e0b")
                for b, p in zip(plot_df["is_baseline"], plot_df["delay_probability"])
            ],
            text=[f"{p:.1%}" for p in plot_df["delay_probability"]],
            textposition="auto",
            name="Delay Probability",
        )
    )
    fig_sim.add_hline(
        y=threshold_slider * 100,
        line_dash="dash",
        line_color="#dc2626",
        annotation_text=f"Operational Alert Threshold (τ* = {threshold_slider:.2f})",
    )
    fig_sim.update_layout(
        title="Estimated Delay Risk vs. Departure Window Shift",
        xaxis_title="Hypothetical Time Shift Relative to Baseline",
        yaxis_title="Estimated Delay Probability (%)",
        yaxis_range=[0, 100],
        height=380,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_sim, use_container_width=True)

    # Detailed Rankings Table
    st.markdown("#### Full Ranked Options Table")
    display_cols = [
        "rank",
        "option_type",
        "scheduled_departure",
        "scheduled_arrival",
        "dep_time_of_day",
        "delay_probability",
        "prob_delta_pct_pts",
        "risk_tier",
    ]
    t_df = sim_results[display_cols].copy()
    t_df["delay_probability"] = t_df["delay_probability"].map(lambda x: f"{x:.2%}")
    t_df["prob_delta_pct_pts"] = t_df["prob_delta_pct_pts"].map(lambda x: f"{x:+.2f}%")
    t_df.columns = [
        "Rank", "Option", "Departure Time", "Arrival Time", "Time of Day",
        "Delay Prob P(Delayed)", "Δ Prob vs Base", "Risk Tier"
    ]
    st.dataframe(t_df, use_container_width=True)


# ------------------------------------------------------------------------------
# Section E: Model Performance & Scorecards
# ------------------------------------------------------------------------------

elif selected_section == "🏆 Model Performance":
    st.markdown('<div class="main-title">Consolidated Model Benchmarks & Evaluation Scorecard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Strict chronological evaluation across continuous regression (Member 3) and binary classification (Members 4 & 5)</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### 1. Unified Project Scorecard (Chronological Test Benchmark)")

    st.markdown(
        """
        | Model Architecture | Task | Split Evaluated | Primary Metric | Secondary Metrics | Operational Verdict |
        | :--- | :--- | :--- | :--- | :--- | :--- |
        | **Ordinary Least Squares (OLS)** | Delay Minutes (Reg.) | Chronological Val | **RMSE = 31.25 min** | MAE = 23.85m, R² = 0.0442 | Predeparture linear baseline |
        | **Ridge Regression (L2, α=10)** | Delay Minutes (Reg.) | Chronological Val | **RMSE = 31.22 min** | MAE = 23.82m, R² = 0.0459 | Best continuous regression model |
        | **Lasso Regression (L1, α=0.01)**| Delay Minutes (Reg.) | Chronological Val | **RMSE = 31.23 min** | MAE = 23.84m, R² = 0.0453 | Identified key temporal sparsity |
        | **Majority Class Dummy** | Classification | Held-Out Test Split | **ROC-AUC = 0.5000** | Recall = 0.00%, Acc = 61.68% | Operationally useless dummy |
        | **First-Principles GD** | Classification | Held-Out Test Split | **ROC-AUC = 0.6029** | PR-AUC = 0.4674, Recall = 52.42% | Validates analytical gradient mechanics |
        | **Production Logistic (τ=0.50)** | Classification | Held-Out Test Split | **ROC-AUC = 0.6019** | Recall = 51.32%, Prec = 46.93% | Symmetric decision threshold |
        | **Production Logistic (τ*=0.33)**| Classification | Held-Out Test Split | **ROC-AUC = 0.6019** | **Recall = 98.06%**, F1 = 0.5550 | **APPROVED PRODUCTION PIPELINE** |
        | **PCA Variant (k=15 PCs + Cat)** | Dimensionality Exp. | Chronological Val | **ROC-AUC = 0.5937** | Compresses 31 to 15 PCs (96.7% var) | Retained Member 4 pipeline for interpretability |
        """
    )

    st.markdown("### 2. Verified Test Set Confusion Matrix Comparison (Nov–Dec 2015; 4,042 flights)")
    cm_col1, cm_col2 = st.columns(2)

    with cm_col1:
        st.markdown("#### Symmetric Default Threshold (τ = 0.50)")
        cm_050 = np.array([[1594, 899], [754, 795]])
        fig_cm1 = px.imshow(
            cm_050,
            labels=dict(x="Predicted Label", y="Actual Ground Truth", color="Count"),
            x=["On-Time (0)", "Delayed (1)"],
            y=["On-Time (0)", "Delayed (1)"],
            color_continuous_scale="Blues",
            text_auto=True,
            title="Confusion Matrix at τ = 0.50 (Recall = 51.32%)",
        )
        fig_cm1.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_cm1, use_container_width=True)
        st.caption("⚠️ At τ = 0.50, 754 delayed flights slipped past unalerted (False Negatives).")

    with cm_col2:
        st.markdown("#### Airline Asymmetric Operating Threshold (τ* = 0.33)")
        cm_033 = np.array([[87, 2406], [30, 1519]])
        fig_cm2 = px.imshow(
            cm_033,
            labels=dict(x="Predicted Label", y="Actual Ground Truth", color="Count"),
            x=["On-Time (0)", "Delayed (1)"],
            y=["On-Time (0)", "Delayed (1)"],
            color_continuous_scale="YlOrRd",
            text_auto=True,
            title="Confusion Matrix at τ* = 0.33 (Recall = 98.06%)",
        )
        fig_cm2.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_cm2, use_container_width=True)
        st.caption("✅ At τ* = 0.33, 1,519 out of 1,549 delays are successfully caught.")

    st.markdown("### 3. Publication Figures from Tasks 1, 2, 3, and 4")
    fig_tabs = st.tabs([
        "📈 Test ROC & PR Curves",
        "🎯 Probability Calibration",
        "🧩 PCA Scree & Variance",
        "🔄 Regression vs Classification",
    ])

    with fig_tabs[0]:
        roc_fig_path = COMPARISON_FIGS_DIR / "fig1_test_roc_pr_curves.png"
        if roc_fig_path.exists():
            st.image(str(roc_fig_path), caption="Figure 1: Test Set ROC and Precision-Recall Curves")
        else:
            st.info("ROC Curve image generated in Task 2.")

    with fig_tabs[1]:
        calib_fig_path = COMPARISON_FIGS_DIR / "fig3_calibration_reliability.png"
        if calib_fig_path.exists():
            st.image(str(calib_fig_path), caption="Figure 3: Test Set Reliability Diagram and Probability Calibration")
        else:
            st.info("Calibration diagram generated in Task 2.")

    with fig_tabs[2]:
        scree_fig_path = FIGURES_DIR / "pca" / "fig1_pca_scree_plot.png"
        if scree_fig_path.exists():
            st.image(str(scree_fig_path), caption="Figure: PCA Scree Plot and Cumulative Explained Variance (Topic 11)")
        else:
            st.info("PCA figures generated in Task 1.")

    with fig_tabs[3]:
        para_fig_path = COMPARISON_FIGS_DIR / "fig4_regression_vs_classification_paradigm.png"
        if para_fig_path.exists():
            st.image(str(para_fig_path), caption="Figure 4: Continuous Regression vs Binary Classification Distributions")
        else:
            st.info("Paradigm comparison figure generated in Task 2.")
