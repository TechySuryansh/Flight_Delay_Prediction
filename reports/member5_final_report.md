# Member 5: System Integration, PCA Experiment, Final Evaluation, Rescheduling Simulation, and Streamlit Frontend

**Project 21:** Commercial Flight Delay Prediction and Rescheduling  
**Team Role:** Member 5 (System Integration, Web Application & Final Phase 1 Synthesis)  
**Phase:** AML Phase 1 — Traditional Machine Learning (Academic Syllabus Integration)  
**Date:** October 2026  
**Status:** Task 5 Complete — Production Streamlit Dashboard & Comprehensive Phase 1 Report Validated  

---

## Executive Summary

As **Member 5**, this report delivers the final integration and Phase 1 synthesis of the **Commercial Flight Delay Prediction and Rescheduling** project. Building directly upon the foundation established by Members 1A, 1B, 2A, 2B, 3, and 4, this work consolidates all prior experimental findings, executes the syllabus-mandated Principal Component Analysis (PCA) experiment (Topic 11), benchmarks regression versus classification under frozen chronological partitions, constructs an offline flight rescheduling simulator, and deploys an interactive, modern Streamlit web frontend for live academic demonstrations.

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               PHASE 1 FINAL BENCHMARK SCORECARD                                  │
├──────────────────────────────┬──────────────┬────────────┬─────────────┬─────────────┬───────────┤
│ Model Architecture           │ Problem Type │ Partition  │ Key Primary │ Key Secondary│ Status   │
├──────────────────────────────┼──────────────┼────────────┼─────────────┼─────────────┼───────────┤
│ Ordinary Least Squares (OLS) │ Delay Minutes│ Chron. Val │ RMSE=31.25m │ R² = 0.0442 │ Evaluated │
│ Ridge Regression (L2, α=10)  │ Delay Minutes│ Chron. Val │ RMSE=31.22m │ R² = 0.0459 │ Evaluated │
│ Lasso Regression (L1, α=0.01)│ Delay Minutes│ Chron. Val │ RMSE=31.23m │ R² = 0.0453 │ Evaluated │
│ Majority Class Dummy Baseline│ Delay Flag   │ Frozen Test│ ROC = 0.5000│ Recall = 0% │ Baseline  │
│ Educational GD (Logistic)    │ Delay Flag   │ Frozen Test│ ROC = 0.6029│ PR = 0.4674 │ Verified  │
│ Production Logistic (τ=0.50) │ Delay Flag   │ Frozen Test│ ROC = 0.6019│ Rec = 51.32%│ Default   │
│ Production Logistic (τ*=0.33)│ Delay Flag   │ Frozen Test│ ROC = 0.6019│ Rec = 98.06%│ SELECTED  │
│ PCA Model (15 PCs + Cat)     │ Dimensionality Chron. Val │ ROC = 0.5937│ 96.7% Var   │ Documented│
└──────────────────────────────┴──────────────┴────────────┴─────────────┴─────────────┴───────────┘
```

---

## 1. Problem Definition & Predeparture Operational Horizon

### 1.1 The Operational Standard
Per the United States Department of Transportation (US DOT) Bureau of Transportation Statistics (BTS) standard:
$$\text{A commercial flight is designated \textbf{DELAYED} if its actual arrival delay exceeds 15 minutes:}$$
$$\text{DELAYED} = \begin{cases} 1 & \text{if } \text{ARRIVAL\_DELAY} > 15.0 \text{ minutes} \\ 0 & \text{if } \text{ARRIVAL\_DELAY} \le 15.0 \text{ minutes} \end{cases}$$

### 1.2 The $T - 2$ Hours Predeparture Horizon & Anti-Leakage Guardrails
Operational interventions (such as aircraft turnaround staging, gate slot reallocation, proactive passenger connection rebooking, and buffer scheduling) must be initiated **prior to gate pushback**. 

Predictions are strictly generated at **$T - 2$ hours (120 minutes) prior to scheduled departure time**.
- **Permitted Pre-Departure Inputs:** Scheduled departure and arrival times, scheduled block duration, planned flight distance, operating airline, origin airport, destination airport, calendar dates, and strictly historical priors computed prior to dispatch.
- **Strictly Excluded Post-Departure Variables (Data Leakage):**
  - Actual departure time (`DEPARTURE_TIME`) and departure delay (`DEPARTURE_DELAY`)
  - Surface movement times (`TAXI_OUT`, `TAXI_IN`, `WHEELS_OFF`, `WHEELS_ON`)
  - In-flight telemetry (`AIR_TIME`, `ELAPSED_TIME`, `ARRIVAL_TIME`)
  - Post-hoc delay cause breakdowns (`AIR_SYSTEM_DELAY`, `SECURITY_DELAY`, `AIRLINE_DELAY`, `LATE_AIRCRAFT_DELAY`, `WEATHER_DELAY`)
  - Operational non-arrivals (`CANCELLED`, `DIVERTED`): Arrival delay is undefined for non-arrivals; these are filtered during preprocessing to ensure clean ground-truth training signals.

---

## 2. Dataset Provenance, Partitions, and Cleaning Summary

### 2.1 Verified Dataset Scope
- **Source:** US DOT BTS National On-Time Flight Performance Database (2015 Calendar Year).
- **Temporal Range:** Exclusively **January 1, 2015 to December 31, 2015** (365 unique calendar days).
- **Operative Cohort Size:** Exactly **24,522 clean, operative flights** (derived from 25,000 raw records after excluding 411 cancellations and 67 diversions; 98.09% retention).
- **Target Prevalence:** 9,476 delayed flights (**38.64% delayed**) and 15,046 on-time flights (**61.36% on-time**).

### 2.2 Strict Chronological Non-Overlapping Partitions
To guarantee zero lookahead bias, all data splitting strictly follows calendar dates:
1. **Training Partition (`SPLIT_SET == 'train'`):**
   - Dates: **January 1, 2015 – August 31, 2015** (243 days).
   - Rows: **16,366 flights** (38.63% positive delay rate).
   - Purpose: Feature scalers, median imputers, one-hot encoders, PCA eigenvectors, empirical Bayes priors, and regression/classification model parameters are fitted **exclusively on this partition**.
2. **Validation Partition (`SPLIT_SET == 'val'`):**
   - Dates: **September 1, 2015 – October 31, 2015** (61 days).
   - Rows: **4,114 flights** (38.99% positive delay rate).
   - Purpose: Hyperparameter tuning, PCA component dimension selection ($k$), and asymmetric operating threshold optimization ($\tau^* = 0.33$).
3. **Held-Out Test Partition (`SPLIT_SET == 'test'`):**
   - Dates: **November 1, 2015 – December 31, 2015** (61 days).
   - Rows: **4,042 flights** (38.32% positive delay rate).
   - Purpose: Frozen final benchmark evaluation conducted strictly once after all models and thresholds were locked.

---

## 3. Pre-Departure Feature Engineering Architecture

From Member 2B's handoff (`data/processed/flights_features.csv`), the modeling pipeline ingests exactly **36 pre-departure features**:

### 3.1 31 Numerical Features (Continuous & Cyclical)
1. **Departure Temporal Coordinates:** `DEP_HOUR` (0–23), `DEP_MINUTE` (0–59), `DEP_TIME_MINUTES` (0–1439).
2. **Cyclical Departure Encodings:**
   $$\text{DEP\_HOUR\_SIN} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \text{DEP\_HOUR\_COS} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$
   $$\text{DEP\_TIME\_SIN} = \sin\left(\frac{2\pi \cdot \text{minutes}}{1440}\right), \quad \text{DEP\_TIME\_COS} = \cos\left(\frac{2\pi \cdot \text{minutes}}{1440}\right)$$
3. **Diurnal Risk Indicators:** `DEP_HOUR_PEAK` (binary flag for evening rush: 17:00–20:59).
4. **Arrival Coordinates & Curfew:** `ARR_HOUR`, `ARR_MINUTE`, `ARR_TIME_MINUTES`, `ARR_HOUR_SIN`, `ARR_HOUR_COS`, and `ARR_NIGHT_CURFEW` (binary flag for 23:00–05:59 arrival).
5. **Calendar Seasonality Coordinates:** `DAY_OF_WEEK` (1=Mon, 7=Sun), `DOW_SIN`, `DOW_COS`, `IS_WEEKEND`, `MONTH` (1–12), `MONTH_SIN`, `MONTH_COS`, `DAY` (1–31), `QUARTER` (1–4), `IS_MONTH_END`.
6. **Flight Geometry & Buffer Proxy:** `DISTANCE` (statute miles), `DISTANCE_LOG` ($\ln(1 + \text{dist})$), `SCHEDULED_TIME` (minutes), and `PLANNED_SPEED` ($(\text{dist} / \text{time}) \times 60$).
7. **Empirical Bayes Smoothed Historical Priors (Train-Only):**
   $$\text{Prior}(c) = \frac{\sum y_{\text{train}} + m \cdot \bar{y}_{\text{train}}}{N_c + m}$$
   Computed for `CARRIER_HIST_DELAY_RATE`, `ORIGIN_HIST_DELAY_RATE`, and `DEST_HIST_DELAY_RATE` with pseudo-count smoothing $m = 10.0$ and out-of-fold cross-validation on train.

### 3.2 5 Categorical Features (One-Hot Encoded)
1. `AIRLINE` (14 carriers: AA, AS, B6, DL, EV, F9, HA, MQ, NK, OO, UA, US, VX, WN)
2. `ORIGIN_AIRPORT` (15 hubs: ATL, BOS, DEN, DFW, DTW, EWR, IAH, JFK, LAS, LAX, MSP, ORD, PHX, SEA, SFO)
3. `DESTINATION_AIRPORT` (15 hubs)
4. `DEP_TIME_OF_DAY` (5 operational periods: Early Morning, Morning, Afternoon, Evening, Night)
5. `DISTANCE_CATEGORY` (3 FAA distance bands: Short $<500$mi, Medium $500-1500$mi, Long $>1500$mi)

---

## 4. Task 1: Principal Component Analysis (PCA) Experiment

### 4.1 Academic Syllabus Alignment (Topic 11)
Per the AML syllabus, **Topic 11 addresses Unsupervised Dimensionality Reduction via Principal Component Analysis (PCA)**. 

### 4.2 Mathematical Formulation
Given the standardized training feature matrix $Z \in \mathbb{R}^{m \times d}$ across the $d = 31$ numerical features:
1. **Sample Covariance Matrix:** $\Sigma = \frac{1}{m - 1} Z^T Z \in \mathbb{R}^{31 \times 31}$.
2. **Spectral Decomposition:** $\Sigma v_j = \lambda_j v_j$, where eigenvectors $V = [v_1, \dots, v_d]$ define orthogonal principal directions and eigenvalues $\lambda_j$ denote variance along axis $v_j$.
3. **Explained Variance Ratio:** $\text{EVR}_j = \lambda_j / \sum_{i=1}^d \lambda_i$.
4. **Orthogonal Projection:** $Z_{\text{pca}} = Z V_k \in \mathbb{R}^{m \times k}$.

### 4.3 Variance Concentration Findings
In `notebooks/06_pca_experiment.ipynb`, PCA was fitted **strictly on the 16,366 training flights**:
- **Top 5 Components:** Capture **64.23%** of total variance.
- **Top 10 Components:** Capture **85.93%** of total variance (reaches 80% threshold at $k=8$).
- **Top 12 Components:** Capture **91.83%** of total variance (reaches 90% threshold at $k=12$).
- **Top 15 Components:** Capture **96.71%** of total variance (reaches 95% threshold at $k=15$).
- **Top 20 Components:** Capture **99.26%** of total variance.

```
Individual Variance (Top 5 PCs):
  PC01: 22.84% (Flight geometry: distance, scheduled time, log-distance)
  PC02: 17.51% (Diurnal departure & arrival schedule progression)
  PC03: 11.23% (Calendar month and quarterly seasonality)
  PC04:  7.45% (Day of week and weekend indicators)
  PC05:  5.20% (Empirical Bayes historical delay priors)
```

### 4.4 Controlled Comparative Experiment: Baseline vs. PCA Models
We trained Logistic Regression classifiers across different component cutoffs $k$, combining the $k$ PCA components with the identical 52 one-hot categorical indicators, evaluated on the **chronological validation split** (Sep–Oct 2015; 4,114 flights):

| Model Configuration | Num Features | Cat Features | Total Dimensions | Variance Retained | Val ROC-AUC | Val PR-AUC | Val F1 ($\tau^*=0.33$) | Val Brier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Original Numerics + Cat)** | **31** | **52** | **83** | **100.0%** | **0.5919** | **0.4603** | **0.5645** | **0.2338** |
| PCA Variant ($k = 5$ PCs + Cat) | 5 | 52 | 57 | 64.23% | 0.5945 | 0.4618 | 0.5658 | 0.2336 |
| PCA Variant ($k = 10$ PCs + Cat) | 10 | 52 | 62 | 85.93% | 0.5946 | 0.4621 | 0.5661 | 0.2335 |
| PCA Variant ($k = 12$ PCs + Cat) | 12 | 52 | 64 | 91.83% | 0.5939 | 0.4615 | 0.5652 | 0.2337 |
| **PCA Variant ($k = 15$ PCs + Cat)** | **15** | **52** | **67** | **96.71%** | **0.5937** | **0.4612** | **0.5650** | **0.2337** |
| PCA Variant ($k = 20$ PCs + Cat) | 20 | 52 | 72 | 99.26% | 0.5916 | 0.4598 | 0.5640 | 0.2339 |
| PCA Variant ($k = 25$ PCs + Cat) | 25 | 52 | 77 | 99.98% | 0.5918 | 0.4601 | 0.5644 | 0.2338 |

### 4.5 Scientific Conclusions on PCA
1. **Collinearity Reduction:** Compressing 31 numerical features to 10–15 orthogonal principal components yields a marginal gain of $+0.002$ in validation ROC-AUC ($0.5919 \to 0.5937$) by eliminating multicollinearity among cyclical time variables (`DEP_HOUR_SIN`, `DEP_TIME_COS`, etc.).
2. **Loss of Interpretability:** In Member 4's original feature space, each logistic regression coefficient directly yields an interpretable **Odds Ratio** $\text{OR}_j = \exp(\beta_j)$ (e.g., evening rush hour increases delay odds by $1.35\times$). Under PCA, each principal component is an uninterpretable linear combination of all 31 features ($0.38 \cdot \text{dist} - 0.24 \cdot \text{hour} + 0.15 \cdot \text{month}$).
3. **Production Decision:** Because the discrimination difference is negligible ($\sim 0.002$) while operational interpretability is critical for airline stakeholders and frontend dashboards, **PCA is retained as a documented syllabus experiment**, and **Member 4's validated original pipeline (`models/classification_pipeline.joblib`) is preserved for production**.

---

## 5. Task 2: Final Model Comparison and Frozen Test Evaluation

### 5.1 Continuous Regression vs. Binary Classification
- **Continuous Delay Regression (Member 3):**
  - Ridge ($L_2$): MAE = 23.82 min, RMSE = 31.22 min, $R^2 = 0.0459$.
  - Lasso ($L_1$): MAE = 23.84 min, RMSE = 31.23 min, $R^2 = 0.0453$.
  - Ordinary Least Squares: MAE = 23.85 min, RMSE = 31.25 min, $R^2 = 0.0442$.
  - *Finding:* En-route convective weather and surface taxiway congestion occurring hours after gate departure represent high irreducible variance ($\sigma^2$), bounding linear regression to $R^2 \approx 0.046$.
- **Binary Classification (Member 4 & 5):**
  - Absorbs minute-level jitter into an actionable operational binary label (`DELAYED == 1`).
  - Achieves stable probabilistic discrimination (ROC-AUC $\approx 0.602$, PR-AUC $\approx 0.466$).

### 5.2 Frozen Held-Out Test Set Benchmark (Nov 1 – Dec 31, 2015; 4,042 flights)

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               FROZEN HELD-OUT TEST EVALUATION MATRIX                             │
├──────────────────────────────┬───────────┬────────┬──────────┬────────┬─────────┬───────┬────────┤
│ Model Architecture           │ Threshold │ Recall │ Prec.    │ F1     │ ROC-AUC │ PR-AUC│ Brier  │
├──────────────────────────────┼───────────┼────────┼──────────┼────────┼─────────┼───────┼────────┤
│ Majority Class Dummy Baseline│ 0.50      │ 0.00%  │ 0.00%    │ 0.0000 │ 0.5000  │ 0.3832│ 0.3832 │
│ Educational First-Prin. GD   │ 0.50      │ 52.42% │ 47.07%   │ 0.4960 │ 0.6029  │ 0.4674│ 0.2379 │
│ Production Logistic (MLE, L2)│ 0.50      │ 51.32% │ 46.93%   │ 0.4903 │ 0.6019  │ 0.4663│ 0.2379 │
│ Production Logistic (SELECTED│ 0.33*     │ 98.06% │ 38.70%   │ 0.5550 │ 0.6019  │ 0.4663│ 0.2379 │
└──────────────────────────────┴───────────┴────────┴──────────┴────────┴─────────┴───────┴────────┘
```

### 5.3 Asymmetric Cost Threshold Rationale ($\tau^* = 0.33$)
In commercial airline dispatch:
$$\text{Cost} = 2.5 \times \text{False Negatives} + 1.0 \times \text{False Positives}$$
- **At $\tau = 0.50$:** Missed **754 delayed flights** (Recall = 51.32%). Total cost = $2.5(754) + 1.0(899) = 2,784$.
- **At $\tau^* = 0.33$:** Missed only **30 delayed flights** (Recall = 98.06%, or 98.64% on Member 4's calibration cohort). Total cost = $2.5(30) + 1.0(2,406) = 2,481$ (cost reduction of **-303 points**).

### 5.4 Probability Calibration
- Evaluated via Reliability Diagrams across 10 quantile bins on the test partition (`results/figures/comparison/fig3_calibration_reliability.png`).
- Brier score loss = **0.2379** (vs. uninformative dummy baseline of 0.3832), confirming that predicted probabilities closely match empirical class rates.

---

## 6. Task 3: Offline Rescheduling Simulator

### 6.1 Architecture & Design (`src/rescheduling_simulator.py`)
The simulator evaluates hypothetical schedule adjustments (e.g. shifts $\Delta t \in \{-120, -90, -60, -30, 0, +30, +60, +90, +120\}$ minutes) around a baseline scheduled flight:
1. **Dynamic Temporal Recomputation:** For each hypothetical departure time, the simulator updates:
   - Departure hour, minute, and minutes from midnight.
   - Cyclical coordinates: $\sin/\cos$ for departure hour and departure minutes.
   - Evening peak rush indicator (`17 <= hour <= 20`).
   - Operational time-of-day category (`Early Morning`, `Morning`, `Afternoon`, `Evening`, `Night`).
   - Arrival timing: adds scheduled block duration to compute arrival hour, arrival minute, cyclical arrival coordinates, and night curfew flags (`23 <= arr_hour or arr_hour < 6`).
2. **Context Preservation:** Preserves airline, origin, destination, planned distance, scheduled time, planned speed, calendar date coordinates, and Empirical Bayes historical delay priors.
3. **Pipeline Ingestion:** Passes the updated 36 features through `pipeline.transform()` and `model.predict_proba()` from `models/classification_pipeline.joblib`.
4. **Ranking & Delta Computation:** Ranks options by estimated delay probability $P(\text{Delayed})$ and computes risk deltas relative to baseline ($\Delta P = P_{\text{alt}} - P_{\text{baseline}}$).

### 6.2 Operational & Scientific Safeguards
```
OPERATIONAL & SCIENTIFIC DISCLAIMER:
1. Model-Based Scenario Estimate: All delay probabilities and rankings are offline statistical
   estimates derived from historical 2015 flight patterns. They do NOT guarantee actual on-time
   arrival or delay reduction.
2. Non-Causal Association: The predictive model captures historical correlations (e.g., lower
   congestion in early morning hours) and does NOT establish that rescheduling a flight causes lower
   operational delay risk.
3. No Operational Availability Guarantee: This simulation does NOT check live airline seat
   inventory, air traffic control slot clearance, aircraft tail routing, or airline scheduling approval.
```

---

## 7. Tasks 4 & 5: Streamlit Web Application Frontend

### 7.1 Architecture & Structure (`app/app.py`)
Built with Python, Streamlit, and Plotly, the web frontend provides an interactive, responsive dashboard for live academic demonstrations:

1. **🏠 Home / Overview:**
   - Operational delay definition ($> 15$ min arrival delay per US DOT BTS).
   - Verified 2015 dataset cohort metrics (24,522 rows, full calendar year).
   - 8-stage ML lifecycle architecture and anti-leakage policies.
2. **🛫 Predict Flight Delay:**
   - Input controls for 14 airlines, 15 origin airports, 15 destination airports, flight dates, departure times, planned distance, and duration.
   - Live speed validation (flags implausible commercial aircraft speeds $< 150$ or $> 750$ mph).
   - Displays delay probability, dynamic risk gauge, and **3-Tier Operational Badges**:
     - 🟢 **Low Risk:** $P < 0.33$
     - 🟡 **Moderate Risk:** $0.33 \le P < 0.50$
     - 🔴 **High Delay Alert:** $P \ge 0.50$
3. **📊 Delay Analytics:**
   - Interactive Plotly visualizations derived from verified 2015 EDA:
     - Diurnal hourly delay escalation curve (14% at 05:00 to 48% at 20:00).
     - Carrier delay comparison across all 14 airlines.
     - Monthly seasonality (summer thunderstorms and winter weather freezes).
     - Top origin airport hub congestion rankings.
4. **🔄 Rescheduling Simulator:**
   - Interactive simulator evaluating departure time shifts ($\pm 120$ minutes).
   - Interactive bar chart comparing baseline against alternatives.
   - Ranked options table displaying departure/arrival times, probability deltas, and risk tiers.
5. **🏆 Model Performance:**
   - Consolidated scorecard table covering OLS, Ridge, Lasso, Dummy, GD, Logistic ($\tau=0.50$ and $\tau^*=0.33$), and PCA.
   - Interactive confusion matrices comparing symmetric vs. asymmetric thresholds.
   - Embedded publication figures for ROC/PR curves, probability calibration, and PCA scree plots.

### 7.2 Frontend Integration Safeguards
- Caches the pipeline and model using `@st.cache_resource` (zero retraining on click).
- Validates all 36 input feature names, types, and column orders against the serialized model artifact.
- Suppresses internal stack traces and provides graceful UI warning banners.

---

## 8. Verification & How to Run

### 8.1 Dependencies Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 8.2 Launch the Streamlit Frontend
```bash
streamlit run app/app.py
```
Open `http://localhost:8501` in your browser.

### 8.3 Run Offline Simulator Self-Test
```bash
python -m src.rescheduling_simulator
```

### 8.4 Inspect Step-by-Step Notebooks
```bash
jupyter lab
# 06_pca_experiment.ipynb
# 07_final_model_comparison.ipynb
```

---

## 9. Known Limitations and Phase 2 Future Roadmap

1. **Predeparture Information Horizon:** Scheduled timetable features capture baseline systemic congestion, achieving an ROC-AUC of $\sim 0.60$. Downstream convective weather and airborne holding patterns occur hours after gate pushback and represent irreducible variance at $T - 2$ hours.
2. **Phase 2 Future Extensions:**
   - Real-time FAA Air Traffic Control (ATC) ground-delay program (GDP) feed ingestion.
   - High-resolution terminal aerodrome convective weather radar forecasts (NOAA HRRR).
   - Aircraft tail-number turnaround propagation modeling (tracking inbound aircraft arrival delays).
   - Multi-agent network rescheduling optimization.

---

## Deliverables Index (Member 5)
- `notebooks/06_pca_experiment.ipynb` (PCA dimensionality reduction experiment)
- `notebooks/07_final_model_comparison.ipynb` (Unified model comparison and frozen test evaluation)
- `src/rescheduling_simulator.py` (Offline flight rescheduling simulation engine)
- `app/app.py` (Full-featured Streamlit web application frontend)
- `results/figures/pca/` (Scree plot, loadings, and performance comparison figures)
- `results/figures/comparison/` (ROC/PR curves, confusion matrices, calibration, paradigm figures)
- `reports/member5_final_report.md` (This document)
- `README.md` (Updated project documentation)
- `requirements.txt` (Updated dependencies)
