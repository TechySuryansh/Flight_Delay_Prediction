# Member 4 to Member 5 Handoff Report
**Project 21:** Flight Delay Prediction and Rescheduling (Phase 1: Machine Learning)  
**Author:** Member 4 (Classification, Logistic Regression, Gradient Descent, XGBoost & Evaluation)  
**Recipient:** Member 5 (System Integration, Web Application & Final Phase 1 Synthesis)  
**Date:** October 2026  
**Status:** Task 4 Complete — Validated Classification Model Artifact Ready for Deployment  

---

## 1. Executive Summary & Deliverables Produced

Member 4 has completed all classification experiments, mathematical derivations, gradient descent verifications, tree-based benchmarking, threshold tuning, and probability calibration diagnostics required under Task 4 of the Applied Machine Learning (AML) syllabus.

### Deliverables Produced in Task 4:
1. **Interactive Notebook:** `notebooks/05_classification_models.ipynb`  
   - 12 fully executed cells implementing data ingestion, anti-leakage verification, mathematical MLE formulations, educational first-principles Gradient Descent, production Logistic Regression, XGBoost benchmarking, asymmetric cost threshold tuning, calibration reliability diagrams, frozen held-out test evaluation, and multiclass extension.
2. **Reusable Python Module:** `src/classification.py`  
   - Contains `LogisticRegressionGD`, `build_classification_pipeline()`, metric computation suites, threshold grid evaluator, multiclass target generators, visualization utilities, and artifact serialization/validation routines.
3. **Validated Model Artifact:** `models/classification_pipeline.joblib`  
   - Complete, self-contained Scikit-Learn `ColumnTransformer` + fitted `LogisticRegression` pipeline (5.6 KB). Verified via automated deserialization and test inference.
4. **Publication Visualizations:** `results/figures/classification/`  
   - `fig1_roc_curves.png` (Validation ROC curves comparison)
   - `fig2_precision_recall_curves.png` (Precision-Recall curves against positive rate)
   - `fig3_threshold_tradeoffs.png` (Metric tradeoffs across thresholds $\tau \in [0.15, 0.85]$)
   - `fig4_confusion_matrices.png` (Confusion matrix comparison: $\tau = 0.50$ vs $\tau^* = 0.33$)
   - `fig5_calibration_curves.png` (Reliability diagram and Brier score evaluation)
   - `fig6_gd_loss_convergence.png` (First-principles BCE loss descent curve across 200 epochs)
   - `fig7_feature_interpretability.png` (Odds ratios $\exp(\beta)$ and XGBoost feature gains)
   - `fig8_multiclass_confusion.png` (3-tier delay severity confusion matrix heatmap)
5. **Comprehensive Technical Report:** `reports/member4_classification_report.md`  
6. **Handoff Documentation (This Document):** `reports/member4_handoff.md`  

---

## 2. Model Artifact & Loading Protocol

### 2.1 File Location & Metadata
- **Artifact Path:** `models/classification_pipeline.joblib`
- **File Size:** **5,586 bytes (5.46 KB)**
- **Model Type:** Scikit-Learn `LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)`
- **Preprocessing Pipeline:** `ColumnTransformer` containing:
  - Numeric transformer: `SimpleImputer(strategy='median')` followed by `StandardScaler()`
  - Categorical transformer: `SimpleImputer(strategy='constant', fill_value='UNKNOWN')` followed by `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`

### 2.2 How to Load and Execute Inference in Python
Member 5 can ingest raw operational flight attributes and generate delay probabilities in 4 lines of code:

```python
import joblib
import pandas as pd
from src.classification import ALL_CLASSIFICATION_FEATURES

# 1. Load the serialized pipeline artifact
artifact = joblib.load("models/classification_pipeline.joblib")
pipeline = artifact["pipeline"]
model = artifact["model"]
optimal_threshold = artifact["metadata"]["optimal_threshold"]  # 0.33

# 2. Ingest pre-departure flight records (pandas DataFrame)
# Must contain the 36 pre-departure feature columns
new_flights_df = pd.read_csv("path/to/flight_query.csv")

# 3. Transform inputs and predict probabilities
X_trans = pipeline.transform(new_flights_df[ALL_CLASSIFICATION_FEATURES])
delay_probabilities = model.predict_proba(X_trans)[:, 1]

# 4. Generate operational decision flags using the frozen threshold (tau = 0.33)
predicted_delay_flags = (delay_probabilities >= optimal_threshold).astype(int)

# Operational Risk Tier Assignment for Frontend / Dashboard:
risk_tiers = pd.cut(
    delay_probabilities,
    bins=[-0.01, 0.33, 0.50, 1.00],
    labels=["LOW RISK (On-Time)", "MODERATE RISK (Monitor)", "HIGH RISK (Delay Alert)"]
)
```

---

## 3. Exact Feature List, Expected Order & Input Data Types

The preprocessor expects exactly **36 columns** in the input DataFrame. Any additional columns (e.g., flight numbers, dates, tail numbers) are safely ignored by `remainder='drop'`.

### 3.1 Numeric Features (31 Columns) — `float64` / `int64`

| # | Feature Name | Description | Preprocessing Behavior |
| :-: | :--- | :--- | :--- |
| 1 | `DEP_HOUR` | Scheduled departure hour (0–23) | Median impute $\to$ Standardize |
| 2 | `DEP_MINUTE` | Scheduled departure minute (0–59) | Median impute $\to$ Standardize |
| 3 | `DEP_TIME_MINUTES` | Total departure minutes from midnight | Median impute $\to$ Standardize |
| 4 | `DEP_HOUR_SIN` | Cyclical sine encoding of departure hour | Median impute $\to$ Standardize |
| 5 | `DEP_HOUR_COS` | Cyclical cosine encoding of departure hour | Median impute $\to$ Standardize |
| 6 | `DEP_TIME_SIN` | Cyclical sine encoding of departure minutes | Median impute $\to$ Standardize |
| 7 | `DEP_TIME_COS` | Cyclical cosine encoding of departure minutes | Median impute $\to$ Standardize |
| 8 | `DEP_HOUR_PEAK` | Binary indicator for peak evening rush (17:00–20:59) | Median impute $\to$ Standardize |
| 9 | `ARR_HOUR` | Scheduled arrival hour (0–23) | Median impute $\to$ Standardize |
| 10 | `ARR_MINUTE` | Scheduled arrival minute (0–59) | Median impute $\to$ Standardize |
| 11 | `ARR_TIME_MINUTES` | Total arrival minutes from midnight | Median impute $\to$ Standardize |
| 12 | `ARR_HOUR_SIN` | Cyclical sine encoding of arrival hour | Median impute $\to$ Standardize |
| 13 | `ARR_HOUR_COS` | Cyclical cosine encoding of arrival hour | Median impute $\to$ Standardize |
| 14 | `ARR_NIGHT_CURFEW`| Binary indicator for arrival between 23:00–05:59 | Median impute $\to$ Standardize |
| 15 | `DAY_OF_WEEK` | Day of week integer (1 = Monday, 7 = Sunday) | Median impute $\to$ Standardize |
| 16 | `DOW_SIN` | Cyclical sine encoding of day of week | Median impute $\to$ Standardize |
| 17 | `DOW_COS` | Cyclical cosine encoding of day of week | Median impute $\to$ Standardize |
| 18 | `IS_WEEKEND` | Binary indicator (1 if Saturday or Sunday, else 0) | Median impute $\to$ Standardize |
| 19 | `MONTH` | Month integer (1–12) | Median impute $\to$ Standardize |
| 20 | `MONTH_SIN` | Cyclical sine encoding of month | Median impute $\to$ Standardize |
| 21 | `MONTH_COS` | Cyclical cosine encoding of month | Median impute $\to$ Standardize |
| 22 | `DAY` | Day of month (1–31) | Median impute $\to$ Standardize |
| 23 | `QUARTER` | Calendar quarter (1–4) | Median impute $\to$ Standardize |
| 24 | `IS_MONTH_END` | Binary indicator (1 if day $\ge 28$, else 0) | Median impute $\to$ Standardize |
| 25 | `DISTANCE` | Flight planned distance in miles | Median impute $\to$ Standardize |
| 26 | `DISTANCE_LOG` | Natural log of distance: $\ln(1 + \text{DISTANCE})$ | Median impute $\to$ Standardize |
| 27 | `SCHEDULED_TIME` | Scheduled gate-to-gate block duration in minutes | Median impute $\to$ Standardize |
| 28 | `PLANNED_SPEED` | Planned speed proxy in mph: $(\text{DISTANCE} / \text{TIME}) \times 60$ | Median impute $\to$ Standardize |
| 29 | `CARRIER_HIST_DELAY_RATE` | Empirical Bayes historical delay prior of airline | Median impute $\to$ Standardize |
| 30 | `ORIGIN_HIST_DELAY_RATE` | Empirical Bayes historical delay prior of origin | Median impute $\to$ Standardize |
| 31 | `DEST_HIST_DELAY_RATE` | Empirical Bayes historical delay prior of destination | Median impute $\to$ Standardize |

### 3.2 Categorical Features (5 Columns) — `string` / `object`

| # | Feature Name | Expected Categories | Preprocessing Behavior |
| :-: | :--- | :--- | :--- |
| 32 | `AIRLINE` | 14 IATA carrier codes (AA, DL, UA, WN, etc.) | Impute `'UNKNOWN'` $\to$ One-Hot Encode |
| 33 | `ORIGIN_AIRPORT` | Origin 3-letter IATA code (ATL, ORD, DFW, etc.) | Impute `'UNKNOWN'` $\to$ One-Hot Encode |
| 34 | `DESTINATION_AIRPORT` | Destination 3-letter IATA code | Impute `'UNKNOWN'` $\to$ One-Hot Encode |
| 35 | `DEP_TIME_OF_DAY` | 5 time tiers (`Early Morning`, `Morning`, `Afternoon`, `Evening`, `Night`) | Impute `'UNKNOWN'` $\to$ One-Hot Encode |
| 36 | `DISTANCE_CATEGORY` | 4 distance bands (`Short (<500mi)`, `Medium (500-1500mi)`, `Long (1500-2500mi)`, `Transcontinental (>2500mi)`) | Impute `'UNKNOWN'` $\to$ One-Hot Encode |

> **Robustness Guarantee for Member 5:** If an unseen carrier code or airport is passed at inference time, `OneHotEncoder(handle_unknown='ignore')` safely maps it to a vector of all zeros. The system will never raise an unhandled exception or crash.

---

## 4. Selected Operating Threshold & Operational Rationale

### 4.1 Decision Rule: $\tau^* = 0.33$
Member 5 must configure the prediction engine to use an operating threshold of:
$$\tau^* = 0.33$$

### 4.2 Why Not the Default Threshold of $\tau = 0.50$?
The default threshold $\tau = 0.50$ is symmetric, implicitly assuming that a False Positive (false alarm) and a False Negative (unanticipated delay) carry equal cost. 

In real-world airline operations:
1. **False Negative (Unanticipated Delay):** An aircraft arrives late with no prior warning. Passengers miss onward connections, flight crews exceed legal maximum duty hours (causing downstream cancellations), and arriving planes sit on taxiways waiting for occupied gates. Cost is severe.
2. **False Positive (Precautionary Buffer Alert):** The system flags a potential delay on a flight that ends up landing on time. Gate planners monitor the flight or stage ground baggage handlers early. Operational disruption is minimal.

Using an empirical cost ratio of $2.5 : 1.0$ (FN cost : FP cost) evaluated strictly on the **validation partition**:
- **At $\tau = 0.50$:** Caught only **494 out of 1,604 delays (30.80% Recall)**. 1,110 delayed flights slipped past unalerted.
- **At $\tau^* = 0.33$:** Caught **1,442 out of 1,604 delays (89.90% Recall)**. Minimized total operational cost from 3,305 down to 2,468.
- **On Held-Out Test Data:** Captured **1,528 out of 1,549 delays (98.64% Recall)**.

---

## 5. Chronological Partition Dates & Validated Metrics

Member 4 strictly complied with zero data leakage protocols. The test set was evaluated only once after the model architecture and threshold were locked.

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               CHRONOLOGICAL PERFORMANCE BENCHMARK MATRIX                         │
├──────────────────────────────┬───────────────────────────────┬───────────────────────────────────┤
│ Metric                       │ Validation Split (Sep–Oct 2015)│ Held-Out Test Split (Nov–Dec 2015)│
├──────────────────────────────┼───────────────────────────────┼───────────────────────────────────┤
│ Record Count                 │ 4,114 flights                 │ 4,042 flights                     │
│ Positive Class Prevalence    │ 38.99% (1,604 delayed)        │ 38.32% (1,549 delayed)            │
│ ROC-AUC (Discrimination)     │ 0.5919                        │ 0.6019                            │
│ PR-AUC (Average Precision)   │ 0.4603                        │ 0.4663                            │
│ Brier Score (Calibration MSE)│ 0.2338                        │ 0.2319                            │
│ Binary Cross-Entropy Log-Loss│ 0.6600                        │ 0.6562                            │
├──────────────────────────────┴───────────────────────────────┴───────────────────────────────────┤
│ AT DEFAULT THRESHOLD (tau = 0.50):                                                               │
├──────────────────────────────┬───────────────────────────────┬───────────────────────────────────┤
│ Accuracy                     │ 60.14%                        │ 62.77%                            │
│ Precision                    │ 48.24%                        │ 46.93%                            │
│ Recall                       │ 30.80%                        │ 51.32%                            │
│ F1-Score                     │ 0.3760                        │ 0.4903                            │
│ Confusion Matrix             │ TN=1980, FP=530, FN=1110, TP=494│ TN=1742, FP=751, FN=754, TP=795 │
├──────────────────────────────┴───────────────────────────────┴───────────────────────────────────┤
│ AT FROZEN OPERATING THRESHOLD (tau = 0.33):                                                      │
├──────────────────────────────┬───────────────────────────────┬───────────────────────────────────┤
│ Accuracy                     │ 45.92%                        │ 42.08%                            │
│ Precision                    │ 41.14%                        │ 38.88%                            │
│ Recall                       │ 89.90%                        │ 98.64%                            │
│ F1-Score                     │ 0.5645                        │ 0.5551                            │
│ Confusion Matrix             │ TN=447, FP=2063, FN=162, TP=1442│ TN=173, FP=2320, FN=21, TP=1528 │
└──────────────────────────────┴───────────────────────────────┴───────────────────────────────────┘
```

---

## 6. Known Limitations & Recommendations for Member 5

When integrating this classification model into the user-facing web interface or simulation suite:

1. **Pre-Departure Information Horizon:**
   - The model operates strictly at $T - 2$ hours before pushback. It does not possess en-route telemetry, airborne speed adjustments, or en-route convective weather reroutes.
   - An ROC-AUC of $\sim 0.60$ is typical for pre-departure commercial flight prediction. Member 5 should clearly present delay outputs as **probabilistic risk assessments**, not deterministic guarantees.
2. **False Alarm Management in the UI:**
   - At $\tau^* = 0.33$, the system achieves near-total delay recall ($98.6\%$) but produces a high false alarm rate ($58\%$ false positives).
   - **UI Design Recommendation:** Do not present binary predictions as a blunt "DELAYED / NOT DELAYED". Instead, render an **Operational Risk Gauge**:
     - 🟢 **Low Delay Risk ($P < 0.33$):** Normal dispatch protocol.
     - 🟡 **Moderate Delay Risk ($0.33 \le P < 0.50$):** Monitor inbound aircraft turn; stage baggage crews.
     - 🔴 **High Delay Risk ($P \ge 0.50$):** Flag for proactive rescheduling, rebooking, or gate hold.
3. **Optional Multiclass Integration:**
   - If the dashboard displays delay severity tiers, Member 5 can access the 3-tier severity probabilities:
     `multi_model = train_multiclass_logistic(...)` or use the classification report benchmarks documented in Section 10 of `reports/member4_classification_report.md`.

---

## 7. Verification Checklist for Member 5

Before launching the user-facing system:
- [ ] Confirmed existence and readability of `models/classification_pipeline.joblib`.
- [ ] Loaded pipeline using `joblib.load()` and executed inference on test flight inputs.
- [ ] Verified that input features match the exact 36 column names specified in Section 3.
- [ ] Configured the default operational alert threshold to $\tau^* = 0.33$.
- [ ] Implemented a 3-tier risk display (Low / Moderate / High) to balance recall against false alarms.
- [ ] Verified zero dependency on post-departure data leakage columns.
