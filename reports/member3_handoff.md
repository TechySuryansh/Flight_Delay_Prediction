# Member 3 to Member 4 Handoff Report
**Project 21:** Flight Delay Prediction and Rescheduling (Phase 1: Machine Learning)  
**Author:** Member 3 (Linear Regression, OLS, Gradient Descent & Regularization)  
**Recipient:** Member 4 (Classification, Logistic Regression, Gradient Descent for Classification & Evaluation)  
**Date:** October 2026  
**Status:** Task 3 Complete — Regression Benchmarks Established; Ready for Classification  

---

## 1. Executive Summary & Deliverables Produced

Member 3 has completed all regression experiments required under topics 2–10 of the Applied Machine Learning (AML) syllabus.

### Deliverables Produced in Task 3:
1. **Interactive Notebook:** `notebooks/04_regression_experiments.ipynb`  
   - 24 executable cells implementing Simple OLS, Multiple OLS, Batch GD, Mini-Batch GD, SGD, Polynomial expansions, Gauss-Markov residual diagnostics, Bias-Variance tradeoff curves, feature selection, and Ridge/Lasso regularization paths.
2. **Reusable Python Module:** `src/regression_experiments.py`  
   - First-principles BGD, MBGD, and SGD regressors; assumption evaluators; feature selectors; and Ridge/Lasso tuning functions.
3. **Publication Visualizations:** `results/figures/regression/`  
   - `fig1_simple_ols_comparison.png` (Simple OLS comparison)
   - `fig2_gradient_descent_convergence.png` (BGD vs. MBGD vs. SGD convergence curves)
   - `fig3_residual_diagnostics.png` (4-panel Gauss-Markov diagnostics)
   - `fig4_bias_variance_tradeoff.png` (Complexity curves across degrees 1–3)
   - `fig5_regularization_paths.png` (Ridge vs. Lasso coefficient paths and sparsity)
4. **Comprehensive Regression Report:** `reports/member3_regression_report.md`  
5. **Handoff Documentation (This Document):** `reports/member3_handoff.md`  

---

## 2. Key Insights for Member 4: Transition from Regression to Classification

### 2.1 Why Continuous Delay Regression Has Low $R^2$ (~0.04–0.05)
Our regression experiments empirically revealed that pre-departure linear regression achieves an $R^2 \approx 0.04\text{--}0.05$:
- **Root Cause:** En-route weather shifts, convective storm fronts, airspace holding patterns, and airport surface taxiway congestion occur *hours after gate departure*.
- These downstream operational shocks represent high **irreducible variance** ($\sigma^2$) that cannot be forecasted from pre-departure scheduled attributes alone.
- Predicting the exact delay minute ($\pm 31$ min RMSE) is intrinsically noisy.

### 2.2 Why Binary Classification is the Right Project Framing
In operational airline management, gate agents and passengers do not require exact minute-level forecasts; they need to know **whether a flight will suffer a meaningful operational delay ($\ge 15$ minutes)** to initiate rebooking, crew rescheduling, or gate swaps.
- Predicting the binary threshold `DELAYED == 1` absorbs individual minute-level fluctuations into a robust probabilistic signal.
- Member 4 is responsible for implementing this primary project objective using **Logistic Regression**.

---

## 3. Ingestion Guide and Target Specification for Member 4

### 3.1 Dataset Location
- **File Path:** `data/processed/flights_features.csv`
- **Total Records:** 24,522 rows, 46 columns, 0 missing values.

### 3.2 Target Variable for Member 4
- **Target Column:** **`DELAYED`**
- **Data Type:** `int64` ($1$ if `ARRIVAL_DELAY > 15.0` minutes, else $0$).
- **Distribution Across Partitions:**
  - **Train (`SPLIT_SET == 'train'`):** 16,366 flights | **38.63% Delayed** (6,323 delayed / 10,043 on-time)
  - **Validation (`SPLIT_SET == 'val'`):** 4,114 flights | **38.99% Delayed** (1,604 delayed / 2,510 on-time)
  - **Held-Out Test (`SPLIT_SET == 'test'`):** 4,042 flights | **38.32% Delayed** (1,549 delayed / 2,493 on-time)
  - *Class Imbalance:* Approximately **1.6 : 1** (On-Time to Delayed) in this benchmark cohort (~4.58 : 1 in the full national dataset).

### 3.3 Target Isolation Rule
- ⚠️ **MANDATORY:** Drop both `DELAYED` and continuous `ARRIVAL_DELAY` from feature inputs $X$.

---

## 4. Candidate Features Recommended by Task 3 Experiments

Based on Task 3's OLS, feature selection, and Lasso sparsity results, Member 4 should prioritize these predictive signals:

```python
# Confirmed Top Predictors for Classification Modeling:
RECOMMENDED_NUMERIC_FEATURES = [
    # Scheduled Departure Temporal Signals (Strongest Predictors)
    "DEP_HOUR", "DEP_HOUR_PEAK", "DEP_HOUR_SIN", "DEP_HOUR_COS", "DEP_TIME_SIN", "DEP_TIME_COS",
    
    # Scheduled Arrival Signals
    "ARR_HOUR_SIN", "ARR_HOUR_COS", "ARR_NIGHT_CURFEW",
    
    # Flight Geometry & Schedule Buffer Proxy
    "PLANNED_SPEED", "DISTANCE_LOG",
    
    # Chronological Historical Delay Priors (Empirical Bayes Smoothed)
    "CARRIER_HIST_DELAY_RATE", "ORIGIN_HIST_DELAY_RATE", "DEST_HIST_DELAY_RATE",
    
    # Calendar Coordinates
    "IS_WEEKEND", "MONTH_SIN", "MONTH_COS"
]

RECOMMENDED_CATEGORICAL_FEATURES = [
    "AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "DEP_TIME_OF_DAY", "DISTANCE_CATEGORY"
]
```

---

## 5. Quick-Start Ingestion Snippet for Member 4

```python
import pandas as pd
import numpy as np
from src.features import build_feature_pipeline

# 1. Load feature dataset
df = pd.read_csv("data/processed/flights_features.csv")

# 2. Extract chronological partitions
train_df = df[df["SPLIT_SET"] == "train"].copy()
val_df   = df[df["SPLIT_SET"] == "val"].copy()
test_df  = df[df["SPLIT_SET"] == "test"].copy()

# 3. Extract feature matrix X and classification target y
feature_cols = [c for c in df.columns if c not in ["DELAYED", "ARRIVAL_DELAY", "FLIGHT_DATE", "FLIGHT_NUMBER", "TAIL_NUMBER", "ROUTE", "SPLIT_SET", "YEAR"]]

X_train = train_df[feature_cols]
y_train = train_df["DELAYED"].values

X_val = val_df[feature_cols]
y_val = val_df["DELAYED"].values

X_test = test_df[feature_cols]
y_test = test_df["DELAYED"].values

# 4. Fit preprocessing pipeline STRICTLY on X_train only
preprocessor = build_feature_pipeline()
preprocessor.fit(X_train)

X_train_trans = preprocessor.transform(X_train)
X_val_trans   = preprocessor.transform(X_val)
X_test_trans  = preprocessor.transform(X_test)

print(f"X_train Transformed Shape: {X_train_trans.shape}")
print(f"y_train Delayed Proportion: {y_train.mean():.2%}")
```

---

## 6. Syllabus Boundaries & Directives for Member 4

Per the AML academic syllabus:

### Permitted Modeling Scope for Member 4:
1. **Logistic Regression via Maximum Likelihood Estimation (MLE):**
   - Sigmoid hypothesis function: $h_\theta(x) = \sigma(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$.
   - Cross-entropy loss: $J(\theta) = -\frac{1}{m} \sum_{i=1}^m [y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)}))]$.
2. **Gradient Descent for Logistic Regression:**
   - Implement Batch Gradient Descent and Mini-Batch Gradient Descent with cross-entropy gradients:
     $$\nabla_\theta J(\theta) = \frac{1}{m} X^T (\sigma(X\theta) - y)$$
3. **Regularized Logistic Regression:**
   - $L_1$ (Lasso) and $L_2$ (Ridge) penalties with threshold tuning.
4. **Classification Evaluation Metrics:**
   - Accuracy, Precision, Recall, F1-Score, PR-AUC, and ROC-AUC.
   - Confusion Matrix analysis (True Positives, False Positives, True Negatives, False Negatives).
   - Precision-Recall curves (critical due to class imbalance).
5. **Baseline Comparison:**
   - Compare against a Majority Class Dummy Classifier (predicting all 0s yields 61.36% accuracy on benchmark, but 0.00 recall).

### Prohibited Topics (Save for Later / Strictly Excluded):
- ❌ **No Tree Models:** Do NOT use Random Forest, Extra Trees, Gradient Boosting, or XGBoost.
- ❌ **No Deep Learning:** Do NOT use PyTorch, TensorFlow, or neural networks.
- ❌ **No Rescheduling Simulation:** Reserved for Phase 2.

---

## 7. Verification Checklist for Member 4

Before concluding Task 4:
- [ ] Ingested `data/processed/flights_features.csv` without modifying the file.
- [ ] Used `DELAYED` as the classification target.
- [ ] Ensured `ARRIVAL_DELAY` and `DELAYED` never entered feature inputs $X$.
- [ ] Fit `ColumnTransformer` (or scalers) **strictly on training data**.
- [ ] Evaluated Logistic Regression via cross-entropy loss.
- [ ] Compared Batch vs. Mini-Batch Gradient Descent for classification.
- [ ] Evaluated Accuracy, Precision, Recall, F1, and ROC-AUC on the validation split.
- [ ] Handed off to Member 5 for final project synthesis.
