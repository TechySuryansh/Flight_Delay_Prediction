# Member 2B to Member 3 Handoff Report
**Project 21:** Flight Delay Prediction and Rescheduling (Phase 1: Machine Learning)  
**Author:** Member 2B (Feature Engineering, Encoding Pipelines & Dataset Preparation)  
**Recipient:** Member 3 (Linear Regression, Ordinary Least Squares & Logistic Regression Baselines)  
**Date:** October 2026  
**Status:** Task 2B Complete — Feature Dataset Ready for Member 3  

---

## 1. Feature Dataset Location & Overview

The model-ready feature dataset has been generated, validated, and saved to:
- **File Path:** `data/processed/flights_features.csv`
- **Total Records:** **24,522 rows** (local benchmark cohort) / compatible with the 5.7M full cohort
- **Total Columns:** **46 columns**
- **File Size:** ~9.5 MB
- **Missing Values:** **0 nulls** across all features and target columns (100% complete)
- **Duplicates:** **0 duplicate rows**

> **Source Integrity Note:** Neither Member 1B's cleaned dataset (`data/processed/flights_cleaned.csv`) nor the raw archives were modified. All engineered features were written strictly to `data/processed/flights_features.csv`.

---

## 2. Chronological Split Partitions (No Random Splitting)

Member 3 must use the pre-assigned chronological partitions stored in the **`SPLIT_SET`** column:

| Partition (`SPLIT_SET`) | Calendar Date Range (2015) | Row Count | Percentage | Delayed Flights | Delay Rate (%) | Continuous Delay Mean |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`train`** | `2015-01-01` to `2015-08-31` (Months 1–8) | **16,366** | 66.74% | 6,323 | 38.63% | +6.74 min |
| **`val`** | `2015-09-01` to `2015-10-31` (Months 9–10) | **4,114** | 16.78% | 1,604 | 38.99% | +7.39 min |
| **`test`** | `2015-11-01` to `2015-12-31` (Months 11–12) | **4,042** | 16.48% | 1,549 | 38.32% | +6.25 min |

### Partition Loading Snippet for Member 3:
```python
import pandas as pd

# Load prepared feature dataset
df = pd.read_csv("data/processed/flights_features.csv")

# Extract non-overlapping chronological partitions
train_df = df[df["SPLIT_SET"] == "train"].copy()
val_df   = df[df["SPLIT_SET"] == "val"].copy()
test_df  = df[df["SPLIT_SET"] == "test"].copy()

print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
```

---

## 3. Strict Column Segregation Architecture

To prevent data leakage during model training, Member 3 must maintain strict separation between features, targets, and identifiers:

### 3.1 Ground-Truth Targets (Never in Feature Matrix $X$)
- **`ARRIVAL_DELAY`** (`float64`): Continuous delay in minutes. **Target for Task 3 OLS Linear Regression experiments.**
- **`DELAYED`** (`int64`): Binary classification label ($1$ if `ARRIVAL_DELAY > 15.0`, else $0$). **Target for Task 3 Logistic Regression baseline experiments.**
- ⚠️ **MANDATORY RULE:** Both columns must be extracted into target vectors $y$ and dropped from feature inputs $X$.

### 3.2 Metadata and Audit Identifiers (Do Not Include in $X$)
The following 5 columns are preserved for audit logging, route evaluation, and partitioning, but should not be fed directly into regression models:
- `FLIGHT_DATE`: Flight date string (`YYYY-MM-DD`).
- `FLIGHT_NUMBER`: Operational carrier flight identifier.
- `TAIL_NUMBER`: Aircraft registration code (imputed `'UNKNOWN'`).
- `ROUTE`: Origin-Destination interaction string (`ORIGIN_AIRPORT + '_' + DESTINATION_AIRPORT`).
- `SPLIT_SET`: Partition indicator (`'train'`, `'val'`, `'test'`).

### 3.3 Candidate Model Features (39 Features Total)

#### Numeric Features (34 Columns):
```python
FEATURE_COLUMNS_NUMERIC = [
    # Scheduled Departure Encodings
    "DEP_HOUR", "DEP_MINUTE", "DEP_TIME_MINUTES",
    "DEP_HOUR_SIN", "DEP_HOUR_COS", "DEP_TIME_SIN", "DEP_TIME_COS",
    "DEP_HOUR_PEAK",
    
    # Scheduled Arrival Encodings
    "ARR_HOUR", "ARR_MINUTE", "ARR_TIME_MINUTES",
    "ARR_HOUR_SIN", "ARR_HOUR_COS", "ARR_NIGHT_CURFEW",
    
    # Calendar & Temporal Encodings
    "DAY_OF_WEEK", "DOW_SIN", "DOW_COS", "IS_WEEKEND",
    "MONTH", "MONTH_SIN", "MONTH_COS", "DAY", "QUARTER", "IS_MONTH_END",
    
    # Distance & Operational Buffer Proxy
    "DISTANCE", "DISTANCE_LOG", "SCHEDULED_TIME", "PLANNED_SPEED",
    
    # Chronological Historical Delay Priors (Empirical Bayes Smoothed)
    "CARRIER_HIST_DELAY_RATE", "ORIGIN_HIST_DELAY_RATE", "DEST_HIST_DELAY_RATE"
]
```

#### Categorical Features (5 Columns):
```python
FEATURE_COLUMNS_CATEGORICAL = [
    "AIRLINE",             # 14 operating carriers
    "ORIGIN_AIRPORT",      # Origin IATA code
    "DESTINATION_AIRPORT", # Destination IATA code
    "DEP_TIME_OF_DAY",     # 5 tiers: Early Morning, Morning, Afternoon, Evening, Night
    "DISTANCE_CATEGORY"    # 3 tiers: Short, Medium, Long
]
```

---

## 4. Reusable Preprocessing Pipeline Ingestion

Member 2B has implemented `build_feature_pipeline()` in `src/features.py`. Member 3 can import and execute the pipeline directly:

```python
import numpy as np
import pandas as pd
from src.features import (
    FEATURE_COLUMNS_ALL,
    FEATURE_COLUMNS_NUMERIC,
    FEATURE_COLUMNS_CATEGORICAL,
    build_feature_pipeline,
)

# 1. Load data and split
df = pd.read_csv("data/processed/flights_features.csv")
train_df = df[df["SPLIT_SET"] == "train"]
val_df   = df[df["SPLIT_SET"] == "val"]
test_df  = df[df["SPLIT_SET"] == "test"]

# 2. Isolate feature matrices and targets
X_train = train_df[FEATURE_COLUMNS_ALL]
y_train_reg = train_df["ARRIVAL_DELAY"]
y_train_clf = train_df["DELAYED"]

X_val = val_df[FEATURE_COLUMNS_ALL]
y_val_reg = val_df["ARRIVAL_DELAY"]
y_val_clf = val_df["DELAYED"]

X_test = test_df[FEATURE_COLUMNS_ALL]
y_test_reg = test_df["ARRIVAL_DELAY"]
y_test_clf = test_df["DELAYED"]

# 3. Instantiate pipeline
preprocessor = build_feature_pipeline(
    numeric_features=FEATURE_COLUMNS_NUMERIC,
    categorical_features=FEATURE_COLUMNS_CATEGORICAL,
)

# 4. FIT STRICTLY ON X_TRAIN ONLY (Never fit on val or test!)
preprocessor.fit(X_train)

# 5. Transform all partitions
X_train_trans = preprocessor.transform(X_train)
X_val_trans   = preprocessor.transform(X_val)
X_test_trans  = preprocessor.transform(X_test)

print(f"Transformed X_train shape: {X_train_trans.shape}")  # (16366, 81)
print(f"Transformed X_val   shape: {X_val_trans.shape}")    # (4114, 81)
print(f"Transformed X_test  shape: {X_test_trans.shape}")   # (4042, 81)
```

---

## 5. Directives and Academic Syllabus Boundaries for Member 3

Per the academic project syllabus constraints:

### Permitted Modeling Approaches for Member 3:
1. **Simple Linear Regression (OLS):** Using single features (e.g. `DEP_HOUR` or `PLANNED_SPEED`) to predict `ARRIVAL_DELAY`.
2. **Multiple Linear Regression (OLS):** Closed-form normal equations $\hat{\beta} = (X^T X)^{-1} X^T y$ and scikit-learn `LinearRegression()`.
3. **Regression Metrics:** Evaluate using Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Coefficient of Determination ($R^2$).
4. **Baseline Logistic Regression:** Fit `LogisticRegression()` using Maximum Likelihood Estimation (MLE) to predict binary target `DELAYED`.
5. **Classification Metrics:** Evaluate using Accuracy, Precision, Recall, F1-Score, PR-AUC, and ROC-AUC.
6. **Dummy Baseline Models:** Compare against dummy baselines (e.g. always predicting majority class `0` or predicting mean arrival delay).

### Prohibited Topics (Save for Later Members / Disallowed):
- ❌ **No Tree Models:** Do NOT use Random Forest, Extra Trees, Gradient Boosting, or XGBoost.
- ❌ **No Deep Learning:** Do NOT use PyTorch, TensorFlow, neural networks, or embeddings.
- ❌ **No PCA or Feature Selection:** Member 4 will handle PCA and feature selection.
- ❌ **No Rescheduling Simulation:** Reserved for Phase 2.

---

## 6. Verification Checklist for Member 3

Before concluding Task 3 experiments, verify:
- [ ] Loaded `data/processed/flights_features.csv` without modifying the file.
- [ ] Split records using `SPLIT_SET == 'train'`, `'val'`, and `'test'`.
- [ ] Separated `ARRIVAL_DELAY` and `DELAYED` into $y$ and ensured they never entered $X$.
- [ ] Fit `ColumnTransformer` (or scalers) **strictly on training data**.
- [ ] Evaluated models on validation split (`SPLIT_SET == 'val'`) without tuning hyperparameters on the test split.
- [ ] Documented baseline metrics (Dummy Classifier vs. Logistic Regression; Mean Predictor vs. OLS).
