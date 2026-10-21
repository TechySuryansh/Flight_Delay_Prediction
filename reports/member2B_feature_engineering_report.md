# Member 2B: Feature Engineering and Prepared Feature Dataset Report
**Project 21:** Flight Delay Prediction and Rescheduling  
**Team Role:** Member 2B (Feature Engineering, Encoding Pipelines & Dataset Preparation)  
**Phase:** AML Phase 1 — Traditional Machine Learning  
**Date:** October 2026  
**Status:** Task 2B Complete and Validated  

---

## Executive Summary

This report formalizes the feature engineering, chronological dataset partitioning, and model-ready dataset preparation executed under **Task 2B** of AML Phase 1. Building upon the Exploratory Data Analysis and Descriptive Time-Series findings established by **Member 2A** (`reports/member2A_eda_report.md`, `reports/member2A_handoff.md`, `notebooks/03a_eda_time_series.ipynb`) and the cleaned operational cohort produced by **Member 1B** (`data/processed/flights_cleaned.csv`), Member 2B has developed a leakage-safe feature engineering pipeline, constructed reusable scikit-learn transformers, established strict chronological train/validation/test partitions, and generated the verified model-ready dataset at `data/processed/flights_features.csv`.

### Key Results Summary:
1. **Operational Prediction Point ($T - 2$ Hours Before Departure):**
   - The decision boundary is established at exactly 120 minutes prior to scheduled gate departure.
   - At this juncture, only planned timetable parameters, calendar/seasonal coordinates, and strictly prior historical outcomes are available.
   - All 17 post-departure and outcome variables (`DEPARTURE_TIME`, `DEPARTURE_DELAY`, `TAXI_OUT`, `WHEELS_OFF`, `ELAPSED_TIME`, `AIR_TIME`, `WHEELS_ON`, `TAXI_IN`, `ARRIVAL_TIME`, `ARRIVAL_DELAY`, `CANCELLED`, `DIVERTED`, and delay cause breakdowns) are quarantined.
2. **Chronological Partitioning (Zero Date Overlap):**
   - **Training Set (Months 1–8: Jan 1 to Aug 31, 2015):** 16,366 rows (66.74%), delay rate = **38.63%**.
   - **Validation Set (Months 9–10: Sep 1 to Oct 31, 2015):** 4,114 rows (16.78%), delay rate = **38.99%**.
   - **Test Set (Months 11–12: Nov 1 to Dec 31, 2015):** 4,042 rows (16.48%), delay rate = **38.32%**.
   - Non-overlapping date boundaries: $\max(\text{Train}) < \min(\text{Val}) \le \max(\text{Val}) < \min(\text{Test})$.
3. **Leakage-Safe Predeparture Features:**
   - **Cyclical Temporal Encodings:** Continuous $\sin/\cos$ transformations for 24-hour departure and arrival hours, 1440-minute day cycle, 7-day week, and 12-month calendar year.
   - **Diurnal Escalation Indicators:** Binary peak rush flag (`DEP_HOUR_PEAK` for 17:00–22:00) and operational time-of-day categories (`Early Morning`, `Morning`, `Afternoon`, `Evening`, `Night`).
   - **Flight Geometry & Buffer Proxy:** Log-transformed distance, FAA distance categories (`Short`, `Medium`, `Long`), and planned gate-to-gate speed buffer ($\text{DISTANCE} / \text{SCHEDULED\_TIME}$).
   - **Categoricals & Interactions:** Route pair (`ORIGIN_AIRPORT + '_' + DESTINATION_AIRPORT`), carrier, and airport codes.
4. **Chronological Historical Delay Priors (Empirical Bayes Target Encoding):**
   - Computed carrier, origin airport, and destination airport historical delay priors strictly on the training partition using Bayesian smoothing ($m = 10.0$ pseudo-counts).
   - Within the training partition, computed 5-fold Out-Of-Fold (OOF) target encodings to ensure no observation observes its own target outcome.
   - Mapped forward to validation and test partitions with fallback to the global training baseline.
   - Formally asserted zero contamination from future partitions into historical priors.
5. **Reusable Scikit-Learn Pipeline (`src/features.py`):**
   - Built `build_feature_pipeline()` utilizing `ColumnTransformer` with `StandardScaler` for numeric inputs and `OneHotEncoder(handle_unknown='ignore')` for categorical inputs.
   - Fit strictly on `X_train` only.
6. **Dataset Validation & Integrity:**
   - 24,522 rows in = 24,522 rows out (0 dropped, 0 duplicates, 0 unresolved nulls, 0 infinite values).
   - Ground-truth target columns (`DELAYED`, `ARRIVAL_DELAY`) and audit identifiers (`FLIGHT_DATE`, `FLIGHT_NUMBER`, `TAIL_NUMBER`, `ROUTE`, `SPLIT_SET`) strictly isolated from feature matrix $X$.
   - Prepared file saved to `data/processed/flights_features.csv` (9.5 MB, 46 columns).

---

## 1. Review of Member 2A Findings and Reconciliation

Member 2B thoroughly examined:
- `reports/member2A_eda_report.md`
- `reports/member2A_handoff.md`
- `notebooks/03a_eda_time_series.ipynb`

### Reconciliation Matrix:
| Member 2A Finding / Guidance | Member 2B Engineering Action | Validation Result |
| :--- | :--- | :--- |
| **Diurnal Delay Escalation:** Delay probability rises monotonically from ~26% at 06:00 to ~52% at 22:00 | Engineered `DEP_HOUR_SIN`, `DEP_HOUR_COS`, `DEP_TIME_SIN`, `DEP_TIME_COS`, `DEP_HOUR_PEAK`, and `DEP_TIME_OF_DAY` | Captured circular continuity and evening surge without arbitrary step artifacts |
| **Weak Distance Correlation:** Modest variation across distance bands (~37.4% to ~40.1%) | Log-transformed distance `DISTANCE_LOG` and derived planned speed buffer `PLANNED_SPEED` ($\text{miles}/\text{min}$) | Preserved scale normalization for linear models while adding buffer proxy |
| **Airline & Airport Heterogeneity:** Carrier delay rates vary across airlines; origin hubs differ in congestion | Derived Empirical Bayes smoothed delay rates `CARRIER_HIST_DELAY_RATE`, `ORIGIN_HIST_DELAY_RATE`, `DEST_HIST_DELAY_RATE` | Fit strictly on training partition with 5-fold OOF encoding on train |
| **Chronological Partitioning:** Train (Jan–Aug), Val (Sep–Oct), Test (Nov–Dec) strictly by date | Added standardized `SPLIT_SET` column (`'train'`, `'val'`, `'test'`) | Verified zero date overlap between partitions |
| **Tail Number Missingness:** 65 missing tail numbers (0.27%) | Imputed missing with `'UNKNOWN'`, retained as identifier, excluded from linear model feature inputs | 0 nulls remain; no extreme dimensional explosion |

---

## 2. Operational Prediction Point and Leakage Prevention

### 2.1 The Operational Horizon: $T - 2$ Hours
To ensure academic validity and operational relevance:
- **Prediction Point Definition:** Predictions are generated **exactly 120 minutes prior to scheduled gate departure**.
- **Operational Justification:**
  - In commercial airline dispatch, flight release paperwork and preliminary weight/balance plans are finalized 60–120 minutes before pushback.
  - Passenger rebooking and ground-handling logistics require advance notice to stage alternate aircraft, re-route connecting travelers, or coordinate gate swaps.
  - Making predictions at flight departure or mid-flight renders the model useless for proactive delay mitigation.

### 2.2 Prohibited Feature Audit
All 17 post-departure variables identified by Members 1A, 1B, and 2A are strictly excluded from the model input feature list:
- `DEPARTURE_TIME`, `DEPARTURE_DELAY` (actual pushback)
- `TAXI_OUT`, `WHEELS_OFF` (surface movement & takeoff)
- `ELAPSED_TIME`, `AIR_TIME` (airborne flight duration)
- `WHEELS_ON`, `TAXI_IN`, `ARRIVAL_TIME` (landing & gate arrival)
- `ARRIVAL_DELAY` (ground-truth target, retained in dataset for evaluation only)
- Delay breakdown causes (`WEATHER_DELAY`, `LATE_AIRCRAFT_DELAY`, `AIR_SYSTEM_DELAY`, `SECURITY_DELAY`, `AIRLINE_DELAY`, `CANCELLATION_REASON`)

### 2.3 Strict Column Segregation Architecture
To prevent accidental leakage during modeling:
```
flights_features.csv (46 Columns Total)
├── TARGET_COLUMNS (2 Columns - Never in Model Feature List X)
│   ├── DELAYED (Binary classification label: 1 if ARRIVAL_DELAY > 15 min, else 0)
│   └── ARRIVAL_DELAY (Continuous ground-truth delay minutes for regression)
├── IDENTIFIER_COLUMNS (5 Columns - Never in Model Feature List X)
│   ├── FLIGHT_DATE (Chronological date YYYY-MM-DD)
│   ├── FLIGHT_NUMBER (Arbitrary flight index)
│   ├── TAIL_NUMBER (Aircraft registration, imputed 'UNKNOWN')
│   ├── ROUTE (Origin_Destination interaction)
│   └── SPLIT_SET (Chronological partition: 'train', 'val', 'test')
└── MODEL_FEATURE_COLUMNS (39 Candidate Columns)
    ├── Numeric Features (34 Columns: temporal, cyclical, distance, buffer, priors)
    └── Categorical Features (5 Columns: AIRLINE, ORIGIN, DEST, TIME_OF_DAY, DIST_CAT)
```

---

## 3. Chronological Splitting Methodology

### 3.1 The Flaw of Random Splitting in Flight Delay Modeling
Randomly shuffling flight records across time introduces severe **temporal data leakage**:
1. Flights on December 20, 2015 could appear in the training set while predicting on flights from March 15, 2015.
2. In reality, machine learning models deployed in aviation must predict on future, unobserved operating days using only past historical data.
3. Random splits artificially inflate validation metrics by allowing models to memorize seasonal weather regimes and network-wide delay shocks.

### 3.2 Partition Architecture
Partitioning was executed strictly on `FLIGHT_DATE` across the 2015 calendar year:

| Partition | Calendar Date Range | Total Days | Record Count | Proportion (%) | Delayed Flights | Delay Rate (%) | Avg Delay (min) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TRAIN** | `2015-01-01` to `2015-08-31` | 243 | 16,366 | 66.74% | 6,323 | 38.63% | 6.74 |
| **VALIDATION** | `2015-09-01` to `2015-10-31` | 61 | 4,114 | 16.78% | 1,604 | 38.99% | 7.39 |
| **TEST** | `2015-11-01` to `2015-12-31` | 61 | 4,042 | 16.48% | 1,549 | 38.32% | 6.25 |
| **Total / Global** | `2015-01-01` to `2015-12-31` | 365 | 24,522 | 100.00% | 9,476 | 38.64% | 6.77 |

### 3.3 Verification of Non-Overlap
An automated assertion verifies that:
$$\max(\text{Train Dates}) = 2015-08-31 < \min(\text{Val Dates}) = 2015-09-01$$
$$\max(\text{Val Dates}) = 2015-10-31 < \min(\text{Test Dates}) = 2015-11-01$$
Zero records or flight dates are shared across partitions.

---

## 4. Feature Engineering Specifications & Mathematical Formulations

### 4.1 Temporal Cyclical Encodings
Standard numerical encodings for time (e.g. Hour = 0 to 23) create an artificial discontinuity between 23:59 and 00:00 (distance of 23 units instead of 1 unit). We project time onto circular trigonometric coordinates:

1. **Hourly Cyclical Coordinates (24-Hour Period):**
   $$\text{DEP\_HOUR\_SIN} = \sin\left(\frac{2\pi \cdot \text{DEP\_HOUR}}{24}\right), \quad \text{DEP\_HOUR\_COS} = \cos\left(\frac{2\pi \cdot \text{DEP\_HOUR}}{24}\right)$$
   $$\text{ARR\_HOUR\_SIN} = \sin\left(\frac{2\pi \cdot \text{ARR\_HOUR}}{24}\right), \quad \text{ARR\_HOUR\_COS} = \cos\left(\frac{2\pi \cdot \text{ARR\_HOUR}}{24}\right)$$
2. **Continuous Minute-Level Coordinates (1440-Minute Day):**
   $$\text{DEP\_TIME\_MINUTES} = \text{DEP\_HOUR} \cdot 60 + \text{DEP\_MINUTE}$$
   $$\text{DEP\_TIME\_SIN} = \sin\left(\frac{2\pi \cdot \text{DEP\_TIME\_MINUTES}}{1440}\right), \quad \text{DEP\_TIME\_COS} = \cos\left(\frac{2\pi \cdot \text{DEP\_TIME\_MINUTES}}{1440}\right)$$
3. **Calendar Cyclical Coordinates:**
   $$\text{DOW\_SIN} = \sin\left(\frac{2\pi \cdot (\text{DAY\_OF\_WEEK} - 1)}{7}\right), \quad \text{DOW\_COS} = \cos\left(\frac{2\pi \cdot (\text{DAY\_OF\_WEEK} - 1)}{7}\right)$$
   $$\text{MONTH\_SIN} = \sin\left(\frac{2\pi \cdot (\text{MONTH} - 1)}{12}\right), \quad \text{MONTH\_COS} = \cos\left(\frac{2\pi \cdot (\text{MONTH} - 1)}{12}\right)$$

### 4.2 Operational Time-of-Day Bins & Peak Delay Indicators
- **`DEP_HOUR_PEAK`:** Binary flag (1 if `DEP_HOUR` $\in \{17, 18, 19, 20, 21, 22\}$, else 0). Captures the afternoon/evening network congestion cascade where delay rates peak at 45.3%–51.5%.
- **`DEP_TIME_OF_DAY`:** 5 operational tiers:
  - `Early Morning`: 04:00–07:59 (low delay rate ~26.6%)
  - `Morning`: 08:00–11:59 (moderate delay rate ~32.7%)
  - `Afternoon`: 12:00–16:59 (escalating delay rate ~38.3%)
  - `Evening`: 17:00–21:59 (severe peak delay rate ~48.0%)
  - `Night`: 22:00–03:59 (elevated delay rate ~51.5%)
- **`ARR_NIGHT_CURFEW`:** Binary flag (1 if `ARR_HOUR` $\ge 23$ or $< 6$, else 0). Captures night arrival restrictions and reduced ground support staffing.
- **`IS_WEEKEND`:** Binary flag (1 if `DAY_OF_WEEK` $\in \{6, 7\}$, else 0).
- **`QUARTER`:** Calendar quarter (1 to 4).
- **`IS_MONTH_END`:** Binary flag (1 if `DAY` $\ge 28$, else 0).

### 4.3 Flight Geometry and Planned Buffer Proxy
- **`DISTANCE_LOG`:** Natural log transformation:
  $$\text{DISTANCE\_LOG} = \ln(1 + \text{DISTANCE})$$
  Compresses right-skewed distance distributions, stabilizing linear regression gradients.
- **`DISTANCE_CATEGORY`:** FAA operational tiers: `Short` ($< 500$ mi), `Medium` ($500\text{--}1500$ mi), `Long` ($\ge 1500$ mi).
- **`PLANNED_SPEED` (Buffer Proxy):**
  $$\text{PLANNED\_SPEED} = \frac{\text{DISTANCE}}{\max(1.0, \text{SCHEDULED\_TIME})} \quad (\text{miles per minute})$$
  - In aviation operations, airlines add "schedule padding" (buffer time) to congested routes.
  - A lower planned speed indicates a heavily padded schedule, providing built-in resilience against arrival delays.
  - A higher planned speed represents a tight block schedule where any minor ground delay directly cascades into late arrival.

### 4.4 Features Deliberately Excluded from Default Model Input List
1. **`YEAR`:** In this dataset, `YEAR` is constant ($2015$) across all rows. It has zero sample variance ($\sigma^2 = 0$). Scaling a zero-variance column causes division by zero. Excluded from $X$.
2. **`FLIGHT_NUMBER`:** An arbitrary operational identifier assigned by carriers. It possesses 7,117 unique integers in the sample. Treating it as a continuous number creates nonsense linear slopes; one-hot encoding creates excessive dimensionality ($>7,000$ sparse columns). Retained as an audit identifier only.
3. **`TAIL_NUMBER`:** Aircraft registration identifier with 899 unique categories and 65 missing values (imputed as `'UNKNOWN'`). Excluded from standard linear models to prevent extreme sparsity and severe overfitting. Retained for fleet auditing.
4. **`ROUTE`:** Interaction string (`ORIGIN_AIRPORT + '_' + DESTINATION_AIRPORT`) with 210 unique pairs. Retained for route-level filtering and analysis, while component airports are modeled via categorical encoding and priors.

---

## 5. Chronological Historical Features & Anti-Leakage Safeguards

### 5.1 The Danger of In-Sample Target Leakage
Computing historical statistics (e.g. carrier delay rates) across the full dataset causes fatal data leakage. The model learns to rely on future knowledge that is impossible to possess in production.

### 5.2 Empirical Bayes Smoothed Priors
We implement Bayesian smoothed target encoding fit **strictly on the training partition**:
$$\text{Prior}(c) = \frac{\sum_{i \in \text{Train}_c} y_i + m \cdot \bar{y}_{\text{train}}}{N_{c,\text{train}} + m}$$
- $m = 10.0$: Prior weight (pseudo-counts). For low-volume carriers/airports, the estimate shrinks safely toward the global training baseline $\bar{y}_{\text{train}} = 38.63\%$. For high-volume categories, the estimate converges to the empirical category mean.
- **Priors Computed:**
  1. `CARRIER_HIST_DELAY_RATE`: Smoothed carrier historical delay rate.
  2. `ORIGIN_HIST_DELAY_RATE`: Smoothed origin airport historical delay rate.
  3. `DEST_HIST_DELAY_RATE`: Smoothed destination airport historical delay rate.

### 5.3 Out-of-Fold (OOF) Encoding for the Training Partition
If training observations are mapped to the mean of their own partition, they still leak their own individual target label into their feature vector.
To eliminate self-target leakage:
- 5-fold cross-validation is performed **strictly within the training partition**.
- For each fold, priors are estimated on the remaining 4 folds and applied to the held-out fold.
- **Guarantee:** No training observation's target outcome is included in its own historical delay feature.

### 5.4 Out-of-Sample Mapping for Validation and Test
- Validation and test observations receive priors fit strictly on the completed training partition.
- Unseen categories in validation or test are cleanly imputed with the global training mean $\bar{y}_{\text{train}}$.
- **Guarantee:** Not a single target label from September 1 to December 31 was ever accessed during prior estimation.

---

## 6. Reusable Scikit-Learn Preprocessing Pipeline

To ensure that downstream modeling by Member 3 (Linear & Logistic Regression) and Member 4 (Gradient Descent & Classification) strictly follows professional ML practices, `src/features.py` exports `build_feature_pipeline()`.

### Pipeline Architecture:
```python
ColumnTransformer(transformers=[
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), FEATURE_COLUMNS_NUMERIC),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), FEATURE_COLUMNS_CATEGORICAL)
])
```

### Preprocessing Rules:
1. **Fit Strictly on `X_train`:**
   ```python
   preprocessor.fit(X_train)
   ```
2. **Transform Partitions Separately:**
   ```python
   X_train_trans = preprocessor.transform(X_train)
   X_val_trans   = preprocessor.transform(X_val)
   X_test_trans  = preprocessor.transform(X_test)
   ```
3. **Dimensions:**
   - Input: 34 numeric features + 5 categorical features = 39 features.
   - Output after one-hot encoding: **81 dense numeric columns** (standardized and complete).
   - Zero NaNs or Infs present in transformed feature matrices.

---

## 7. Comprehensive Validation Suite & Quality Audit

The dataset was subjected to automated unit assertions via `validate_feature_dataset()`:

| Quality Dimension | Test Specification | Audit Value | Result |
| :--- | :--- | :---: | :---: |
| **Row Count Fidelity** | Output rows must match input cleaned rows exactly | 24,522 rows | Passed |
| **Deduplication** | Zero duplicate records across all 46 columns | 0 duplicates | Passed |
| **Target Isolation** | `DELAYED` and `ARRIVAL_DELAY` strictly excluded from feature list $X$ | 0 target columns in $X$ | Passed |
| **Leakage Quarantine** | All 17 prohibited post-departure columns absent from feature list $X$ | 0 leakage columns in $X$ | Passed |
| **Chronological Separation** | Train dates strictly precede Val dates; Val dates strictly precede Test dates | Non-overlapping | Passed |
| **Feature Completeness** | Zero unresolved nulls in numeric and categorical candidate features | 0 nulls | Passed |
| **Numeric Stability** | Zero infinite (`np.inf`) values across all engineered numeric features | 0 infinite values | Passed |
| **Historical Priors** | All Empirical Bayes delay rates strictly within $[0.0, 1.0]$ | In bounds | Passed |

---

## 8. Summary of Deliverables & File Inventory

| Deliverable Artifact | File Path | Description & Role |
| :--- | :--- | :--- |
| **Interactive Notebook** | `notebooks/03b_feature_engineering.ipynb` | Executable 19-cell notebook demonstrating end-to-end feature engineering, visualizations, pipeline fitting, and validation |
| **Reusable Feature Module** | `src/features.py` | Standalone Python module containing engineering functions, Empirical Bayes estimators, ColumnTransformer builders, and automated audit assertions |
| **Prepared Feature Dataset** | `data/processed/flights_features.csv` | 24,522 rows × 46 columns (9.5 MB), containing all engineered features, target labels, identifiers, and partition indicator `SPLIT_SET` |
| **Visual Overview** | `results/figures/features/feature_engineering_overview.png` | 4-panel publication visualization (cyclical departure clock, diurnal delay rates, speed buffer boxplots, carrier priors) |
| **Feature Engineering Report** | `reports/member2B_feature_engineering_report.md` | This formal document detailing methodology, formulations, and audit findings |
| **Member 3 Handoff Report** | `reports/member2B_handoff.md` | Direct guidance and ingestion instructions for Member 3 |

---

## 9. Conclusion and Next Steps
Task 2B is 100% complete and verified. The prepared feature dataset is fully decoupled from ground truth and post-departure variables, chronologically partitioned, and ready for baseline linear regression (OLS) and logistic regression modeling by **Member 3**.
