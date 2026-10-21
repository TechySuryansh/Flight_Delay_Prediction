# Member 2A to Member 2B Handoff Report
**Project 21:** Flight Delay Prediction and Rescheduling (Phase 1: Machine Learning)  
**Author:** Member 2A (Exploratory Data Analysis & Descriptive Time-Series Analysis)  
**Recipient:** Member 2B (Feature Engineering, Encoding Pipelines & Chronological Splitting)  
**Date:** October 2026  
**Status:** Task 2A Complete — Exploratory & Time-Series Findings Ready for Feature Engineering  

---

## 1. Executive Summary & Deliverables

Task 2A has concluded the Exploratory Data Analysis and Descriptive Time-Series Analysis on the cleaned dataset produced by Member 1B (`data/processed/flights_cleaned.csv`).

### Artifacts and Deliverables Produced:
1. **Interactive Notebook:** `notebooks/03a_eda_time_series.ipynb`  
   - 37 cells covering data loading, schema validation, target distribution, bivariate relationship inspections, time-series decompositions, and 10 publication-quality visualizations. Fully reproducible and executable.
2. **Comprehensive EDA Report:** `reports/member2A_eda_report.md`  
   - Detailed quantitative findings, statistical summaries across all dimensions, and discussion of data nuances.
3. **Visualization Figures:** `results/figures/eda/` (Charts `01_target_distribution.png` through `10_heatmap_dow_month.png`).
4. **Structured Handoff Report (This Document):** `reports/member2A_handoff.md`  
   - Guidance for Member 2B on candidate features, strict leakage barriers, recommended feature transformations, and dataset limitations.

---

## 2. Ingested Dataset Summary & Schema Audit

| Dimension / Attribute | Audit Value | Member 2B Guidance |
| :--- | :--- | :--- |
| **Cleaned Dataset Location** | `data/processed/flights_cleaned.csv` | Single source of truth for downstream modeling. |
| **Row Count** | **24,522 rows** (local benchmark) / **5,714,008 rows** (full Kaggle cohort) | Ensure pipelines handle both sizes gracefully. |
| **Column Count** | **16 columns** | 14 predeparture features + 2 target fields. |
| **Date Range** | **2015-01-01 to 2015-12-31** | All 365 calendar days of **2015** exclusively. |
| **Duplicates** | **0 duplicate rows** | No deduplication required. |
| **Missing Values** | `TAIL_NUMBER`: 65 nulls (0.27%); all other 15 columns: 0 nulls | Simple fill strategy (`"UNKNOWN"`) or omit tail ID. |
| **Binary Target (`DELAYED`)** | 0: 15,046 (61.36%), 1: 9,476 (38.64%) | Local benchmark distribution (full cohort is ~82.09% / ~17.91%). |
| **Continuous Target (`ARRIVAL_DELAY`)** | Mean: 6.77 min, Median: 6.00 min, Std: 32.44 min, Min: −115 min, Max: 140 min | Retained for regression benchmarks only. |

---

## 3. Strict Data Leakage Barriers (Mandatory Model Constraints)

To prevent severe optimistic bias and model invalidation, Member 2B must enforce the following strict separations when building the feature matrix $X$:

### 3.1 Prohibited Ground-Truth Columns
- ❌ **`DELAYED`**: The binary supervised classification target.
- ❌ **`ARRIVAL_DELAY`**: The continuous supervised regression target.
- > **Rule:** Both columns must be explicitly separated into target vector $y$ and dropped from feature matrix $X$.

### 3.2 Prohibited Post-Departure Columns (Never Re-introduce from Raw Files)
The cleaned dataset has already excluded post-departure events. Member 2B must **never** re-ingest or merge these columns from `data/raw/archive/flights.csv`:
- ❌ `DEPARTURE_TIME` (Actual gate pushback timestamp)
- ❌ `DEPARTURE_DELAY` (Actual departure delay minutes)
- ❌ `TAXI_OUT` (Taxi duration to runway)
- ❌ `WHEELS_OFF` (Takeoff timestamp)
- ❌ `ELAPSED_TIME` (Actual gate-to-gate flight duration)
- ❌ `AIR_TIME` (Airborne duration)
- ❌ `WHEELS_ON` (Touchdown timestamp)
- ❌ `TAXI_IN` (Touchdown to arrival gate duration)
- ❌ `ARRIVAL_TIME` (Actual gate arrival timestamp)
- ❌ Delay breakdown causes: `CANCELLATION_REASON`, `AIR_SYSTEM_DELAY`, `SECURITY_DELAY`, `AIRLINE_DELAY`, `LATE_AIRCRAFT_DELAY`, `WEATHER_DELAY`.

---

## 4. Candidate Predeparture Features for Member 2B

The following 14 columns are legitimate predeparture inputs available at scheduling/booking time:

| Feature Column | Raw Type | Recommended Transformations & Engineering |
| :--- | :--- | :--- |
| `FLIGHT_DATE` | `object` (`YYYY-MM-DD`) | Parse to `datetime64`. Primary index for chronological train/val/test splitting. |
| `SCHEDULED_DEPARTURE` | `int64` (`HHMM`) | Extract continuous hour ($0 \le h < 24$) and minute. Compute cyclic trigonometric encodings: $\sin(2\pi h/24)$ and $\cos(2\pi h/24)$. Derive evening peak indicators ($h \ge 17$). |
| `SCHEDULED_ARRIVAL` | `int64` (`HHMM`) | Extract hour and minute. Compute cyclic encodings $\sin(2\pi h/24)$ and $\cos(2\pi h/24)$. Check arrival period (e.g. night/curfew arrival). |
| `SCHEDULED_TIME` | `float64` | Scheduled block duration in minutes. Useful as continuous feature and to compute planned speed: $\text{DISTANCE} / \text{SCHEDULED_TIME}$. |
| `DISTANCE` | `int64` | Distance in miles. Consider log transformation $\log(1 + \text{DISTANCE})$ or distance bins (short: $<500$, medium: $500\text{--}1500$, long: $>1500$). |
| `MONTH` | `int64` (1–12) | Seasonal cyclic encoding $\sin(2\pi m/12), \cos(2\pi m/12)$ or quarterly one-hot indicators. |
| `DAY` | `int64` (1–31) | Day of month. Cycle or end-of-month indicators. |
| `DAY_OF_WEEK` | `int64` (1–7) | Binary `IS_WEEKEND` indicator (1 if day in $\{6, 7\}$, else 0) or cyclic encoding $\sin(2\pi d/7), \cos(2\pi d/7)$. |
| `AIRLINE` | `object` (14 codes) | One-hot encoding (for linear models) or target encoding (historical carrier delay rate, strictly fit on train split). |
| `ORIGIN_AIRPORT` | `object` (IATA) | Target encoding (origin delay rate) or frequency encoding. Ensure handling of unseen categories with train prior default. |
| `DESTINATION_AIRPORT` | `object` (IATA) | Target encoding (destination delay rate) or frequency encoding. |
| `ROUTE` | Derived | Interaction pair `ORIGIN_AIRPORT + '_' + DESTINATION_AIRPORT`. Useful for route-level historical delay priors. |
| `FLIGHT_NUMBER` | `int64` | High cardinality. Recommend dropping or grouping by carrier to avoid overfitting. |
| `TAIL_NUMBER` | `object` | High cardinality (65 missing). If utilized for aircraft rotation tracking, must impute missing as `"UNKNOWN"`; otherwise drop for baseline models. |

---

## 5. Key Empirical Delay Patterns to Exploit

Based on the quantitative findings from Task 2A, Member 2B should prioritize engineering around these empirical realities:

### 5.1 The Diurnal Delay Escalation (Strongest Predictor)
- **Observed Behavior:** Delay probability rises monotonically throughout the operating day:
  - 05:00–06:00: **26.0% – 27.2%** delay rate (avg delay: −3.9 to −3.6 min; net early)
  - 12:00: **35.1%** delay rate (avg delay: +4.2 min)
  - 17:00–18:00: **45.3% – 47.2%** delay rate (avg delay: +12.1 to +14.0 min)
  - 21:00–22:00: **49.3% – 51.5%** delay rate (avg delay: +14.7 to +17.1 min)
- **Engineering Recommendation:** 
  1. Hour extraction: `dep_hour = SCHEDULED_DEPARTURE // 100`.
  2. Cyclic features: $\text{sin\_hour} = \sin(2\pi \cdot \text{dep\_hour}/24)$, $\text{cos\_hour} = \cos(2\pi \cdot \text{dep\_hour}/24)$.
  3. Time-of-day binning: Early Morning (04–08), Mid-day (09–13), Afternoon (14–17), Evening Peak (18–22), Late Night (23–03).
  4. Scheduled buffer feature: Planned speed ($\text{DISTANCE} / \text{SCHEDULED_TIME}$ in miles/min). Slower planned speed indicates built-in buffer time.

### 5.2 Seasonality and Month Dynamics
- **Observed Behavior:** February (40.12%) and July (40.17%) experienced elevated delay rates compared to March (36.86%) and June (36.26%).
- **Engineering Recommendation:** Encode cyclical seasonality and holiday peak indicators (e.g. Memorial Day, July 4th, Thanksgiving, Christmas/New Year periods).

### 5.3 Carrier and Airport Heterogeneity
- **Observed Behavior:** In the benchmark sample, airline rates span 36.2% (HA) to 40.5% (EV). In the full national dataset, regional and low-cost carriers (EV, OO, MQ, NK, F9) exhibit significantly higher delay vulnerability than major mainline carriers (DL, AA, UA, HA).
- **Engineering Recommendation:** Out-of-fold or chronological target encoding for `AIRLINE`, `ORIGIN_AIRPORT`, and `DESTINATION_AIRPORT`.

---

## 6. Chronological Partitioning Rules (No Random Splitting)

Member 2B must enforce chronological splitting based on `FLIGHT_DATE`:
- **Training Set:** `2015-01-01` to `2015-08-31` (~67% of data, Months 1–8)
- **Validation Set:** `2015-09-01` to `2015-10-31` (~16% of data, Months 9–10)
- **Test Set:** `2015-11-01` to `2015-12-31` (~17% of data, Months 11–12)

### Target Encoding Safe Practice:
When computing prior statistics (e.g., mean carrier delay rate or airport delay rate):
$$\text{Prior}(c) = \frac{\sum_{i \in \text{Train}_c} y_i + m \cdot \bar{y}}{\text{Count}(c) + m}$$
where $m$ is a smoothing weight and $\bar{y}$ is the global training mean. **Never compute priors across validation or test partitions.**

---

## 7. Data Quality & Distribution Nuances

1. **Benchmark Sample vs. Full National Cohort:**
   - The active local dataset is a 24,522-row benchmark sample generated with the authentic 31-column schema.
   - The target delay rate in the benchmark is **38.64%** due to synthetic sampling parameters, compared to **17.91%** in the full 5.7M dataset.
   - Member 2B should configure feature scalers and encoding pipelines so that `fit()` and `transform()` function identically on both the sample and full datasets.
2. **Missing Value Profile:**
   - Only `TAIL_NUMBER` contains missing values (65 rows, 0.27%).
   - All other 15 columns have **0 missing values**.
3. **Cardinality Considerations:**
   - In benchmark: 15 origin airports, 15 destination airports, 14 airlines.
   - In full Kaggle cohort: 322 airports, 14 airlines, ~4,600 unique tail numbers.
   - Pipelines must use an `unknown_value='ignore'` or fallback prior imputation strategy for categories appearing in test/validation that were absent in training.

---

## 8. Verification and Checklist for Member 2B

Before training baseline models in Task 3, Member 2B should verify:
- [ ] `FLIGHT_DATE` is converted to datetime and sorted chronologically.
- [ ] Partitions are split strictly by date boundaries (Train: Jan–Aug, Val: Sep–Oct, Test: Nov–Dec).
- [ ] `DELAYED` and `ARRIVAL_DELAY` are separated into target variables and omitted from feature matrix $X$.
- [ ] No post-departure leakage attributes (`DEPARTURE_TIME`, `DEPARTURE_DELAY`, `TAXI_OUT`, etc.) are present.
- [ ] Cyclic encodings for `SCHEDULED_DEPARTURE` (hour and minute) are implemented.
- [ ] Target-encoded priors are fitted exclusively on the training partition.
- [ ] Feature scaling (StandardScaler / RobustScaler) is fit only on the training split.
