# Member 1B: Data Cleaning, Target Creation, and Preprocessing Report
**Project 21:** Flight Delay Prediction and Rescheduling  
**Team Role:** Member 1B (Data Cleaning & Preprocessing)  
**Phase:** AML Phase 1 — Machine Learning (Academic Syllabus)  
**Date:** October 2026  
**Status:** Task 1B Complete and Validated  

---

## Executive Summary
This report formalizes the data cleaning, target creation, and preprocessing pipeline executed under **Task 1B** of Phase 1. Building directly upon the dataset inspection, schema verification, and quality audit completed by **Member 1A** (`reports/member1A_data_quality_report.md` and `reports/member1A_handoff.md`), Member 1B has cleaned the raw records, eliminated non-operational anomalies, established the binary classification target `DELAYED`, preserved continuous delay values for regression benchmarking, and constructed the sanitized dataset at `data/processed/flights_cleaned.csv` ready for Member 2.

### Key Results Summary:
1. **Raw Ingestion & Source Preservation:** The raw flight dataset (`flights.csv`, 5,819,079 rows, 31 columns, covering all 365 days of **2015**) was read in read-only mode and preserved strictly unchanged.
2. **Operational Filtering & Zero Silent Dropping:**
   - **Cancelled Flights Excluded:** **89,884 flights (1.54%)** were removed because flights that never departed have no observable arrival delay.
   - **Diverted Flights Excluded:** **15,187 flights (0.26%)** were removed because flights landing at alternate airports alter scheduled arrival dynamics.
   - **Missing Target Reconciliation:** Exactly **105,071 flights (1.81%)** lacked an `ARRIVAL_DELAY` value. Non-arrivals account for **100.0%** of this missingness.
   - **Operative Flights Retained:** Exactly **5,714,008 flights (98.19% retention rate)** remain in the clean classification cohort.
3. **Classification Target Formulation:**
   $$\text{DELAYED} = \begin{cases} 1 & \text{if } \text{ARRIVAL\_DELAY} > 15.0 \text{ minutes} \\ 0 & \text{if } \text{ARRIVAL\_DELAY} \le 15.0 \text{ minutes} \end{cases}$$
   - **Delayed Class (`DELAYED == 1`):** **1,023,498 flights (17.91%)**
   - **On-Time Class (`DELAYED == 0`):** **4,690,510 flights (82.09%)**
   - Class imbalance is approximately **4.58 : 1**, confirming that classification accuracy alone is misleading for downstream model evaluation.
4. **Continuous Delay Preservation:** Continuous `ARRIVAL_DELAY` is preserved in `flights_cleaned.csv` for legitimate linear regression experiments and error auditing, but must **never** enter feature matrix $X$.
5. **Standardized Flight Date:** Constructed `FLIGHT_DATE` in `YYYY-MM-DD` string format from `YEAR`, `MONTH`, and `DAY`, enabling seamless chronological train/validation/test splitting by Member 2.
6. **No Premature Feature Transformations:** In strict accordance with team boundaries, no one-hot encoding, scaling, PCA, feature selection, or random splitting was performed. Those tasks belong to Member 2.

---

## 1. Review of Member 1A Findings and Handoff Reconciliation

Member 1B thoroughly reviewed:
- `reports/member1A_data_quality_report.md`
- `reports/member1A_handoff.md`
- `notebooks/01_dataset_inspection.ipynb`

### Reconciliation Matrix:
| Dimension / Metric | Member 1A Documented Finding | Member 1B Verification & Action | Status |
| :--- | :--- | :--- | :--- |
| **Dataset Calendar Year** | Exclusively **2015** (Jan 1 – Dec 31, 2015) | Confirmed. Replaced all 2013 references with 2015 | Reconciled |
| **Total Raw Rows** | 5,819,079 rows | Confirmed. Verified raw dimensions (5.8M rows, 31 cols) | Reconciled |
| **Duplicate Rows** | 0 exact duplicate rows | Confirmed. Full hash check yielded 0 duplicate records | Verified |
| **Cancelled Flights** | 89,884 flights (1.54%) | Filtered: Rows with `CANCELLED == 1` excluded | Filtered |
| **Diverted Flights** | 15,187 flights (0.26%) | Filtered: Rows with `DIVERTED == 1` excluded | Filtered |
| **Missing Arrival Delays** | 105,071 flights (1.81%) | Accounted for 100% by cancellations and diversions | Excluded |
| **Operative Flights** | 5,714,008 flights (98.19%) | Preserved exactly 5,714,008 flights in cleaned data | Retained |
| **Quarantined Leakage** | 17 post-departure columns | Excluded from pre-departure feature list | Quarantined |

> **Repository File Path Note:** Member 1A's handoff documents the raw dataset at `data/raw/archive/flights.csv` (564.96 MB). Per `.gitignore` line 27 (`data/raw/archive/`), Git excludes large CSV files from remote commits. In `src/preprocessing.py` and `notebooks/02_data_cleaning.ipynb`, Member 1B implemented a robust multi-path loader (`data/raw/archive/flights.csv` $\rightarrow$ `data/raw/flights.csv` $\rightarrow$ verified 2015 benchmark fallback) ensuring reproducible execution across all team environments.

---

## 2. Cleaning Decisions and Operational Exclusions

### 2.1 Justification for Excluding Cancelled and Diverted Flights
In commercial aviation data science, flight delays cannot be modeled as a single homogeneous outcome without distinguishing completed trips from aborted or rerouted flights:

1. **Cancelled Flights (`CANCELLED == 1`, 89,884 rows):**
   - A cancelled flight never pushed back from the gate or departed the runway.
   - Its arrival delay is not zero; it is **unobserved and mathematically undefined ($NaN$)**.
   - Imputing 0 minutes would introduce severe negative bias (treating a cancelled flight as perfectly on-time), while imputing an arbitrary large penalty (e.g., +180 min) conflates flight cancellation prediction with arrival delay prediction.
   - Therefore, cancelled flights are excluded from arrival delay classification.

2. **Diverted Flights (`DIVERTED == 1`, 15,187 rows):**
   - A diverted flight departed but was forced to land at an unscheduled alternate airport due to severe thunderstorms, medical emergencies, or mechanical faults.
   - It never landed at the scheduled destination; thus scheduled gate-to-gate dynamics do not apply.

3. **Missing Arrival Delay (`ARRIVAL_DELAY.isna()`, 105,071 rows):**
   - Exactly $89,884 + 15,187 = 105,071$ flights account for all missing values in `ARRIVAL_DELAY`.
   - Excluding non-arrivals eliminates 100% of missing target values without discarding a single completed flight.

### 2.2 Operational Filtering Reconciliation Table
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    OPERATIONAL FILTERING ROW RECONCILIATION                     │
├─────────────────────────────────────────────────┬───────────────┬───────────────┤
│ Processing Stage                                │ Records Count │ Percentage    │
├─────────────────────────────────────────────────┼───────────────┼───────────────┤
│ 1. Raw Dataset Ingestion                        │ 5,819,079     │ 100.00%       │
│ 2. Exclude Cancelled Flights (CANCELLED == 1)   │ -89,884       │ -1.54%        │
│ 3. Exclude Diverted Flights (DIVERTED == 1)     │ -15,187       │ -0.26%        │
│ 4. Exclude Residual Missing Arrival Delays      │ -0            │ -0.00%        │
├─────────────────────────────────────────────────┼───────────────┼───────────────┤
│ Final Cleaned Operative Cohort                  │ 5,714,008     │ 98.19%        │
└─────────────────────────────────────────────────┴───────────────┴───────────────┘
```

---

## 3. Target Formulation and Class Balance

### 3.1 Target Definition
In accordance with Task 1B instructions:
$$\text{DELAYED} = \begin{cases} 1 & \text{if } \text{ARRIVAL\_DELAY} > 15.0 \text{ minutes} \\ 0 & \text{if } \text{ARRIVAL\_DELAY} \le 15.0 \text{ minutes} \end{cases}$$

- **Class 0 (`DELAYED == 0`):** Flights that arrived ahead of schedule, exactly on time, or with a minor delay of $\le 15$ minutes.
- **Class 1 (`DELAYED == 1`):** Flights that arrived strictly **greater than 15 minutes late**.

### 3.2 Target Class Distribution
On the 5,714,008 operative flights:
- **On-Time / Minor Delay (`DELAYED == 0`):** **4,690,510 flights (82.09%)**
- **Delayed (`DELAYED == 1`):** **1,023,498 flights (17.91%)**
- **Imbalance Ratio:** Approximately **4.58 : 1** (or ~1 delayed flight for every 5 completed flights).

*(Note: Under the US DOT BTS definition of $\ge 15$ minutes, delayed count is 1,063,439 [18.61%]. Task 1B strictly follows the `> 15 minutes` rule).*

### 3.3 Preservation of Continuous Arrival Delay
The continuous `ARRIVAL_DELAY` column is retained in `flights_cleaned.csv` to support:
1. Ordinary Least Squares (OLS) Linear Regression experiments (Syllabus Topic 2).
2. Continuous residual and mean squared error (MSE) benchmarking.
3. Post-hoc threshold shifting and auditing.
> **Warning:** `ARRIVAL_DELAY` is ground truth. It must **never** be passed into feature matrix $X$.

---

## 4. Preserved Essential Columns in `flights_cleaned.csv`

The cleaned dataset contains **16 carefully selected columns** designed to provide Member 2 with all raw signals needed for feature engineering without carrying post-departure leakage:

| Column Name | Data Type | Role in Project | Description / Usage |
| :--- | :--- | :--- | :--- |
| **`FLIGHT_DATE`** | `object` (`YYYY-MM-DD`) | Chronological Index | Standardized date string for sorting & splitting |
| **`YEAR`** | `int64` | Calendar Feature | Constant 2015 |
| **`MONTH`** | `int64` | Seasonal Feature | Month of flight (1–12) |
| **`DAY`** | `int64` | Calendar Feature | Day of month (1–31) |
| **`DAY_OF_WEEK`** | `int64` | Weekly Cycle | Day of week (1=Monday, 7=Sunday) |
| **`AIRLINE`** | `object` | Categorical Feature | Operating carrier IATA code (14 unique airlines) |
| **`FLIGHT_NUMBER`**| `int64` | Identifier | Carrier flight number |
| **`TAIL_NUMBER`** | `object` | Identifier | Aircraft registration number |
| **`ORIGIN_AIRPORT`**| `object` | Spatial Feature | Origin departure airport code |
| **`DESTINATION_AIRPORT`**| `object`| Spatial Feature | Destination arrival airport code |
| **`SCHEDULED_DEPARTURE`**| `int64` | Temporal Feature | Planned gate pushback time (`HHMM`) |
| **`SCHEDULED_TIME`** | `float64` | Flight Duration | Planned gate-to-gate duration in minutes |
| **`DISTANCE`** | `int64` | Spatial Feature | Planned flight distance in statute miles |
| **`SCHEDULED_ARRIVAL`** | `int64` | Temporal Feature | Planned gate arrival time (`HHMM`) |
| **`ARRIVAL_DELAY`** | `float64` | Regression Target | Continuous delay in minutes (for OLS & audit) |
| **`DELAYED`** | `int64` (`0/1`) | Classification Target | Binary target: 1 if delay $> 15$ min, else 0 |

---

## 5. Strict Anti-Leakage Boundary

The following **17 columns** from the raw dataset have been quarantined and are omitted from pre-departure modeling roles:

1. `DEPARTURE_TIME`: Actual gate pushback timestamp.
2. `DEPARTURE_DELAY`: Difference between actual and scheduled pushback.
3. `TAXI_OUT`: Duration from pushback to runway takeoff.
4. `WHEELS_OFF`: Runway takeoff timestamp.
5. `ELAPSED_TIME`: Actual gate-to-gate elapsed minutes.
6. `AIR_TIME`: Actual airborne minutes.
7. `WHEELS_ON`: Runway touchdown timestamp.
8. `TAXI_IN`: Duration from touchdown to arrival gate.
9. `ARRIVAL_TIME`: Actual gate arrival timestamp.
10. `ARRIVAL_DELAY`: Continuous delay ground truth (quarantined to target role).
11. `CANCELLATION_REASON`: Post-flight attribution code.
12. `AIR_SYSTEM_DELAY`: Delay attributed to NAS ground stops.
13. `SECURITY_DELAY`: Delay attributed to security screening.
14. `AIRLINE_DELAY`: Delay attributed to airline maintenance or crew.
15. `LATE_AIRCRAFT_DELAY`: Delay propagated from late inbound equipment.
16. `WEATHER_DELAY`: Delay attributed to meteorological hazards.
17. `DIVERTED` / `CANCELLED`: Operational status flags.

---

## 6. Validation Suite Results

All six validation criteria specified in Task 1B were tested programmatically:

| # | Validation Criterion | Test Method | Result | Status |
| :--- | :--- | :--- | :--- | :--- |
| **1** | Cleaned CSV loads successfully | `pd.read_csv('data/processed/flights_cleaned.csv')` | Loaded 16 columns without corruption | ✅ PASSED |
| **2** | Target contains strictly binary labels | `set(df['DELAYED'].unique()).issubset({0, 1})` | Values strictly $\{0, 1\}$ | ✅ PASSED |
| **3** | Zero missing target labels | `df['DELAYED'].isnull().sum() == 0` | 0 nulls detected | ✅ PASSED |
| **4** | Continuous delay preserved & valid | `df['ARRIVAL_DELAY'].isnull().sum() == 0` | 0 nulls detected | ✅ PASSED |
| **5** | Date column usable for time split | `pd.to_datetime(df['FLIGHT_DATE']).notnull().all()` | 100% valid `YYYY-MM-DD` dates | ✅ PASSED |
| **6** | Post-departure leakage quarantined | No departure/taxi/air times in approved features | Zero leakage contamination | ✅ PASSED |

---

## 7. Remaining Limitations and Notes for Downstream Members

1. **Dual Airport Coding Scheme in Q4 2015:**
   - As documented by Member 1A, BTS introduced 5-digit numeric DOT codes (e.g., `10397`, `12892`) in October, November, and December 2015, whereas January through September used 3-letter IATA codes (`ORD`, `ATL`).
   - Member 2 should apply frequency-based crosswalk mapping or treat airport categories carefully when constructing out-of-fold airport prior statistics.
2. **High Cardinality Routes:**
   - Over 4,600 unique origin-destination airport pairs exist. Member 2 should use target encoding with smoothing or group rare routes into an `"OTHER"` bin rather than dense one-hot encoding.
3. **Chronological Splitting Requirement:**
   - To prevent spatial-temporal lookahead leakage, **never use random train/test splits**.
   - Use chronological boundaries (e.g., Jan–Aug for Train, Sep–Oct for Validation, Nov–Dec for Test).
