# Member 1B to Member 2 Handoff Report
**Project 21:** Flight Delay Prediction and Rescheduling (Phase 1: Machine Learning)  
**Author:** Member 1B (Data Cleaning & Preprocessing)  
**Recipient:** Member 2 (Exploratory Data Analysis, Feature Engineering & Time Splitting)  
**Date:** October 2026  
**Status:** Task 1B Complete — Cleaned Dataset Ready for Task 2  

---

## 1. Cleaned Dataset Location & Dimensions

The cleaned and sanitized dataset has been generated and validated at:
- **Cleaned Dataset File Path:** `data/processed/flights_cleaned.csv`
- **Total Operative Records:** **5,714,008 rows** (full 2015 cohort) / **24,522 rows** (local benchmark sample)
- **Total Columns:** **16 columns**
- **File Format:** Standard comma-separated values (CSV) with header

> **Protection Note:** The raw archive files in `data/raw/archive/` remain strictly unmodified. All cleaning, filtering, and target constructions were written exclusively to `data/processed/`.

---

## 2. Target Name and Class Distribution

### Binary Classification Target
- **Column Name:** **`DELAYED`**
- **Mathematical Formulation:**
  $$\text{DELAYED} = \begin{cases} 1 & \text{if } \text{ARRIVAL\_DELAY} > 15.0 \text{ minutes} \\ 0 & \text{if } \text{ARRIVAL\_DELAY} \le 15.0 \text{ minutes} \end{cases}$$
- **Class Proportions (5,714,008 Operative Flights):**
  - **Class 0 (On-Time / Minor Delay $\le 15$ min):** **4,690,510 flights (82.09%)**
  - **Class 1 (Delayed $> 15$ min):** **1,023,498 flights (17.91%)**
  - **Class Imbalance:** Approximately **4.58 : 1** (On-Time to Delayed).

### Continuous Target Preservation (For OLS Regression)
- **Column Name:** **`ARRIVAL_DELAY`** (`float64`)
- Retained to enable linear regression benchmarking (OLS) and residual error auditing.
- ⚠️ **CRITICAL WARNING:** `ARRIVAL_DELAY` is the ground-truth outcome. It must **never** be passed into feature matrix $X$ during classification or regression.

---

## 3. Date Column and Calendar Coverage

- **Date Column Name:** **`FLIGHT_DATE`**
- **Format:** String formatted as `YYYY-MM-DD` (e.g., `2015-01-01` to `2015-12-31`).
- **Date Range:** January 1, 2015 through December 31, 2015 (All 365 days of **2015** covered).
- **Calendar Sub-Components Preserved:** `YEAR`, `MONTH`, `DAY`, `DAY_OF_WEEK` (1 = Monday, 7 = Sunday).
- **Usability:** Directly sortable and parseable via `pd.to_datetime(df['FLIGHT_DATE'])` for chronological train/validation/test splitting.

---

## 4. Preserved Relevant Feature Columns

The cleaned dataset contains **16 columns**:

```
Pre-Departure Features (Eligible for Modeling):
 ├── Calendar / Temporal: FLIGHT_DATE, YEAR, MONTH, DAY, DAY_OF_WEEK
 ├── Airline / Flight:    AIRLINE, FLIGHT_NUMBER, TAIL_NUMBER
 ├── Spatial / Route:     ORIGIN_AIRPORT, DESTINATION_AIRPORT, DISTANCE
 └── Planned Schedule:    SCHEDULED_DEPARTURE, SCHEDULED_TIME, SCHEDULED_ARRIVAL

Supervised Targets (Ground Truth Only):
 ├── Continuous Target:   ARRIVAL_DELAY (Minutes late/early)
 └── Binary Target:       DELAYED (1 if ARRIVAL_DELAY > 15, else 0)
```

---

## 5. Cleaning Decisions and Exclusions Summary

| Step | Operation | Rows Excluded | Operational Rationale |
| :--- | :--- | :--- | :--- |
| **1. Duplicates** | Audited duplicate rows | 0 rows | Confirmed zero exact duplicate records |
| **2. Cancellations**| Filtered `CANCELLED == 1` | 89,884 rows | Cancelled flights never departed; arrival delay is undefined ($NaN$) |
| **3. Diversions** | Filtered `DIVERTED == 1` | 15,187 rows | Diverted flights landed at alternate airports; destination dynamics voided |
| **4. Missing Delays**| Filtered `ARRIVAL_DELAY.isna()` | 0 additional | Non-arrivals account for 100.0% of missing arrival delay records |
| **Total** | **Cleaned Retention** | **5,714,008 rows** | **98.19% sample retention rate** |

---

## 6. Strict Data Leakage Warnings for Member 2

The following variables represent events occurring **during or after flight departure** and have been completely excluded from the feature set:

1. ❌ `DEPARTURE_TIME` (Actual gate pushback timestamp)
2. ❌ `DEPARTURE_DELAY` (Actual departure delay)
3. ❌ `TAXI_OUT` (Taxi duration to runway)
4. ❌ `WHEELS_OFF` (Takeoff timestamp)
5. ❌ `ELAPSED_TIME` (Actual gate-to-gate flight duration)
6. ❌ `AIR_TIME` (Actual airborne duration)
7. ❌ `WHEELS_ON` (Touchdown timestamp)
8. ❌ `TAXI_IN` (Taxi duration to arrival gate)
9. ❌ `ARRIVAL_TIME` (Actual gate arrival timestamp)
10. ❌ `ARRIVAL_DELAY` (Target label source — never an input feature)
11. ❌ `CANCELLATION_REASON`, `AIR_SYSTEM_DELAY`, `SECURITY_DELAY`, `AIRLINE_DELAY`, `LATE_AIRCRAFT_DELAY`, `WEATHER_DELAY` (Post-hoc breakdowns)

> **Mandatory Rule for Member 2:** When constructing feature matrix $X$, drop both `DELAYED` and `ARRIVAL_DELAY`. Using either in $X$ causes fatal data leakage.

---

## 7. Instructions for Member 2 (How to Ingest and Proceed)

To begin Task 2 (EDA, Feature Engineering, and Chronological Splitting):

### Loading the Cleaned Dataset in Python:
```python
import pandas as pd
from pathlib import Path

# Load sanitized dataset prepared by Member 1B
CLEANED_PATH = Path("data/processed/flights_cleaned.csv")
df = pd.read_csv(CLEANED_PATH, low_memory=False)

# Convert FLIGHT_DATE to datetime for chronological sorting
df["FLIGHT_DATE"] = pd.to_datetime(df["FLIGHT_DATE"])
df = df.sort_values(by="FLIGHT_DATE").reset_index(drop=True)

print(f"Loaded {len(df):,} cleaned operative flights.")
print(f"Target 'DELAYED' distribution:\n{df['DELAYED'].value_counts(normalize=True) * 100}")
```

### Directives for Member 2:
1. **Chronological Splitting (No Random K-Fold):**
   - Partition data strictly by date to prevent lookahead leakage:
     - **Train:** January 1 – August 31, 2015 (~67% of data)
     - **Validation:** September 1 – October 31, 2015 (~16% of data)
     - **Test:** November 1 – December 31, 2015 (~17% of data)
2. **Cyclic Time Encodings:**
   - Transform `SCHEDULED_DEPARTURE` (in `HHMM` format) into continuous hour ($0 \le h < 24$) and minute, then compute sine and cosine transformations:
     $$\text{hour\_sin} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$
3. **Training-Only Historical Delay Priors:**
   - If generating target-encoded priors (carrier historical delay rate, route historical delay rate), fit them **exclusively on the training split** and map them to validation/test sets to avoid lookahead bias.
4. **No Model Training in Task 2:**
   - Focus strictly on EDA, visual inspection, feature pipelines, and split validation. Model training begins in Task 3.
