# Member 1A: Comprehensive Data Quality and Dataset Inspection Report
**Project 21:** Flight Delay Prediction and Rescheduling  
**Author:** Sarvjeet Yadav (Member 1A: Data Inspection & Quality Analysis) <sarvjeetyadav2969@gmail.com>  
**Phase:** AML Phase 1 — Machine Learning (Academic Syllabus)  
**Date:** October 20, 2026  
**Status:** Completed and Verified  

---

## Executive Summary
This report provides a formal, comprehensive inspection and data quality audit of the raw flight delay dataset in accordance with the Phase 1 specifications for Task 1A. All statistics, counts, percentages, and schema findings documented in this report are derived directly from the physical dataset files located in `data/raw/archive/` without data fabrication or artificial imputation.

### Key Headline Findings:
1. **Dataset Authenticity & Verified Year:** The raw dataset spans the full calendar year of **2015** (January 1, 2015 through December 31, 2015). **The dataset is NOT from 2013**, despite any initial assumptions in external project descriptions.
2. **Dataset Scale:** The primary dataset `flights.csv` contains exactly **5,819,079 rows** and **31 columns** (~564.96 MB on disk).
3. **Missing Value Reconciliation:** A total of **105,071 rows (1.81%)** have missing `ARRIVAL_DELAY` values. This missingness is **100.0% accounted for** by cancelled flights (**89,884 flights**, 1.54%) and diverted flights (**15,187 flights**, 0.26%), as neither flight category ever arrived at its scheduled destination.
4. **Target Distribution:** Defining delayed flights as `ARRIVAL_DELAY > 15 minutes` yields **1,023,498 delayed flights (17.91%)** among operative flights. Under the standard US DOT BTS threshold `ARRIVAL_DELAY >= 15 minutes`, there are **1,063,439 delayed flights (18.61%)**, establishing an operational class imbalance of approximately **81.4% to 82.1% on-time vs 17.9% to 18.6% delayed**.
5. **Strict Anti-Leakage Audit:** A total of **17 columns** represent information that becomes available only during or after departure (including actual departure time, departure delay, wheels-off time, airborne time, and post-hoc delay cause breakdowns). These are strictly quarantined from model input features.
6. **Data Quality Anomalies Identified:**
   - **Airport Code Representation Shift:** From January through September 2015, airports are identified by standard 3-letter IATA codes (e.g., `ORD`, `ATL`). In October, November, and December 2015, BTS switched to 5-digit numeric DOT codes (e.g., `10397`, `12892`), affecting 486,165 flights (8.35% of dataset).
   - **Negative Delays:** Valid early arrivals range down to `-87.0 minutes` (flights landing ahead of schedule).
   - **Extreme Outliers:** Maximum recorded arrival delay reaches `1,971.0 minutes` (~32.8 hours), caused by severe cascading operational delays.

---

## 1. Machine Learning Project Lifecycle

The development of a production-grade machine learning system for flight delay prediction follows a structured eight-stage lifecycle:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        MACHINE LEARNING PROJECT LIFECYCLE (8 STAGES)                   │
├─────────────────┬──────────────────────────────────────────────────────────────────────┤
│ 1. Problem      │ Formulate the operational problem as a supervised binary             │
│    Definition   │ classification task: Predict whether a flight will arrive late       │
│                 │ (> 15 min delay) using strictly pre-departure information.           │
├─────────────────┼──────────────────────────────────────────────────────────────────────┤
│ 2. Data         │ Ingest raw flight performance archives (Kaggle / US DOT BTS).        │
│    Collection   │ Verify file integrity, storage size, schema structure, and sources.  │
├─────────────────┼──────────────────────────────────────────────────────────────────────┤
│ 3. Data         │ [Task 1A - Member 1A] Audit dimensions, column datatypes,            │
│    Inspection   │ missingness, duplicates, anomalies, calendar coverage, and true year.│
├─────────────────┼──────────────────────────────────────────────────────────────────────┤
│ 4. Data         │ [Task 1B - Member 1B] Filter non-applicable operational records      │
│    Cleaning     │ (cancelled/diverted), reconcile dual airport coding, format types.   │
├─────────────────┼──────────────────────────────────────────────────────────────────────┤
│ 5. Feature      │ Construct chronological train/val/test splits, extract cyclic time   │
│    Engineering  │ encodings (sin/cos), compute historical delay priors, one-hot encode.│
├─────────────────┼──────────────────────────────────────────────────────────────────────┤
│ 6. Model        │ Fit classical models (Logistic Regression via MLE, Batch GD, SGD,    │
│    Training     │ Mini-Batch GD) with L1/L2 penalties per academic syllabus.           │
├─────────────────┼──────────────────────────────────────────────────────────────────────┤
│ 7. Model        │ Benchmark using class-imbalance metrics: Confusion Matrix, Precision,│
│    Evaluation   │ Recall, F1-Score, ROC-AUC, PR-AUC, and inspect Bias-Variance tradeoff.│
├─────────────────┼──────────────────────────────────────────────────────────────────────┤
│ 8. Deployment   │ Analyze operational decision thresholds, latency constraints, model  │
│    & Limits     │ drift over time, and realistic limits of pre-departure forecasting.  │
└─────────────────┴──────────────────────────────────────────────────────────────────────┘
```

### Why Predictions Must Use Only Information Available Before Departure
In real-world airline operations and air traffic control:
1. **Decision Timeframe:** Proactive intervention measures—such as gate reassignment, flight holding, crew rescheduling, fuel adjustments, and passenger re-accommodation—must be executed **prior to gate pushback**.
2. **Data Leakage Hazard:** Any feature recorded during or after departure (e.g., actual departure time, departure delay, taxi-out duration, wheels-off timestamp, airborne time, or post-hoc delay classification codes) constitutes **fatal target leakage**.
3. **Operational Failure:** Incorporating post-departure features yields deceptive offline validation performance (>95% accuracy) but guarantees complete failure in production, because those variables simply do not exist at prediction time.

---

## 2. Prediction Problem & Strict Anti-Leakage Guardrails

### 2.1 Proposed Target Formulation
The binary classification target for Phase 1 is defined as:

$$\text{DELAYED} = \begin{cases} 1 & \text{if } \text{ARRIVAL\_DELAY} > 15 \text{ minutes} \\ 0 & \text{otherwise} \end{cases}$$

*(Under standard US DOT BTS definitions, arrival delay $\ge 15$ minutes is also analyzed).*

> **Crucial Rule:** `ARRIVAL_DELAY` is used **exclusively to construct the ground-truth target label**. It is never passed into the model input feature matrix $X$.

### 2.2 Feature Leakage Audit Matrix

| Column Category | Specific Dataset Fields | Operational Timing | Modeling Decision |
| :--- | :--- | :--- | :--- |
| **Permitted Input Features (Pre-Departure)** | `YEAR`, `MONTH`, `DAY`, `DAY_OF_WEEK`, `AIRLINE`, `FLIGHT_NUMBER`, `ORIGIN_AIRPORT`, `DESTINATION_AIRPORT`, `SCHEDULED_DEPARTURE`, `SCHEDULED_ARRIVAL`, `SCHEDULED_TIME`, `DISTANCE` | Known prior to pushback | ✅ **ELIGIBLE FEATURES** (Safe for modeling) |
| **Departure-Phase Leakage** | `DEPARTURE_TIME`, `DEPARTURE_DELAY`, `TAXI_OUT`, `WHEELS_OFF` | Generated during gate pushback & takeoff | ❌ **STRICT LEAKAGE** (Quarantine/Drop) |
| **In-Flight & Arrival Leakage** | `ELAPSED_TIME`, `AIR_TIME`, `WHEELS_ON`, `TAXI_IN`, `ARRIVAL_TIME` | Generated in airborne flight & landing | ❌ **STRICT LEAKAGE** (Quarantine/Drop) |
| **Post-Hoc Cause Breakdowns** | `AIR_SYSTEM_DELAY`, `SECURITY_DELAY`, `AIRLINE_DELAY`, `LATE_AIRCRAFT_DELAY`, `WEATHER_DELAY`, `CANCELLATION_REASON` | Calculated hours/days later by carrier/BTS | ❌ **STRICT LEAKAGE** (Quarantine/Drop) |
| **Operational Non-Arrivals** | `CANCELLED`, `DIVERTED` | Flight aborted or routed to alternate city | ⚠️ **FILTERING CONDITION** (Exclude from delay models) |

---

## 3. Project Folder and Raw Dataset File Inventory

The dataset files are stored in `data/raw/archive/`:

| File Name | Format | Disk Size (Bytes) | Disk Size (MB) | Total Records | Role in Project |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`flights.csv`** | CSV | 592,406,591 | 564.96 MB | 5,819,079 rows | Primary flight operations dataset |
| **`airlines.csv`** | CSV | 359 | 0.0003 MB | 14 rows | Airline IATA code to carrier name lookup |
| **`airports.csv`** | CSV | 23,867 | 0.0228 MB | 322 rows | Airport IATA code, name, city, coordinates |

### Relationship Between Files:
- `flights.csv.AIRLINE` connects via foreign key to `airlines.csv.IATA_CODE`.
- `flights.csv.ORIGIN_AIRPORT` and `flights.csv.DESTINATION_AIRPORT` connect via foreign key to `airports.csv.IATA_CODE` for January through September flights (and via 5-digit DOT mapping for October through December flights).

---

## 4. Dataset Dimensions, Schema, and Year Verification

### 4.1 Year Verification
- **Verified Years Present:** `[2015]`
- **Verified Months Present:** `1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12` (All 12 calendar months)
- **Verified Date Range:** `2015-01-01` to `2015-12-31` (365 unique calendar days)
- **Year Finding:** The dataset is **strictly from the year 2015**, **not 2013**. Any downstream code, documentation, or modeling assumptions must use 2015.

### 4.2 Complete Dataset Schema (31 Columns)

| Column # | Column Name | Data Type | Null Count | Null % | Pre-Departure Feasibility |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `YEAR` | `int64` | 0 | 0.00% | Eligible |
| 2 | `MONTH` | `int64` | 0 | 0.00% | Eligible |
| 3 | `DAY` | `int64` | 0 | 0.00% | Eligible |
| 4 | `DAY_OF_WEEK` | `int64` | 0 | 0.00% | Eligible |
| 5 | `AIRLINE` | `object` | 0 | 0.00% | Eligible |
| 6 | `FLIGHT_NUMBER` | `int64` | 0 | 0.00% | Eligible |
| 7 | `TAIL_NUMBER` | `object` | 14,721 | 0.25% | Pre-departure aircraft ID |
| 8 | `ORIGIN_AIRPORT` | `object` | 0 | 0.00% | Eligible (Requires dual coding cleaning) |
| 9 | `DESTINATION_AIRPORT` | `object` | 0 | 0.00% | Eligible (Requires dual coding cleaning) |
| 10 | `SCHEDULED_DEPARTURE` | `int64` | 0 | 0.00% | Eligible (HHMM format) |
| 11 | `DEPARTURE_TIME` | `float64` | 86,153 | 1.48% | ⚠️ DATA LEAKAGE |
| 12 | `DEPARTURE_DELAY` | `float64` | 86,153 | 1.48% | ⚠️ DATA LEAKAGE |
| 13 | `TAXI_OUT` | `float64` | 89,047 | 1.53% | ⚠️ DATA LEAKAGE |
| 14 | `WHEELS_OFF` | `float64` | 89,047 | 1.53% | ⚠️ DATA LEAKAGE |
| 15 | `SCHEDULED_TIME` | `float64` | 6 | 0.00% | Eligible |
| 16 | `ELAPSED_TIME` | `float64` | 105,071 | 1.81% | ⚠️ DATA LEAKAGE |
| 17 | `AIR_TIME` | `float64` | 105,071 | 1.81% | ⚠️ DATA LEAKAGE |
| 18 | `DISTANCE` | `int64` | 0 | 0.00% | Eligible |
| 19 | `WHEELS_ON` | `float64` | 92,513 | 1.59% | ⚠️ DATA LEAKAGE |
| 20 | `TAXI_IN` | `float64` | 92,513 | 1.59% | ⚠️ DATA LEAKAGE |
| 21 | `SCHEDULED_ARRIVAL` | `int64` | 0 | 0.00% | Eligible (HHMM format) |
| 22 | `ARRIVAL_TIME` | `float64` | 92,513 | 1.59% | ⚠️ DATA LEAKAGE |
| 23 | `ARRIVAL_DELAY` | `float64` | 105,071 | 1.81% | 🎯 TARGET GENERATION ONLY |
| 24 | `DIVERTED` | `int64` | 0 | 0.00% | Operational Filter |
| 25 | `CANCELLED` | `int64` | 0 | 0.00% | Operational Filter |
| 26 | `CANCELLATION_REASON` | `object` | 5,729,195 | 98.46% | Post-hoc event code |
| 27 | `AIR_SYSTEM_DELAY` | `float64` | 4,755,640 | 81.72% | ⚠️ DATA LEAKAGE |
| 28 | `SECURITY_DELAY` | `float64` | 4,755,640 | 81.72% | ⚠️ DATA LEAKAGE |
| 29 | `AIRLINE_DELAY` | `float64` | 4,755,640 | 81.72% | ⚠️ DATA LEAKAGE |
| 30 | `LATE_AIRCRAFT_DELAY` | `float64` | 4,755,640 | 81.72% | ⚠️ DATA LEAKAGE |
| 31 | `WEATHER_DELAY` | `float64` | 4,755,640 | 81.72% | ⚠️ DATA LEAKAGE |

---

## 5. Detailed Data Quality Audit

### 5.1 Duplicate Rows
- A complete hash-set audit across all **5,819,079 rows** revealed **0 exact duplicate rows**. Every row represents a distinct scheduled flight entry.

### 5.2 Missing Value Patterns
- **Delay Breakdown Attributes:** `AIR_SYSTEM_DELAY`, `SECURITY_DELAY`, `AIRLINE_DELAY`, `LATE_AIRCRAFT_DELAY`, `WEATHER_DELAY` are each missing in **4,755,640 rows (81.72%)**. This is because the FAA and BTS only record cause attribution when a flight is delayed by 15 minutes or more.
- **Cancellation Reasons:** `CANCELLATION_REASON` is missing in **5,729,195 rows (98.46%)**, matching operative non-cancelled flights.
- **Flight & Airport Identifiers:** Core pre-departure features (`YEAR`, `MONTH`, `DAY`, `AIRLINE`, `ORIGIN_AIRPORT`, `DESTINATION_AIRPORT`, `SCHEDULED_DEPARTURE`, `SCHEDULED_ARRIVAL`, `DISTANCE`) have **0 missing values (0.00%)**.

### 5.3 Data Quality Anomalies and Unusual Values
1. **Airport Code Schema Anomaly:**
   - **January–September 2015:** 3-letter IATA codes (5,332,914 flights, 91.65%).
   - **October–December 2015:** 5-digit DOT codes (486,165 flights, 8.35%).
   - This creates 628 distinct strings in `ORIGIN_AIRPORT` and 629 in `DESTINATION_AIRPORT`, whereas `airports.csv` lists only 322 physical airports.
2. **Early Arrivals (Negative Delays):**
   - Valid flight records have arrival delays down to `-87.0 minutes`. Early arrivals are normal operational occurrences due to favorable tailwinds and padded schedule buffers.
3. **Severe Delay Outliers:**
   - 99% of flights have arrival delays $\le 170$ minutes.
   - The maximum delay is `1,971.0 minutes` (~32.8 hours). These points represent true operational black-swan events (snowstorms, ground stops) rather than measurement errors.

---

## 6. Operational Outcomes: Cancellations and Diversions

| Operational Outcome | Row Count | Percentage of Total Flights |
| :--- | :--- | :--- |
| **Cancelled Flights (`CANCELLED == 1`)** | 89,884 | 1.54% |
| **Diverted Flights (`DIVERTED == 1`)** | 15,187 | 0.26% |
| **Combined Non-Arrivals** | **105,071** | **1.81%** |
| **Missing Arrival Delays (`ARRIVAL_DELAY.isna()`)** | **105,071** | **1.81%** |

**Crucial Audit Conclusion:**
There is a **zero discrepancy** between missing arrival delays and non-arriving flights ($89,884 + 15,187 = 105,071$). All missing arrival delays are caused by flights that never landed at their scheduled destination.

---

## 7. Target Variable Distribution

Restricting the dataset to operative flights that completed their journey (**5,714,008 flights**):

| Target Formulation | Delayed Flights | On-Time Flights | Delayed % | On-Time % |
| :--- | :--- | :--- | :--- | :--- |
| **`ARRIVAL_DELAY > 15 min` (Phase 1 Proposed)** | **1,023,498** | 4,690,510 | **17.91%** | 82.09% |
| **`ARRIVAL_DELAY >= 15 min` (US DOT BTS Standard)** | **1,063,439** | 4,650,569 | **18.61%** | 81.39% |

### Delay Summary Statistics (Minutes):
- **Mean Arrival Delay:** `+4.41 minutes`
- **Median Arrival Delay:** `-4.00 minutes` (more than half of flights arrive early)
- **Standard Deviation:** `39.27 minutes`
- **Minimum:** `-87.0 minutes`
- **25th Percentile:** `-13.0 minutes`
- **50th Percentile (Median):** `-4.0 minutes`
- **75th Percentile:** `+10.0 minutes`
- **90th Percentile:** `+39.0 minutes`
- **95th Percentile:** `+72.0 minutes`
- **99th Percentile:** `+170.0 minutes`
- **Maximum:** `+1,971.0 minutes`

---

## 8. Categorical Dimensions

### 8.1 Airlines (14 Operating Carriers)
All 14 airlines present in `flights.csv` are cross-referenced with `airlines.csv`:

| IATA Code | Carrier Full Name | Volume (2015) | Delay Rate (>15m) |
| :--- | :--- | :--- | :--- |
| **WN** | Southwest Airlines Co. | 1,261,855 | ~18.5% |
| **DL** | Delta Air Lines Inc. | 875,881 | ~14.1% (Top Reliability) |
| **AA** | American Airlines Inc. | 725,984 | ~18.6% |
| **OO** | Skywest Airlines Inc. | 588,353 | ~18.3% |
| **EV** | Atlantic Southeast Airlines | 571,977 | ~19.5% |
| **UA** | United Air Lines Inc. | 515,723 | ~18.9% |
| **MQ** | American Eagle Airlines Inc. | 294,632 | ~20.3% |
| **B6** | JetBlue Airways | 267,048 | ~22.1% |
| **US** | US Airways Inc. | 198,715 | ~17.2% |
| **AS** | Alaska Airlines Inc. | 172,521 | ~12.3% (Lowest Delay Rate) |
| **NK** | Spirit Air Lines | 117,379 | ~28.6% (Highest Delay Rate) |
| **F9** | Frontier Airlines Inc. | 90,836 | ~27.8% |
| **HA** | Hawaiian Airlines Inc. | 76,272 | ~12.5% |
| **VX** | Virgin America | 61,903 | ~16.8% |

---

## 9. Generated Figures and Visualizations

The following figures have been generated and saved to `results/figures/`:
1. **Figure 1 (`fig1_missing_values.png`):** Percentage of missing data per column across all 5,819,079 flights.
2. **Figure 2 (`fig2_arrival_delay_distribution.png`):** Arrival delay distribution with KDE, boxplot outlier bounds, and the 15-minute threshold.
3. **Figure 3 (`fig3_flights_over_time.png`):** Daily flight volumes and delay rate fluctuations across all 365 calendar days of 2015.
4. **Figure 4 (`fig4_airline_delays.png`):** Carrier flight volume and comparative delay probability across all 14 airlines.
5. **Figure 5 (`fig5_hourly_delays.png`):** Delay probability by scheduled departure hour, showing steep escalation from morning (~8%) to late evening (>25%).

---

## 10. Conclusion and Next Steps
Task 1A data inspection is 100% complete. No dataset cleaning, transformation, or modeling was executed in accordance with Task 1A scope. The project is ready for handoff to Member 1B for data cleaning and feature pipeline construction.
