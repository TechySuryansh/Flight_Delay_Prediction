# Member 1A to Member 1B Handoff Report
**Project:** Flight Delay Prediction and Rescheduling (Phase 1: Machine Learning)  
**Author:** Member 1A (Data Inspection & Quality Analysis)  
**Recipient:** Member 1B (Data Cleaning & Preprocessing)  
**Date:** October 20, 2026  
**Status:** Task 1A Complete — Ready for Task 1B  

---

## 1. Raw Dataset Location & File Inventory

The raw dataset files are verified and located at:
- **Primary Flight Data:** `data/raw/archive/flights.csv` (592,406,591 bytes / 564.96 MB, 5,819,079 rows, 31 columns)
- **Carrier Lookup Data:** `data/raw/archive/airlines.csv` (359 bytes, 14 carriers)
- **Airport Lookup Data:** `data/raw/archive/airports.csv` (23,867 bytes, 322 airports)

> **Important Protection Rule:** All raw data files in `data/raw/archive/` are read-only. Do not modify, clean, or overwrite the raw files. Write all cleaned or preprocessed data to `data/processed/`.

---

## 2. Verified Year and Date Range

- **Verified Year:** **2015**
- **Verified Calendar Date Range:** `2015-01-01` to `2015-12-31` (All 365 calendar days covered)
- **Verified Months:** All 12 calendar months (`1` to `12`)
- **Important Distinction:** The initial project description suggested 2013, but programmatic verification proves this dataset is exclusively from **2015**. Please ensure all downstream models, time-split logic, and documentation reflect **2015**.

---

## 3. Actual Column Names and Data Types

The raw flight dataset contains the following 31 columns:

1. `YEAR` (`int64`): Calendar year (2015)
2. `MONTH` (`int64`): Calendar month (1–12)
3. `DAY` (`int64`): Day of the month (1–31)
4. `DAY_OF_WEEK` (`int64`): Day of the week (1=Monday, 7=Sunday)
5. `AIRLINE` (`object`): Operating airline IATA code (14 unique airlines)
6. `FLIGHT_NUMBER` (`int64`): Carrier flight number
7. `TAIL_NUMBER` (`object`): Aircraft tail identifier (14,721 missing)
8. `ORIGIN_AIRPORT` (`object`): Origin airport code (IATA or DOT code)
9. `DESTINATION_AIRPORT` (`object`): Destination airport code (IATA or DOT code)
10. `SCHEDULED_DEPARTURE` (`int64`): Scheduled departure time in `HHMM` format
11. `DEPARTURE_TIME` (`float64`): Actual gate departure time in `HHMM` format *(LEAKAGE)*
12. `DEPARTURE_DELAY` (`float64`): Actual departure delay in minutes *(LEAKAGE)*
13. `TAXI_OUT` (`float64`): Gate pushback to wheels-off taxi time in minutes *(LEAKAGE)*
14. `WHEELS_OFF` (`float64`): Wheels-off takeoff time in `HHMM` format *(LEAKAGE)*
15. `SCHEDULED_TIME` (`float64`): Scheduled gate-to-gate elapsed time in minutes
16. `ELAPSED_TIME` (`float64`): Actual gate-to-gate elapsed time in minutes *(LEAKAGE)*
17. `AIR_TIME` (`float64`): Airborne flight duration in minutes *(LEAKAGE)*
18. `DISTANCE` (`int64`): Great-circle flight distance in statute miles
19. `WHEELS_ON` (`float64`): Wheels-on touchdown time in `HHMM` format *(LEAKAGE)*
20. `TAXI_IN` (`float64`): Touchdown to gate arrival taxi time in minutes *(LEAKAGE)*
21. `SCHEDULED_ARRIVAL` (`int64`): Scheduled arrival time in `HHMM` format
22. `ARRIVAL_TIME` (`float64`): Actual gate arrival time in `HHMM` format *(LEAKAGE)*
23. `ARRIVAL_DELAY` (`float64`): Actual arrival delay in minutes *(TARGET ONLY)*
24. `DIVERTED` (`int64`): Diversion indicator (1=Diverted, 0=Not Diverted)
25. `CANCELLED` (`int64`): Cancellation indicator (1=Cancelled, 0=Not Cancelled)
26. `CANCELLATION_REASON` (`object`): Reason code: A=Carrier, B=Weather, C=NAS, D=Security
27. `AIR_SYSTEM_DELAY` (`float64`): Delay due to National Aviation System *(LEAKAGE)*
28. `SECURITY_DELAY` (`float64`): Delay due to security breach *(LEAKAGE)*
29. `AIRLINE_DELAY` (`float64`): Delay due to carrier control *(LEAKAGE)*
30. `LATE_AIRCRAFT_DELAY` (`float64`): Delay propagated from previous flight *(LEAKAGE)*
31. `WEATHER_DELAY` (`float64`): Delay caused by meteorological conditions *(LEAKAGE)*

---

## 4. Missing-Value and Duplicate Summaries

### 4.1 Duplicate Rows
- **Total Duplicate Rows:** **0** (Audited via full 5.8M row tuple hashing).

### 4.2 Missing Value Breakdown

| Column | Missing Count | Missing % | Operational Explanation |
| :--- | :--- | :--- | :--- |
| `CANCELLATION_REASON` | 5,729,195 | 98.46% | Only populated for the 89,884 cancelled flights |
| `AIR_SYSTEM_DELAY` | 4,755,640 | 81.72% | Only populated when arrival delay $\ge 15$ min |
| `SECURITY_DELAY` | 4,755,640 | 81.72% | Only populated when arrival delay $\ge 15$ min |
| `AIRLINE_DELAY` | 4,755,640 | 81.72% | Only populated when arrival delay $\ge 15$ min |
| `LATE_AIRCRAFT_DELAY` | 4,755,640 | 81.72% | Only populated when arrival delay $\ge 15$ min |
| `WEATHER_DELAY` | 4,755,640 | 81.72% | Only populated when arrival delay $\ge 15$ min |
| `ARRIVAL_DELAY` | **105,071** | **1.81%** | Cancelled (89,884) + Diverted (15,187) flights |
| `ELAPSED_TIME` | 105,071 | 1.81% | Flights that did not complete trip |
| `AIR_TIME` | 105,071 | 1.81% | Flights that did not complete trip |
| `ARRIVAL_TIME` | 92,513 | 1.59% | Cancelled flights plus diversions with no gate arrival |
| `WHEELS_ON` | 92,513 | 1.59% | Cancelled flights plus diversions with no gate arrival |
| `TAXI_IN` | 92,513 | 1.59% | Cancelled flights plus diversions with no gate arrival |
| `TAXI_OUT` | 89,047 | 1.53% | Flights cancelled prior to taxiing |
| `WHEELS_OFF` | 89,047 | 1.53% | Flights cancelled prior to taxiing |
| `DEPARTURE_TIME` | 86,153 | 1.48% | Flights cancelled prior to gate departure |
| `DEPARTURE_DELAY` | 86,153 | 1.48% | Flights cancelled prior to gate departure |
| `TAIL_NUMBER` | 14,721 | 0.25% | Aircraft not assigned |
| `SCHEDULED_TIME` | 6 | 0.00% | Negligible missing schedule duration |

All remaining 13 columns have **0 missing values**.

---

## 5. Cancellation and Diversion Information

- **Total Cancelled Flights:** **89,884** (1.54% of dataset)
  - Cancellation Breakdown: Weather (`B`): ~63.7%, Carrier (`A`): ~19.4%, NAS (`C`): ~16.8%, Security (`D`): <0.1%.
- **Total Diverted Flights:** **15,187** (0.26% of dataset)
- **Combined Non-Arrivals:** **105,071** flights
- **Audit Reconciliation:** The 105,071 cancelled and diverted flights account for **exactly 100%** of missing values in `ARRIVAL_DELAY`.

---

## 6. Proposed Target Column

$$\text{DELAYED} = \begin{cases} 1 & \text{if } \text{ARRIVAL\_DELAY} > 15 \text{ minutes} \\ 0 & \text{otherwise} \end{cases}$$

### Class Imbalance (Operative Flights):
- **Delayed Flights (`DELAYED == 1`):** 1,023,498 (17.91%)
- **On-Time Flights (`DELAYED == 0`):** 4,690,510 (82.09%)
- *(Comparison under $\ge 15$ min: 1,063,439 delayed [18.61%])*
- **Modeling Note:** Arrival delay is used **strictly** to compute the target label $y$. It must be omitted from feature matrix $X$.

---

## 7. Potential Data Leakage Columns (Strict Exclusion List)

The following **17 columns** must be strictly excluded from model feature sets:

1. `DEPARTURE_TIME` (Actual gate pushback timestamp)
2. `DEPARTURE_DELAY` (Actual departure delay)
3. `TAXI_OUT` (Taxi duration to runway)
4. `WHEELS_OFF` (Takeoff timestamp)
5. `ELAPSED_TIME` (Actual flight duration)
6. `AIR_TIME` (Airborne duration)
7. `WHEELS_ON` (Touchdown timestamp)
8. `TAXI_IN` (Taxi duration to arrival gate)
9. `ARRIVAL_TIME` (Actual gate arrival timestamp)
10. `ARRIVAL_DELAY` (Target ground truth)
11. `CANCELLATION_REASON` (Post-flight event reason)
12. `AIR_SYSTEM_DELAY` (NAS delay post-hoc breakdown)
13. `SECURITY_DELAY` (Security delay post-hoc breakdown)
14. `AIRLINE_DELAY` (Carrier delay post-hoc breakdown)
15. `LATE_AIRCRAFT_DELAY` (Late inbound aircraft breakdown)
16. `WEATHER_DELAY` (Weather delay breakdown)
17. `DIVERTED` / `CANCELLED` (Operational outcomes; not pre-departure features)

---

## 8. Recommended Cleaning and Preprocessing Actions for Member 1B

When implementing `notebooks/02_data_preprocessing.ipynb` and `src/preprocessing.py`, Member 1B should execute the following operations:

1. **Filter Non-Operative Flights:**
   - Drop rows where `CANCELLED == 1` (89,884 rows) or `DIVERTED == 1` (15,187 rows).
   - This cleanly eliminates all missing `ARRIVAL_DELAY` values and retains 5,714,008 valid flight records.
2. **Reconcile Dual Airport Codes:**
   - For October through December 2015, convert 5-digit numeric DOT codes (e.g., `10397`, `12892`) back to their corresponding 3-letter IATA codes (`ABI`, `ORD`), using the BTS DOT crosswalk or frequency alignment, or treat them consistently across training and test splits.
3. **Parse and Cycle-Encode Temporal Features:**
   - Convert `SCHEDULED_DEPARTURE` and `SCHEDULED_ARRIVAL` from `HHMM` integer formats into continuous decimal hours ($0 \le h < 24$) and minutes ($0 \le m < 1440$).
   - Apply cyclical trigonometric transformations:
     $$\text{hour\_sin} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$
     $$\text{dow\_sin} = \sin\left(\frac{2\pi \cdot \text{day\_of\_week}}{7}\right), \quad \text{dow\_cos} = \cos\left(\frac{2\pi \cdot \text{day\_of\_week}}{7}\right)$$
     $$\text{month\_sin} = \sin\left(\frac{2\pi \cdot \text{month}}{12}\right), \quad \text{month\_cos} = \cos\left(\frac{2\pi \cdot \text{month}}{12}\right)$$
4. **Implement Chronological Data Splitting:**
   - To completely prevent temporal lookahead leakage, **never use random k-fold or random train/test splits**.
   - Use strict chronological partitions:
     - **Training Set:** January 1 – August 31, 2015 (~67% of data)
     - **Validation Set:** September 1 – October 31, 2015 (~16% of data)
     - **Test Set:** November 1 – December 31, 2015 (~17% of data)
5. **Compute Historical Delay Priors Strictly on Training Set:**
   - If generating target-encoded priors (carrier historical delay rate, route historical delay rate), compute them **only on the training split** and map them to validation/test sets to prevent data leakage.
6. **Feature Scaling:**
   - Fit `StandardScaler` only on the training set for continuous features (`DISTANCE`, `SCHEDULED_TIME`).
