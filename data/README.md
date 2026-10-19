# Dataset Directory Instructions

## Where to Place Your Data
Please place your flight delay dataset CSV file(s) into this directory:
- Recommended path: `data/raw/` (or directly in `data/`)
- Accepted formats: `.csv`, `.zip` (containing CSVs), `.parquet`

## Recommended Datasets
Any standard US domestic flight on-time performance dataset can be used:
1. **Kaggle: 2015 Flight Delays and Cancellations** (US DOT BTS)
   - Primary file: `flights.csv` (contains ~5.8M flights)
   - Supplementary files: `airlines.csv`, `airports.csv`
2. **US Bureau of Transportation Statistics (TranStats / BTS)**
   - On-Time Performance Reporting Carrier data (e.g., `Combined_Flights_2022.csv` or monthly files)
3. **Flight Delay Dataset 2019-2023** from Kaggle

---

## Required Column Concepts for Flight Delay Prediction

To predict whether a flight will arrive late ($\ge 15$ minutes) using **only information available before departure**, the dataset should provide:

| Concept | Common Column Names | Role in Phase 1 |
| :--- | :--- | :--- |
| **Scheduled Departure Time** | `SCHEDULED_DEPARTURE`, `CRSDepTime`, `crs_dep_time` | **Predictive Feature** (Hour, minute, time of day) |
| **Scheduled Arrival Time** | `SCHEDULED_ARRIVAL`, `CRSArrTime`, `crs_arr_time` | **Predictive Feature** (Scheduled arrival window) |
| **Scheduled Duration / Distance** | `SCHEDULED_TIME`, `CRSElapsedTime`, `DISTANCE`, `Distance` | **Predictive Feature** (Planned flight duration & distance) |
| **Airline / Carrier** | `AIRLINE`, `OP_UNIQUE_CARRIER`, `Reporting_Airline`, `carrier` | **Predictive Feature** (Carrier operational reliability) |
| **Origin Airport** | `ORIGIN_AIRPORT`, `ORIGIN`, `Origin` | **Predictive Feature** (Origin hub congestion/stats) |
| **Destination Airport** | `DESTINATION_AIRPORT`, `DEST`, `Dest` | **Predictive Feature** (Destination hub congestion/stats) |
| **Date Information** | `YEAR`, `MONTH`, `DAY`, `DAY_OF_WEEK`, `FL_DATE`, `FlightDate` | **Predictive Feature** (Seasonality, day of week effects) |
| **Arrival Delay** | `ARRIVAL_DELAY`, `ARR_DELAY`, `ArrDelay`, `ARR_DEL15` | **TARGET VARIABLE ONLY** ($\ge 15$ min $\rightarrow 1$, else $0$). *Excluded from inputs!* |
| **Cancellations & Diversions** | `CANCELLED`, `Cancelled`, `DIVERTED`, `Diverted` | **Filtering / Preprocessing** (Filter out flights that never arrived) |
| **Actual Departure Time / Delay** | `DEPARTURE_TIME`, `DEPARTURE_DELAY`, `DepTime`, `DepDelay` | ⚠️ **STRICTLY EXCLUDED (Data Leakage)** — occurs after scheduled departure |
| **Actual Arrival / In-flight times** | `ARRIVAL_TIME`, `TAXI_OUT`, `WHEELS_OFF`, `AIR_TIME`, `TAXI_IN` | ⚠️ **STRICTLY EXCLUDED (Data Leakage)** — occurs after scheduled departure |
| **Delay breakdown reasons** | `AIR_SYSTEM_DELAY`, `SECURITY_DELAY`, `AIRLINE_DELAY`, `WEATHER_DELAY` | ⚠️ **STRICTLY EXCLUDED (Data Leakage)** — only known post-flight |

Once you place the file in `data/raw/` or `data/`, rerun `notebooks/01_data_exploration.ipynb` or `python src/data_loader.py` to automatically inspect the dataset.
