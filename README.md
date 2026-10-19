# Flight Delay Prediction via Gradient Boosting and Temporal Deep Learning

## Phase 1: Traditional Machine Learning

An academic machine learning project aimed at predicting commercial flight delays prior to departure using historical flight performance data.

---

## 🎯 Objective
Predict whether a commercial flight will arrive **at least 15 minutes late** (`Arrival Delay >= 15 minutes`) using **only information available before its scheduled departure time**.

---

## ⚠️ Critical Rule: Anti-Leakage Policy
In flight delay prediction, data leakage occurs when models use information that is only known after scheduled departure. 

- **Permitted Predictive Features (Known Pre-Departure):**
  - Scheduled departure time (`SCHEDULED_DEPARTURE` / `CRSDepTime`)
  - Scheduled arrival time (`SCHEDULED_ARRIVAL` / `CRSArrTime`)
  - Scheduled flight duration / distance (`SCHEDULED_TIME`, `DISTANCE`)
  - Airline / Operating Carrier (`AIRLINE` / `Reporting_Airline`)
  - Origin airport (`ORIGIN_AIRPORT` / `ORIGIN`)
  - Destination airport (`DESTINATION_AIRPORT` / `DEST`)
  - Date information (`YEAR`, `MONTH`, `DAY`, `DAY_OF_WEEK`, `FL_DATE`)

- **Strictly Excluded Post-Departure Features (Data Leakage):**
  - ❌ Actual departure time (`DEPARTURE_TIME` / `DepTime`)
  - ❌ Departure delay (`DEPARTURE_DELAY` / `DepDelay` / `DepDel15`)
  - ❌ Taxi-out time (`TAXI_OUT`)
  - ❌ Wheels-off time (`WHEELS_OFF`)
  - ❌ In-flight airborne time (`AIR_TIME`)
  - ❌ Actual arrival time (`ARRIVAL_TIME` / `ArrTime`)
  - ❌ Taxi-in time (`TAXI_IN`)
  - ❌ Delay cause breakdown columns (`AIR_SYSTEM_DELAY`, `SECURITY_DELAY`, `AIRLINE_DELAY`, `LATE_AIRCRAFT_DELAY`, `WEATHER_DELAY`)

- **Target Variable (Supervised Signal Only):**
  - Arrival delay (`ARRIVAL_DELAY` / `ArrDelay`) is used **strictly** to compute the binary target $y \in \{0, 1\}$:
    $$y = \begin{cases} 1 & \text{if } \text{Arrival Delay} \ge 15 \text{ min} \\ 0 & \text{if } \text{Arrival Delay} < 15 \text{ min} \end{cases}$$
  - It is **never** passed into the feature matrix $X$.

---

## 🛠️ Technology Stack
- **Language**: Python 3.12+
- **Data Manipulation**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **Machine Learning**: `scikit-learn`, `xgboost`
- **Interactive Analysis**: `jupyter`, `ipykernel`

---

## 📁 Project Directory Structure
```
Flight_Delay_Prediction/
├── data/
│   ├── raw/                 # Place raw CSV files here (e.g., flights.csv)
│   ├── processed/           # Cached clean datasets and feature matrices
│   └── README.md            # Dataset placement & column schema guide
├── notebooks/
│   ├── 01_data_exploration.ipynb            # Initial data exploration & schema audit
│   └── 02_exploratory_data_analysis.ipynb   # In-depth EDA, distributions & leak audits
├── src/
│   ├── __init__.py          # Package initializer
│   ├── config.py            # Project paths, aliases & leakage guardrails
│   ├── data_loader.py       # Automated dataset discovery, inspection & loading
│   └── utils.py             # Time parsing, group delay statistics & figure saving
├── results/
│   ├── figures/             # Saved exploratory & evaluation plots
│   └── metrics/             # Evaluation logs & metrics
├── requirements.txt         # Pinned Python package dependencies
├── .gitignore               # Ignores large datasets, caches & venv
└── README.md                # Project documentation
```

---

## 🚀 Getting Started

### 1. Environment Setup
A dedicated virtual environment `venv` is configured in this repository.

Activate it using:
```bash
source venv/bin/activate
```

If setting up on a new machine:
```bash
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name flight-delay-venv --display-name "Python (Flight Delay Venv)"
```

### 2. Providing the Dataset
Place your flight dataset CSV (e.g. Kaggle's 2015 `flights.csv` or US DOT BTS `Combined_Flights_2022.csv`) inside:
```
data/raw/
```
Refer to `data/README.md` for complete column recommendations.

### 3. Run Automated Dataset Inspection
Run the automated schema and leakage inspection script:
```bash
./venv/bin/python src/data_loader.py
```

### 4. Launch Jupyter Notebooks
Start JupyterLab or Jupyter Notebook:
```bash
./venv/bin/jupyter lab
```
Open `notebooks/01_data_exploration.ipynb` and select kernel `Python (Flight Delay Venv)`.

---

## 📌 Phase 1 Roadmap
- [x] Workspace structure setup (`data/`, `notebooks/`, `src/`, `results/`)
- [x] Environment configuration & `requirements.txt`
- [x] Column alias mapping & schema detection (`src/config.py`, `src/data_loader.py`)
- [x] Anti-leakage guardrail definitions & operational filtering audit
- [x] Initial exploratory data exploration notebook (`notebooks/01_data_exploration.ipynb`)
- [x] Comprehensive Step 2 EDA notebook (`notebooks/02_exploratory_data_analysis.ipynb`)
- [ ] *Pending user data placement*: Provide `flights.csv` in `data/raw/` to generate live visualizations and summary stats
- [ ] Step 3: Preprocessing & Feature Engineering (Cyclical time encoding, airport frequency, route delay historical priors)
- [ ] Step 4: Baseline ML Models: Logistic Regression & Random Forest
- [ ] Step 5: Gradient Boosting Model: XGBoost Classifier with time-based train/validation split
- [ ] Step 6: Evaluation & Benchmarking (Precision, Recall, ROC-AUC, PR-AUC, Confusion Matrix)
# Flight_Delay_Prediction
