# Flight Delay Prediction

## Phase 1: Traditional Machine Learning (Academic Syllabus)

A foundational Machine Learning project aimed at predicting commercial flight delays prior to departure using classical machine learning algorithms, statistical principles, and optimization techniques.

---

## 📚 Machine Learning Syllabus Coverage
This project strictly covers the following beginner-level topics from the academic curriculum:
1. **Machine Learning Project Lifecycle** (Problem formulation, EDA, Preprocessing, Modeling, Evaluation)
2. **Simple and Multiple Linear Regression using OLS**
3. **Batch Gradient Descent (BGD)**
4. **Stochastic Gradient Descent (SGD) and Mini-Batch Gradient Descent (MBGD)**
5. **Regression and Classification Evaluation Metrics** (Confusion Matrix, Precision, Recall, F1, ROC-AUC, PR-AUC)
6. **Polynomial Regression and Linear Regression Assumptions**
7. **Bias-Variance Tradeoff**
8. **Feature Selection**
9. **Principal Component Analysis (PCA)**
10. **Regularization** (L1 Lasso, L2 Ridge)
11. **Time Series & Chronological Analysis** (Zero lookahead leakage)
12. **Maximum Likelihood Estimation (MLE) and Logistic Regression**
13. **Gradient Descent for Logistic Regression and Multiclass Classification**

> **Strict Rule:** Only beginner-level concepts from this syllabus are implemented. Advanced methods (XGBoost, Random Forest, Deep Learning, neural networks, embeddings, SHAP, AutoML) are strictly excluded.

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
- **Language**: Python 3.12
- **Data Manipulation**: `pandas`, `NumPy`
- **Visualization**: `matplotlib`, `seaborn`
- **Machine Learning**: `scikit-learn`
- **Interactive Notebooks**: `Jupyter Notebook`

---

## 📁 Project Directory Structure
```
Flight_Delay_Prediction/
├── data/                    # Place dataset CSV files here (e.g., data/raw/flights.csv)
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Lifecycle, schema discovery, data quality & leakage checks
│   ├── 02_data_preprocessing.ipynb    # Filtering, chronological split, scaling & encoding
│   ├── 03_logistic_regression.ipynb   # MLE, Logistic Regression, L1/L2 Regularization
│   ├── 04_gradient_descent.ipynb      # BGD, SGD, and Mini-Batch GD from first principles
│   └── 05_model_evaluation.ipynb     # Confusion Matrix, Precision, Recall, F1, ROC/PR curves
├── src/                     # Reusable modular pipeline code
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   └── utils.py
├── results/
│   └── figures/             # Output evaluation plots and figures
├── README.md                # Project documentation
└── requirements.txt         # Dependencies
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
Open any notebook in `notebooks/` and select kernel `Python (Flight Delay Venv)`.

---

## 📌 Phase 1 Academic Roadmap
- [x] Workspace structure setup (`data/`, `notebooks/`, `src/`, `results/figures/`, `requirements.txt`)
- [x] Traditional ML dependency configuration (`pandas`, `NumPy`, `scikit-learn`, `matplotlib`, `seaborn`)
- [x] Anti-leakage guardrail definitions & canonical schema alias mapping
- [x] `notebooks/01_data_exploration.ipynb` (Project Lifecycle, schema inspection, data quality, anti-leakage audit)
- [x] `notebooks/02_data_preprocessing.ipynb` (Operational filtering, chronological splitting, scaling & encoding)
- [x] `notebooks/03_logistic_regression.ipynb` (MLE foundation, Binary cross-entropy, L1/L2 Regularization, Bias-Variance)
- [x] `notebooks/04_gradient_descent.ipynb` (First-principles Batch, Stochastic, and Mini-Batch Gradient Descent)
- [x] `notebooks/05_model_evaluation.ipynb` (Confusion Matrix, Precision, Recall, F1, ROC-AUC, PR-AUC, Decision Thresholds)
- [ ] *Pending user data placement*: Place `flights.csv` into `data/raw/` (or `data/`)
- [ ] *Pending user approval*: Fit models, train gradient descent algorithms, and benchmark evaluation metrics
