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
├── app/
│   └── app.py                         # Streamlit interactive web application frontend
├── data/
│   ├── raw/                           # Raw flight archives
│   └── processed/
│       ├── flights_cleaned.csv        # Sanitized operational cohort (24,522 rows)
│       └── flights_features.csv       # Model-ready feature dataset (46 columns)
├── models/
│   └── classification_pipeline.joblib # Validated production classification artifact
├── notebooks/
│   ├── 01_dataset_inspection.ipynb    # Member 1A: Schema discovery, data quality & leakage checks
│   ├── 02_data_cleaning.ipynb         # Member 1B: Filtering, target formulation, standardized dates
│   ├── 03a_eda_time_series.ipynb      # Member 2A: Diurnal hourly escalation & seasonality analysis
│   ├── 03b_feature_engineering.ipynb  # Member 2B: Cyclical encodings, empirical Bayes priors, splits
│   ├── 04_regression_experiments.ipynb# Member 3: Continuous OLS, GD, polynomial, Lasso/Ridge
│   ├── 05_classification_models.ipynb # Member 4: Logistic MLE, first-principles GD, thresholding
│   ├── 06_pca_experiment.ipynb        # Member 5: PCA dimensionality reduction (Syllabus Topic 11)
│   └── 07_final_model_comparison.ipynb# Member 5: Unified scorecard, frozen test evaluation
├── reports/                           # Comprehensive technical handoffs and member reports
│   ├── member1A_data_quality_report.md
│   ├── member1B_preprocessing_report.md
│   ├── member2A_eda_report.md
│   ├── member2B_feature_engineering_report.md
│   ├── member3_regression_report.md
│   ├── member4_classification_report.md
│   └── member5_final_report.md        # Member 5: Final synthesis and system integration report
├── results/
│   └── figures/                       # Publication figures (classification, regression, eda, pca, comparison)
├── src/                               # Modular, reusable production code
│   ├── classification.py              # Logistic Regression, GD, pipeline builder, evaluation
│   ├── config.py                      # Canonical schema mappings and directory configurations
│   ├── data_loader.py                 # Automated dataset discovery and inspection
│   ├── features.py                    # Pre-departure feature engineering & Bayes priors
│   ├── preprocessing.py               # Operational cleaning, filtering, chronological splitting
│   ├── regression_experiments.py      # OLS, gradient descent, regularization
│   ├── rescheduling_simulator.py      # Member 5: Offline flight departure rescheduling simulator
│   └── utils.py                       # Helper utilities
├── README.md                          # Project documentation
└── requirements.txt                   # Project dependencies (including Streamlit & Plotly)
```

---

## 🚀 Getting Started & Launching the Application

### 1. Environment Setup
Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Launch Streamlit Web Application (Member 5 Integration)
Launch the interactive web application frontend:
```bash
streamlit run app/app.py
```
The interface provides five core modules:
1. **🏠 Home / Overview:** Project objectives, DOT BTS definition, 8-stage ML lifecycle, and disclaimers.
2. **🛫 Predict Flight Delay:** Input predeparture attributes to receive real-time probability estimates, 3-tier risk gauges (Low/Moderate/High), and operational threshold alerts ($\tau^* = 0.33$).
3. **📊 Delay Analytics:** Interactive Plotly charts of diurnal hourly escalation, carrier rates, and airport hub delays.
4. **🔄 Rescheduling Simulator:** Explore hypothetical departure shifts (±120 min) and rank options by estimated delay probability.
5. **🏆 Model Performance:** Consolidated scorecards, test confusion matrices, ROC/PR curves, and calibration reliability diagrams.

### 3. Run Automated Rescheduling Simulator in Python
Run the command-line self-test of the offline simulator:
```bash
python -m src.rescheduling_simulator
```

### 4. Launch Jupyter Notebooks
Start Jupyter to inspect the step-by-step educational notebooks:
```bash
jupyter lab
# Or run individual notebooks:
# notebooks/06_pca_experiment.ipynb
# notebooks/07_final_model_comparison.ipynb
```

---

## 📌 Phase 1 Academic Roadmap & Status
- [x] **Task 1A (Member 1A):** Dataset inspection, data quality audit, and ML lifecycle definition (`reports/member1A_data_quality_report.md`).
- [x] **Task 1B (Member 1B):** Operational cleaning, cancellation/diversion filtering, binary target creation, and dataset export (`reports/member1B_preprocessing_report.md`).
- [x] **Task 2A (Member 2A):** Exploratory data analysis, diurnal delay escalation, and descriptive time-series analysis (`reports/member2A_eda_report.md`).
- [x] **Task 2B (Member 2B):** Pre-departure feature engineering, empirical Bayes priors, and chronological partitioning (`reports/member2B_feature_engineering_report.md`).
- [x] **Task 3 (Member 3):** Continuous delay regression, OLS, GD, polynomial expansions, Gauss-Markov diagnostics, Ridge/Lasso regularization (`reports/member3_regression_report.md`).
- [x] **Task 4 (Member 4):** Binary classification, Logistic Regression via MLE, first-principles BCE GD, asymmetric cost threshold tuning ($\tau^*=0.33$), probability calibration, and pipeline artifact export (`reports/member4_classification_report.md`).
- [x] **Task 5 (Member 5):** PCA experiment (Syllabus Topic 11), unified test set evaluation, offline rescheduling simulation, Streamlit web frontend, and final Phase 1 synthesis (`reports/member5_final_report.md`).

---

## 👥 Team & Contributors

| Member | Role | Key Contributions |
| :--- | :--- | :--- |
| **Sarvjeet Yadav** | Member 1A | Dataset discovery, schema validation, data quality audit, anti-leakage audit, initial pipeline handoff |
| **Ajeesh** | Member 1B | Operational data cleaning, cancellation filtering, binary target formulation |
| **Akshat Chauhan** | Member 2A | Exploratory data analysis, diurnal delay escalation, descriptive time-series |
| **Pranjal Modi** | Member 2B | Pre-departure feature engineering, empirical Bayes priors, chronological splits |
| **TechySuryansh** | Member 3 / 4 / 5 | Regression experiments, classification pipelines, offline simulator, and Streamlit app integration |
