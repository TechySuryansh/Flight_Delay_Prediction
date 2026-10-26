# ✈️ Commercial Flight Delay Prediction & Rescheduling Engine

> **Applied Machine Learning (AML) — Phase 1 Project**  
> *A foundational, leakage-safe Machine Learning system predicting commercial flight delays prior to pushback and simulating tactical departure slot rescheduling.*

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Classical_Algorithms-orange.svg?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Phase_1-Completed-brightgreen.svg)]()
[![Recall](https://img.shields.io/badge/Delay_Recall-98.06%25-success.svg)]()

---

## 👥 Project Team & Work Distribution

| # | Team Member | Roll No. / Student ID | Assigned Module | Key Technical Deliverables |
| :-: | :--- | :---: | :--- | :--- |
| **1** | **Sarvjeet Yadav** | `2401010424` | **Task 1: Data Audit, Quality Quarantine & Preprocessing** | • Raw schema inspection of 5.8M BTS records<br>• 100% missing value resolution via non-arrival quarantine<br>• Formulation of binary target ($y = \mathbb{I}(\text{ArrDelay} \ge 15)$)<br>• Data pipelines (`notebooks/01`, `02` & `src/preprocessing.py`) |
| **2** | **Pranjal Modi** | `2401010334` | **Task 2: EDA, Time-Series & Feature Engineering** | • Diurnal hourly delay escalation analysis ($26\% \rightarrow 52\%$ evening surge)<br>• Strict chronological data partitioning (Train / Val / Frozen Test)<br>• Trigonometric cyclical encodings ($\sin/\cos$) for 4 temporal horizons<br>• Out-Of-Fold Empirical Bayes smoothed target priors ($m=10$) |
| **3** | **Suryansh Singh** | `2401010467` | **Task 3: Continuous Delay Regression Experiments** | • Simple and Multiple Ordinary Least Squares (OLS) closed-form<br>• First-principles Batch GD, Mini-Batch GD ($B=64$), and SGD ($B=1$)<br>• Gauss-Markov diagnostic audit (Durbin-Watson $= 2.0081$)<br>• Bias-Variance polynomial complexity & Lasso/Ridge regularization |
| **4** | **Akshat Chauhan** | `2401010055` | **Task 4: Binary Classification & Operational Calibration** | • Logistic Regression via Maximum Likelihood Estimation (MLE)<br>• First-principles Binary Cross-Entropy GD optimizer<br>• Cost-sensitive asymmetric threshold calibration ($\tau^* = 0.33$)<br>• Probability calibration (Brier Score $= 0.2241$) & pipeline serialization |
| **5** | **Ajeesh Amreet** | `2401010048` | **Task 5: Dimensionality Reduction & Rescheduling Simulation** | • Unsupervised PCA compression (15 components capture $96.72\%$ variance)<br>• Unified cross-partition model benchmark scorecard<br>• Discrete-slot departure rescheduling simulation engine ($\Delta t \in \pm 120$ min)<br>• Comprehensive Phase 1 synthesis and evaluation pipeline (`notebooks/06`, `07`) |

---

## 📖 Executive Summary & Core Objective

Flight delays cost the US aviation industry and economy over **\$33 billion annually**. While post-departure flight data can trivially classify delays, practical airline dispatch systems require predictions made **well before aircraft pushback**.

The primary objective is to predict whether a commercial flight will arrive **15+ minutes late** (BTS standard) using **only information available at least 2 hours prior to scheduled departure** ($T - 2$ hours). High-risk flights are then fed into a **Rescheduling Simulator** that identifies optimal alternative departure windows to proactively mitigate delays.

```
       OPERATIONAL PREDICTION BOUNDARY (T - 2 Hours)
─────────────────────────────────────────────────────────────► TIME
         ▲                                   ▲
  [Prediction Time]                 [Scheduled Departure]
  All features strictly             Flight leaves gate.
  frozen 120 mins before            17 post-departure telemetry
  departure schedule.               variables quarantined.
```

---

## 🛡️ Anti-Leakage & Operational Discipline

To guarantee zero lookahead data leakage, we enforced strict isolation between pre-departure predictors and post-departure outcomes:

* **✅ Permitted Predictors (Available at $T - 2$h):**
  * Scheduled departure & arrival timestamps (`SCHEDULED_DEPARTURE`, `SCHEDULED_ARRIVAL`)
  * Scheduled flight duration & flight distance (`SCHEDULED_TIME`, `DISTANCE`)
  * Carrier code (`AIRLINE`), Origin hub (`ORIGIN_AIRPORT`), Destination (`DESTINATION_AIRPORT`)
  * Calendar temporal features (`MONTH`, `DAY`, `DAY_OF_WEEK`, cyclic representations)
* **❌ Quarantined Post-Departure Features (Strictly Discarded):**
  * Actual departure & arrival times (`DEPARTURE_TIME`, `ARRIVAL_TIME`)
  * Departure delay (`DEPARTURE_DELAY`) & taxi telemetry (`TAXI_OUT`, `TAXI_IN`, `WHEELS_OFF`, `WHEELS_ON`)
  * In-flight airborne time (`AIR_TIME`, `ELAPSED_TIME`)
  * Post-hoc delay causes (`CARRIER_DELAY`, `WEATHER_DELAY`, `NAS_DELAY`, `SECURITY_DELAY`, `LATE_AIRCRAFT_DELAY`)
* **📅 Zero Lookahead Chronological Splitting:**
  * Shuffling time-series records causes data snooping. Data was partitioned strictly by date:
    * **Train Set:** Jan 1 – Aug 31, 2015 ($N = 16,366$, 66.7%, Delay rate: $38.63\%$)
    * **Validation Set:** Sep 1 – Oct 31, 2015 ($N = 4,114$, 16.8%, Delay rate: $38.99\%$)
    * **Frozen Test Set:** Nov 1 – Dec 31, 2015 ($N = 4,042$, 16.5%, Delay rate: $38.32\%$)
  * All encoders, scalers, and models were fitted **strictly on the Train partition**.

---

## 🔬 Mathematical Architecture & Syllabus Alignment

This project implements all foundational topics from the Applied Machine Learning (AML) syllabus:

### 1. Cyclical Trigonometric Encodings
To avoid the artificial numerical discontinuity between 23:59 and 00:00, temporal coordinates are mapped to harmonic unit circles:
$$\text{DEP\_HOUR\_SIN} = \sin\left(\frac{2\pi \cdot h}{24}\right), \quad \text{DEP\_HOUR\_COS} = \cos\left(\frac{2\pi \cdot h}{24}\right)$$

### 2. Empirical Bayes Smoothed Priors
High-cardinality categorical entities (airlines and airports) are mapped to smoothed delay rates computed with 5-fold Out-Of-Fold (OOF) cross-validation to prevent target leakage:
$$\text{Prior}(c) = \frac{\sum_{i \in \text{Train}_c} y_i + m \cdot \bar{y}_{\text{global}}}{N_{c,\text{train}} + m}, \quad m = 10.0$$

### 3. First-Principles Gradient Descent Optimization
We implemented vectorized Batch GD, Mini-Batch GD ($B=64$), and Stochastic GD ($B=1$) from scratch, proving convergence to the analytical OLS normal equation:
$$\theta^{(t+1)} = \theta^{(t)} - \frac{\alpha}{|B_t|} X_{B_t}^T \left(X_{B_t} \theta^{(t)} - y_{B_t}\right)$$

### 4. Asymmetric Cost Threshold Calibration
In airline operations, a **False Negative** (unpredicted delay causing missed passenger connections and stranded crew) costs 5–10× more than a **False Positive** (staging buffer). We calibrated $\tau \in [0.10, 0.90]$, selecting $\tau^* = 0.33$:
$$\hat{y} = \mathbb{I}(P(\text{delay} \mid x) \ge 0.33) \implies \textbf{98.06\% Delay Detection Recall}$$

### 5. Principal Component Analysis (PCA)
Features are projected onto orthogonal eigenvectors of the sample covariance matrix $\Sigma = \frac{1}{m} X^T X$:
$$\Sigma v_i = \lambda_i v_i$$
15 components preserve **96.72%** of cumulative variance, eliminating collinearity with zero loss in downstream classification discrimination.

---

## 📊 Consolidated Model Benchmark Scorecard

| Model Architecture | Task Formulation | Evaluation Partition | Primary Metric | Secondary Metric | $R^2$ / Recall |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Dummy Baseline (Mean)** | Continuous Delay | Chronological Val | RMSE = 32.41 min | MAE = 25.68 min | $R^2 = -0.0004$ |
| **Simple OLS (`DEP_HOUR`)** | Continuous Delay | Chronological Val | RMSE = 31.75 min | MAE = 25.14 min | $R^2 = +0.0400$ |
| **Multiple OLS (31 Features)** | Continuous Delay | Chronological Val | RMSE = 31.77 min | MAE = 25.16 min | $R^2 = +0.0388$ |
| **Ridge Regression ($\alpha=1000$)**| Continuous Delay | Chronological Val | RMSE = 31.75 min | MAE = 25.15 min | $R^2 = +0.0400$ |
| **Lasso Regression ($\alpha=0.1$)** | Continuous Delay | Chronological Val | **RMSE = 31.74 min**| **MAE = 25.14 min**| $\mathbf{R^2 = +0.0405}$ |
| **Multiple OLS (Frozen Test)** | Continuous Delay | Frozen Test (Nov–Dec)| RMSE = 31.93 min | MAE = 25.39 min | $R^2 = +0.0475$ |
| **Lasso Regression (Frozen Test)**| Continuous Delay | Frozen Test (Nov–Dec)| RMSE = 31.95 min | MAE = 25.41 min | $R^2 = +0.0464$ |
| **Majority Dummy Baseline** | Binary Delay Flag| Frozen Test (Nov–Dec)| ROC-AUC = 0.5000 | PR-AUC = 0.3832 | Recall = 0.00% |
| **Logistic Regression ($\tau=0.50$)**| Binary Delay Flag| Frozen Test (Nov–Dec)| ROC-AUC = 0.6019 | PR-AUC = 0.4663 | Recall = 51.32% |
| **Logistic Regression ($\tau^*=0.33$)**| Binary Delay Flag| Frozen Test (Nov–Dec)| **ROC-AUC = 0.6019**| **PR-AUC = 0.4663**| $\mathbf{\text{Recall} = 98.06\%}$ |
| **PCA Logistic ($k=15$ PCs)** | Binary Delay Flag| Chronological Val | ROC-AUC = 0.5937 | PR-AUC = 0.4578 | Recall = 50.12% |

> **Key Empirical Finding:** Pre-departure continuous delay minutes possess an irreducible variance floor ($\sigma^2 \approx 31.7$ min, $R^2 \approx 0.045$) due to mid-flight tactical weather and ATC ground stops that occur hours after departure. Conversely, probability-calibrated binary classification at $\tau^* = 0.33$ successfully identifies **98.06% of all delayed flights**.

---

## 🗂️ Repository Structure

```
Flight_Delay_Prediction/
├── data/
│   ├── raw/                           # Raw flight records (2015 BTS format)
│   └── processed/
│       ├── flights_cleaned.csv        # Sanitized operational cohort (24,522 rows)
│       └── flights_features.csv       # Model-ready engineered dataset (46 cols)
├── models/
│   └── classification_pipeline.joblib # Validated production classification model
├── notebooks/
│   ├── 01_dataset_inspection.ipynb    # Task 1: Schema audit, quality & leakage checks
│   ├── 02_data_cleaning.ipynb         # Task 1: Non-arrival quarantine & target formulation
│   ├── 03a_eda_time_series.ipynb      # Task 2: Diurnal escalation & time-series analysis
│   ├── 03b_feature_engineering.ipynb  # Task 2: Cyclical transforms, Bayes priors & splitting
│   ├── 04_regression_experiments.ipynb# Task 3: OLS, GD from scratch, polynomials, Lasso/Ridge
│   ├── 05_classification_models.ipynb # Task 4: Logistic MLE, calibration & threshold tuning
│   ├── 06_pca_experiment.ipynb        # Task 5: PCA eigendecomposition & dimensionality reduction
│   └── 07_final_model_comparison.ipynb# Task 5: Unified scorecard & frozen test set audit
├── reports/                           # Member handoffs & academic project report
│   ├── member1A_data_quality_report.md
│   ├── member1B_preprocessing_report.md
│   ├── member2A_eda_report.md
│   ├── member2B_feature_engineering_report.md
│   ├── member3_regression_report.md
│   ├── member4_classification_report.md
│   ├── member5_final_report.md
│   └── overleaf_project_report.tex    # Complete LaTeX Overleaf report
├── results/
│   └── figures/                       # High-resolution figures across all tasks
│       ├── classification/            # ROC, PR, threshold tradeoff, calibration curves
│       ├── comparison/                # Test curves, confusion matrices, paradigm plots
│       ├── eda/                       # Hourly escalation, airline/airport heatmaps
│       ├── features/                  # Feature engineering summary diagram
│       ├── pca/                       # Scree plots, loadings, subspace comparison
│       └── regression/                # OLS lines, GD loss, bias-variance, Lasso paths
├── src/                               # Modular, production-grade Python package
│   ├── classification.py              # Logistic Regression, GD, pipeline builder
│   ├── config.py                      # Canonical paths and column definitions
│   ├── data_loader.py                 # Safe dataset ingestion and schema inspection
│   ├── features.py                    # Pre-departure feature generation & OOF priors
│   ├── preprocessing.py               # Filtering, quarantine & chronological splitting
│   ├── regression_experiments.py      # OLS, GD optimizers, regularization, diagnostics
│   ├── rescheduling_simulator.py      # Offline departure slot reassignment engine
│   └── utils.py                       # Logging and general helpers
├── requirements.txt                   # Dependency specifications
└── README.md                          # Project documentation
```

---

## ⚡ Quickstart & How to Run

### 1. Setup Environment
Clone the repository and install dependencies in a virtual environment:
```bash
git clone https://github.com/TechySuryansh/Flight_Delay_Prediction.git
cd Flight_Delay_Prediction

# Create and activate Python virtual environment
python3 -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run Offline Rescheduling Simulator
Test the automated slot evaluation and delay delta calculations from the terminal:
```bash
python -m src.rescheduling_simulator
```

### 3. Explore Interactive Notebooks
Launch Jupyter to review step-by-step mathematical workflows and experiments:
```bash
jupyter lab
```
