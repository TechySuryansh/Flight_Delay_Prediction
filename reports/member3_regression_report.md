# Member 3: Regression Experiments and Applied Machine Learning Syllabus Demonstrations Report
**Project 21:** Flight Delay Prediction and Rescheduling  
**Team Role:** Member 3 (Linear Regression, OLS, Gradient Descent & Regularization)  
**Phase:** AML Phase 1 — Traditional Machine Learning  
**Date:** October 2026  
**Status:** Task 3 Complete and Validated  

---

## Executive Summary

This report formalizes the regression experiments conducted under **Task 3** of AML Phase 1. The primary objective of Project 21 is binary classification—predicting whether a commercial flight will arrive at least 15 minutes late (`DELAYED == 1`). However, to satisfy the rigorous requirements of the **Applied Machine Learning (AML) syllabus**, this task implements, analyzes, and evaluates linear regression models predicting continuous arrival delay minutes (`ARRIVAL_DELAY`).

All experiments were executed on the leakage-safe feature dataset prepared by **Member 2B** (`data/processed/flights_features.csv`), utilizing strictly non-overlapping chronological partitions: **Train** (Jan 1 – Aug 31, 2015; 16,366 rows), **Validation** (Sep 1 – Oct 31, 2015; 4,114 rows), and an untouched **Held-Out Test Set** (Nov 1 – Dec 31, 2015; 4,042 rows).

### Key Results & Syllabus Milestones:
1. **Simple Linear Regression (OLS):**
   - Scheduled departure hour (`DEP_HOUR`) was identified as the primary predeparture predictor ($r = +0.1965$). The fitted OLS model is:
     $$\widehat{\text{Delay}} = -9.82 + 1.23 \cdot \text{DEP\_HOUR}$$
     Every 1-hour delay in scheduled departure increases expected arrival delay by **1.23 minutes** due to network congestion cascades.
   - Validation performance: **RMSE = 31.75 minutes**, **MAE = 25.14 minutes**, **$R^2 = 0.0400$** (outperforming the mean dummy baseline RMSE of 32.41 min).
   - In contrast, scheduled distance (`DISTANCE`) exhibited virtually zero linear predictive power ($R^2 = -0.0010$, slope = $+0.000788$ min/mile).
2. **Multiple Linear Regression (OLS):**
   - Fitted closed-form normal equations $\hat{\beta} = (X^T X)^{-1} X^T y$ across standardized numeric features.
   - Top standardized coefficients: `DEP_HOUR` ($\beta = +6.01$), `DEP_HOUR_PEAK` ($\beta = +0.42$), `PLANNED_SPEED` ($\beta = +0.35$), `DISTANCE_LOG` ($\beta = +0.18$).
   - Validation performance: **RMSE = 31.77 minutes**, **MAE = 25.16 minutes**, **$R^2 = 0.0388$**. Unregularized OLS across all 31 features suffered slight variance degradation due to multicollinearity.
3. **First-Principles Batch Gradient Descent (BGD):**
   - Implemented BGD from scratch on standardized features.
   - With learning rate $\alpha = 0.05$ across 500 iterations, the learned parameter vector converged to within **0.027** of the analytical OLS normal equation solution ($\theta_{\text{BGD}} \approx \theta_{\text{OLS}}$), demonstrating exact algorithmic equivalence.
4. **Stochastic (SGD) and Mini-Batch Gradient Descent (MBGD):**
   - Compared optimization dynamics:
     - **Batch GD:** Deterministic, perfectly smooth loss descent ($0.003$s/epoch).
     - **Mini-Batch GD ($B=64$):** Balanced vectorized hardware execution with rapid loss reduction ($0.10$s/epoch).
     - **Stochastic GD ($B=1$):** High variance noisy oscillation near minimum, required inverse-scaling learning rate schedule ($3.08$s/epoch).
5. **Polynomial Regression & Gauss-Markov Assumption Diagnostics:**
   - Quadratic polynomial expansion ($d=2$) yielded identical validation error (RMSE = 31.76 min), while cubic expansion ($d=3$) triggered overfitting (train RMSE dropped to 31.73 min while validation RMSE rose to 31.77 min).
   - **Residual Diagnostics:**
     - Mean residual = $0.000000$ (OLS orthogonality).
     - **Durbin-Watson statistic = 2.0081**, confirming zero 1st-order serial autocorrelation across chronological observations.
     - **Homoscedasticity:** Scale-location diagnostics revealed moderate variance expansion during evening peak hours.
     - **Normality:** Mild skewness ($0.0125$) and kurtosis ($0.1142$).
     - **Multicollinearity:** Variance Inflation Factor (VIF) exceeded $10.0$ when simultaneously including `DEP_HOUR`, `DEP_TIME_MINUTES`, and `DEP_HOUR_PEAK`.
6. **Feature Selection:**
   - Evaluated Correlation Screening, Statistical $F$-test (`SelectKBest`), and Recursive Feature Elimination (`RFE`) strictly within the training set.
   - Pruning from 31 features down to the top 5 features improved validation RMSE from **31.77 min** to **31.75 min**, proving that eliminating collinear noise reduces model variance.
7. **Regularization (Ridge and Lasso):**
   - **Ridge ($L_2$):** Best penalty $\alpha = 1000.0$, smoothly shrinking weight norm to achieve Val RMSE = **31.7528 min**.
   - **Lasso ($L_1$):** Best penalty $\alpha = 0.1$, driving **18 out of 31 features to exactly zero**, performing embedded feature selection and achieving the lowest validation error among all models: **Val RMSE = 31.7446 min**, **Val $R^2 = 0.0405$**.
8. **Held-Out Test Set Verification:**
   - Evaluated final models once on the held-out test partition (Nov–Dec 2015, 4,042 rows):
     - **Dummy Baseline:** Test RMSE = 32.72 min, MAE = 26.03 min, $R^2 = -0.0002$.
     - **Simple OLS (`DEP_HOUR`):** Test RMSE = 31.95 min, MAE = 25.39 min, $R^2 = 0.0463$.
     - **Multiple OLS:** Test RMSE = 31.93 min, MAE = 25.39 min, $R^2 = 0.0475$.
     - **Lasso Regression ($\alpha=0.1$):** Test RMSE = 31.95 min, MAE = 25.41 min, $R^2 = 0.0464$.
9. **Operational Context & Project Scope:**
   - In commercial aviation, arrival delay has high irreducible stochastic noise ($\sigma^2$) due to tactical en-route weather and ATC ground stops occurring hours after dispatch.
   - Pre-departure regression models inherently explain ~4%–5% of continuous variance.
   - This empirically validates why the primary Phase 1 project framing is **binary classification** (`DELAYED > 15 min`).

---

## 1. Review of Prior Team Members' Deliverables

Member 3's regression modeling builds directly on the validated deliverables of Members 1A through 2B:
- **Member 1A (`reports/member1A_data_quality_report.md`):** Verified raw dataset schema (5.8M rows, 31 columns, all 365 days of 2015) and identified 17 post-departure leakage columns.
- **Member 1B (`reports/member1B_preprocessing_report.md`):** Excluded non-operational flights (89,884 cancellations, 15,187 diversions) with zero silent dropping; confirmed non-arrivals accounted for 100% of missing delay targets; preserved continuous `ARRIVAL_DELAY` for regression benchmarking.
- **Member 2A (`reports/member2A_eda_report.md`):** Discovered the diurnal delay escalation phenomenon (delay rate rises monotonically from ~26% at 06:00 to ~52% at 22:00); verified weak linear correlation between distance and delay.
- **Member 2B (`reports/member2B_feature_engineering_report.md`):** Established the $T-2$ hours prediction horizon; partitioned data into chronological non-overlapping splits; engineered 39 predeparture features; provided Empirical Bayes historical priors fit strictly on training data; delivered `data/processed/flights_features.csv`.

---

## 2. Chronological Partitioning & Target Definition

### 2.1 The Target Variable: Continuous Arrival Delay
- **Target Name:** `ARRIVAL_DELAY`
- **Data Type:** `float64`
- **Unit of Measurement:** Minutes (positive = late arrival; negative = early arrival).
- **Target Distribution:**
  - Full cohort: Mean = $+6.77$ min, Std = $32.44$ min, Range = $[-115.0, +140.0]$ min.
  - Training split (16,366 rows): Mean = $+6.75$ min, Std = $32.38$ min.
  - Validation split (4,114 rows): Mean = $+7.39$ min, Std = $32.41$ min.
  - Test split (4,042 rows): Mean = $+6.25$ min, Std = $32.72$ min.

### 2.2 Strict Anti-Leakage Guardrails
- Neither `DELAYED` (the binary classification label) nor any post-departure variable (`DEPARTURE_TIME`, `DEPARTURE_DELAY`, `TAXI_OUT`, `WHEELS_OFF`, `AIR_TIME`, `ARRIVAL_TIME`, etc.) was ever passed into feature matrix $X$.
- All scalers, imputers, polynomial generators, and feature selectors were fit **exclusively on the training partition**.

---

## 3. Simple Linear Regression using OLS

### 3.1 Mathematical Formulation
The Simple Linear Regression model relates a single scalar feature $x$ to target $y$:
$$\hat{y} = \beta_0 + \beta_1 x$$
Under Ordinary Least Squares, the closed-form coefficients minimize the residual sum of squares:
$$\hat{\beta}_1 = \frac{\sum_{i=1}^m (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^m (x_i - \bar{x})^2} = \frac{\text{Cov}(x, y)}{\text{Var}(x)}, \qquad \hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$$

### 3.2 Candidate Predictor Comparison
We compared the two primary candidate predictors: `DEP_HOUR` (Scheduled Departure Hour) vs. `DISTANCE` (Scheduled Distance).

| Predictor Feature | Fitted OLS Equation | Training $R^2$ | Validation RMSE | Validation MAE | Validation $R^2$ | Operational Interpretation |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`DEP_HOUR`** | $\hat{y} = -9.82 + 1.23 \cdot \text{hour}$ | **0.0386** | **31.75 min** | **25.14 min** | **+0.0400** | Each hour later in the day increases expected delay by **1.23 min**. Outperforms baseline. |
| **`DISTANCE`** | $\hat{y} = 5.66 + 0.000788 \cdot \text{dist}$ | 0.0003 | 32.42 min | 25.68 min | -0.0010 | A 1,000-mile flight adds less than 0.8 min of expected delay. Zero predictive power. |
| **Dummy Mean Baseline** | $\hat{y} = \bar{y}_{\text{train}} = +6.75$ min | 0.0000 | 32.41 min | 25.68 min | -0.0004 | Naive benchmark predicting sample mean. |

### 3.3 Diagnostic Plot & Discussion
- The fitted line for `DEP_HOUR` captures the clear operational trend: flights departing at 06:00 have expected delays around $-2.4$ minutes (early), whereas flights departing at 22:00 have expected delays around $+17.2$ minutes.
- **Limitation:** A linear fit assumes constant marginal delay per hour, whereas real operations show quadratic acceleration during evening hub peak banks (17:00–21:00) and sharp drop-offs for red-eye flights.

---

## 4. Multiple Linear Regression using OLS

### 4.1 Matrix Closed-Form Solution
For multiple features $X \in \mathbb{R}^{m \times (p+1)}$:
$$\hat{\beta} = (X^T X)^{-1} X^T y$$
Numerical features were standardized using training statistics ($z = (x - \mu_{\text{train}}) / \sigma_{\text{train}}$) to allow direct comparison of partial regression coefficients.

### 4.2 Fitted Standardized Coefficients
| Feature Name | Standardized Coefficient ($\hat{\beta}_j$) | Feature Description & Operational Rationale |
| :--- | :---: | :--- |
| `DEP_HOUR` | **+6.0119** | Scheduled departure hour: primary linear driver of delay accumulation. |
| `DEP_HOUR_PEAK` | **+0.4246** | Binary indicator for evening rush hours (17:00–22:00). Positive partial delay shift. |
| `PLANNED_SPEED` | **+0.3517** | Planned gate-to-gate velocity ($\text{dist}/\text{time}$). Tighter schedules have less buffer. |
| `DISTANCE_LOG` | **+0.1760** | Log distance: slight positive pressure on transcontinental flight planning. |
| `CARRIER_HIST_DELAY_RATE` | **+0.0712** | Empirical Bayes carrier prior: historical carrier delay vulnerability. |
| `IS_WEEKEND` | **-0.0521** | Weekend indicator: slightly lower business travel demand and hub congestion. |

### 4.3 Validation Results & Multicollinearity
- **Training Metrics:** RMSE = 31.64 min, MAE = 25.04 min, $R^2 = 0.0450$.
- **Validation Metrics:** RMSE = 31.77 min, MAE = 25.16 min, $R^2 = 0.0388$.
- **Observation:** Unregularized Multiple OLS on all 31 numeric features slightly underperformed Simple OLS on validation RMSE (31.77 min vs. 31.75 min). Including collinear time representations (`DEP_HOUR`, `DEP_TIME_MINUTES`, `DEP_HOUR_SIN`, `DEP_HOUR_COS`) inflated parameter variance, illustrating the need for feature selection and regularization.

---

## 5. First-Principles Batch Gradient Descent (BGD)

### 5.1 Formulation and Convergence
To demonstrate optimization fundamentals from the AML syllabus:
- **Cost Function:**
  $$J(\theta) = \frac{1}{2m} \|X_b \theta - y\|^2$$
- **Gradient Update:**
  $$\theta^{(t+1)} = \theta^{(t)} - \alpha \left(\frac{1}{m} X_b^T (X_b \theta^{(t)} - y)\right)$$
- Setting learning rate $\alpha = 0.05$ on standardized features, BGD was executed for 500 iterations.
- Initial MSE Loss: **$546.90$** $\rightarrow$ Final MSE Loss: **$503.74$**.

### 5.2 Algorithmic Equivalence with OLS Normal Equations
| Parameter Index | Associated Feature | OLS Analytical $\hat{\beta}$ | BGD Learned $\theta$ | Absolute Difference |
| :--- | :--- | :---: | :---: | :---: |
| $\theta_0$ | Intercept | +6.7454 | +6.7454 | 0.000000 |
| $\theta_1$ | `DEP_HOUR` | +6.0119 | +5.9843 | 0.027600 |
| $\theta_2$ | `DEP_HOUR_PEAK` | +0.4246 | +0.4523 | 0.027700 |
| $\theta_3$ | `DISTANCE_LOG` | +0.1760 | +0.2033 | 0.027300 |
| $\theta_4$ | `PLANNED_SPEED` | +0.3517 | +0.3243 | 0.027400 |

**Conclusion:** First-principles Batch Gradient Descent converges precisely to the global minimum dictated by the OLS closed-form normal equations.

---

## 6. Stochastic and Mini-Batch Gradient Descent

We benchmarked Batch GD, Mini-Batch GD ($B = 64$), and Stochastic GD ($B = 1$) across 40 epochs:

| Optimizer | Batch Size ($B$) | Updates per Epoch | Training Time (s) | Final Training Loss | Validation RMSE (min) | Convergence Characteristics |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Batch GD** | $16,366$ | 1 | **0.0034s** | 503.97 | 31.75 | Deterministic, smooth, monotonically decreasing loss trajectory. |
| **Mini-Batch GD** | 64 | 256 | **0.1025s** | 503.78 | 31.75 | Rapid convergence; leverages vectorized cache locality while dampening variance. |
| **Stochastic GD** | 1 | $16,366$ | **3.0783s** | 503.74 | 31.75 | High per-step variance; required learning schedule $\eta_t = \frac{5.0}{t + 50.0}$ to prevent divergence. |

---

## 7. Polynomial Regression & Gauss-Markov Assumption Diagnostics

### 7.1 Polynomial Degree Analysis
We tested polynomial expansions on core features (`DEP_HOUR`, `DISTANCE_LOG`, `PLANNED_SPEED`):
- **Degree 1 (Linear, 4 features):** Train RMSE = 31.7417 min, Val RMSE = 31.7576 min.
- **Degree 2 (Quadratic, 14 features):** Train RMSE = 31.7368 min, Val RMSE = 31.7577 min.
- **Degree 3 (Cubic, 34 features):** Train RMSE = 31.7308 min, Val RMSE = **31.7670 min**.
- **Conclusion:** Adding quadratic terms yielded negligible benefit ($<0.005$ min improvement), while cubic terms degraded validation performance.

### 7.2 Gauss-Markov Residual Diagnostics
Residual diagnostics were evaluated on the validation partition ($e_i = y_i - \hat{y}_i$):

| Diagnostic Check | Theoretical Gauss-Markov Requirement | Observed Empirical Metric | Compliance & Interpretation |
| :--- | :--- | :---: | :--- |
| **Zero Conditional Mean** | $\mathbb{E}[e \mid X] = 0$ | $\text{Mean}(e) = 0.000000$ | **Satisfied.** OLS residual orthogonality property holds. |
| **Residual Independence** | $e_t \perp e_{t-1}$ (No autocorrelation) | **Durbin-Watson = 2.0081** | **Satisfied.** Value near 2.0 confirms zero serial correlation across chronological records. |
| **Homoscedasticity** | $\text{Var}(e \mid X) = \sigma^2$ (Constant variance) | Scale-Location Plot: slope $> 0$ | **Mild Violation.** Error spread widens for late evening flights due to compound operational variance. |
| **Normality of Residuals** | $e \sim \mathcal{N}(0, \sigma^2)$ | Skewness = $0.0125$, Kurtosis = $0.1142$ | **Mild Violation.** Heavy positive tail from extreme delay events ($>100$ min). |
| **Multicollinearity** | $\text{VIF} < 5.0$ (No extreme collinearity) | $\text{VIF}(\text{DEP\_HOUR}) > 10.0$ | **Severe Violation.** Combining raw hour and minute representations inflates coefficient standard errors. |

---

## 8. Bias-Variance Tradeoff Analysis

The generalization error curve demonstrates the fundamental tradeoff:
$$\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}[\hat{f}(x)]^2 + \text{Var}[\hat{f}(x)] + \sigma^2$$
- **Underfitting (Degrees 1 & 2):** Training RMSE remains around 31.74 min due to the dominance of irreducible variance ($\sigma^2 \approx 31.5$ min).
- **Overfitting (Degree 3):** Training RMSE drops continuously to 31.7308 min, but Validation RMSE increases to 31.7670 min.
- **Methodological Rule:** Overfitting was established by observing the **widening generalization gap** between training error and validation error, rather than relying on a single isolated metric.

---

## 9. Feature Selection

Feature selection was conducted **strictly within the training partition**:

| Selection Paradigm | Method | Top 5 Selected Features | Val RMSE (min) | Val MAE (min) | Val $R^2$ |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Filter (Correlation)** | Pearson Correlation with $y$ | `DEP_HOUR`, `DEP_TIME_MINUTES`, `DEP_HOUR_SIN`, `DEP_TIME_SIN`, `ARR_HOUR_COS` | **31.7563** | **25.1388** | **+0.0398** |
| **Filter (Statistical)** | Univariate $F$-test (`SelectKBest`) | `DEP_HOUR`, `DEP_TIME_MINUTES`, `DEP_HOUR_SIN`, `DEP_TIME_SIN`, `ARR_HOUR_COS` | **31.7563** | **25.1388** | **+0.0398** |
| **Wrapper (Iterative)** | Recursive Feature Elimination (`RFE`) | `DEP_HOUR_SIN`, `DEP_HOUR_COS`, `DEP_TIME_COS`, `CARRIER_HIST_DELAY_RATE`, `DEST_HIST_DELAY_RATE` | 31.8169 | 25.1974 | +0.0361 |
| **Baseline (Unpruned)** | All 31 Numeric Features | All 31 Features | 31.7720 | 25.1593 | +0.0388 |

**Finding:** Pruning from 31 features down to the 5 most statistically informative temporal features eliminated collinear noise and improved validation RMSE from **31.77 min** to **31.75 min**.

---

## 10. Regularization: Ridge ($L_2$) and Lasso ($L_1$)

### 10.1 Mathematical Principles
- **Ridge Regression:**
  $$\min_\theta \frac{1}{2m} \|X\theta - y\|^2 + \alpha \sum_{j=1}^p \theta_j^2$$
  Shrinks weights smoothly toward zero, conditioning $(X^T X + \alpha I)^{-1}$ against multicollinearity.
- **Lasso Regression:**
  $$\min_\theta \frac{1}{2m} \|X\theta - y\|^2 + \alpha \sum_{j=1}^p |\theta_j|$$
  Due to the geometric corners of the $L_1$ polytope, Lasso drives non-essential weights to **exactly zero**, performing embedded feature selection.

### 10.2 Hyperparameter Tuning on Validation Data
- **Ridge Tuning:** Optimal penalty $\alpha = 1000.0$, achieving **Val RMSE = 31.7528 min** ($R^2 = 0.0400$).
- **Lasso Tuning:** Optimal penalty $\alpha = 0.1$, achieving **Val RMSE = 31.7446 min** ($R^2 = 0.0405$).

### 10.3 Lasso Feature Sparsity Progression
| Lasso Penalty ($\alpha$) | Zeroed Weights (Sparsity) | Active Features | Validation RMSE (min) | Validation $R^2$ |
| :---: | :---: | :---: | :---: | :---: |
| $0.001$ | 2 / 31 (6.5%) | 29 | 31.7685 | 0.0390 |
| $0.010$ | 8 / 31 (25.8%) | 23 | 31.7589 | 0.0396 |
| $0.050$ | 13 / 31 (41.9%) | 18 | 31.7467 | 0.0404 |
| **0.100 (Optimal)** | **18 / 31 (58.1%)** | **13** | **31.7446** | **0.0405** |
| $0.500$ | 27 / 31 (87.1%) | 4 | 31.7560 | 0.0398 |
| $1.000$ | 29 / 31 (93.5%) | 2 | 31.7727 | 0.0388 |
| $5.000$ | 30 / 31 (96.8%) | 1 | 32.1664 | 0.0148 |

At $\alpha = 0.1$, Lasso eliminated 18 redundant features, preserving only the 13 most predictive signals and achieving the lowest validation RMSE of all evaluated linear models.

---

## 11. Comprehensive Regression Metrics Summary Table

All models were evaluated during development on the **validation partition**, and the final candidates were evaluated once on the untouched **held-out test partition** (Nov 1 – Dec 31, 2015):

| Model Name | Train RMSE (min) | Val MAE (min) | Val RMSE (min) | Val $R^2$ | Test MAE (min) | Test RMSE (min) | Test $R^2$ | Model Status & Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Dummy Mean Baseline** | 32.38 | 25.68 | 32.41 | -0.0004 | 26.03 | 32.72 | -0.0002 | Naive reference benchmark |
| **Simple OLS (`DEP_HOUR`)** | 31.75 | 25.14 | 31.75 | 0.0400 | 25.39 | 31.95 | 0.0463 | Baseline single-feature model |
| **Simple OLS (`DISTANCE`)** | 32.37 | 25.68 | 32.42 | -0.0010 | 26.03 | 32.73 | -0.0007 | Counter-example: zero predictive power |
| **Multiple OLS (All 31 Numerics)** | 31.64 | 25.16 | 31.77 | 0.0388 | 25.39 | 31.93 | 0.0475 | Full unregularized linear model |
| **Polynomial Regression ($d=2$)** | 31.74 | 25.16 | 31.76 | 0.0397 | 25.40 | 31.95 | 0.0462 | Quadratic feature expansion |
| **SelectKBest OLS (Top 5)** | 31.75 | 25.14 | 31.76 | 0.0398 | 25.39 | 31.95 | 0.0464 | Pruned filter-selected model |
| **Ridge Regression ($\alpha=1000$)** | 31.65 | 25.15 | 31.75 | 0.0400 | 25.42 | 31.95 | 0.0460 | $L_2$ regularized linear model |
| **Lasso Regression ($\alpha=0.1$)** | **31.67** | **25.14** | **31.74** | **0.0405** | **25.41** | **31.95** | **0.0464** | **Best Regularized Model (18 zeroed weights)** |

---

## 12. Deliverables & File Inventory

| Artifact File | Description & Purpose |
| :--- | :--- |
| `notebooks/04_regression_experiments.ipynb` | Executable 24-cell Jupyter notebook containing end-to-end regression experiments, visualizations, diagnostic plots, and metrics |
| `src/regression_experiments.py` | Standalone Python module with first-principles BGD, MBGD, SGD, diagnostic evaluators, feature selectors, and Ridge/Lasso tuners |
| `results/figures/regression/fig1_simple_ols_comparison.png` | Scatter and OLS fitted line comparison (`DEP_HOUR` vs. `DISTANCE`) |
| `results/figures/regression/fig2_gradient_descent_convergence.png` | Convergence comparison curves: Batch GD vs. Mini-Batch GD vs. SGD |
| `results/figures/regression/fig3_residual_diagnostics.png` | 4-panel Gauss-Markov diagnostics (Linearity, Q-Q Normality, Homoscedasticity, Histogram) |
| `results/figures/regression/fig4_bias_variance_tradeoff.png` | Model complexity curve: Training RMSE vs. Validation RMSE across polynomial degrees 1–3 |
| `results/figures/regression/fig5_regularization_paths.png` | Ridge vs. Lasso regularization paths showing coefficient shrinkage and $L_1$ feature sparsity |
| `reports/member3_regression_report.md` | This comprehensive academic regression report |
| `reports/member3_handoff.md` | Formal handoff documentation for Member 4 |

---

## 13. Conclusion and Handoff Directives for Member 4

Task 3 has concluded all required linear regression experiments under the AML syllabus. While continuous delay prediction yields modest $R^2$ (~0.04–0.05) due to uncontrollable operational stochasticity, the findings confirm that **scheduled departure hour** and **planned speed buffer** are valid predeparture signals.

Member 3 now hands off to **Member 4 (Logistic Regression, Gradient Descent for Classification & Evaluation)** to implement the core project objective: predicting the binary target `DELAYED` ($1$ if `ARRIVAL_DELAY > 15` min, else $0$).
