"""
Regression Experiments Module
Project 21: Flight Delay Prediction and Rescheduling — AML Phase 1
Author: Member 3 (Linear Regression, OLS, Gradient Descent & Regularization)

This module implements:
1. Analytical Ordinary Least Squares (OLS) closed-form solutions:
   beta_hat = (X^T X)^{-1} X^T y
2. First-principles Gradient Descent Optimizers:
   - Batch Gradient Descent (BGD) with convergence tracking
   - Mini-Batch Gradient Descent (MBGD) with epoch shuffling
   - Stochastic Gradient Descent (SGD) with inverse-scaling learning schedule
3. Regression Evaluation Metrics:
   - Mean Absolute Error (MAE)
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)
   - Coefficient of Determination (R^2)
4. Linear Regression Assumption Diagnostics:
   - Durbin-Watson statistic for residual autocorrelation
   - Residual distribution moments (skewness and kurtosis)
   - Variance Inflation Factor (VIF) for multicollinearity
5. Feature Selection Procedures (Training Data Only):
   - Univariate Correlation Screening
   - Statistical F-Regression (SelectKBest)
   - Recursive Feature Elimination (RFE)
6. Regularization Frameworks:
   - Ridge (L2) Regression hyperparameter tuning
   - Lasso (L1) Regression path & sparsity analysis
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src.config import (
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
)
from src.features import (
    FEATURE_COLUMNS_ALL,
    FEATURE_COLUMNS_NUMERIC,
    TARGET_COLUMNS,
)

REGRESSION_FIGURES_DIR = FIGURES_DIR / "regression"
REGRESSION_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 1. Evaluation Metrics
# ==============================================================================

def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute standard regression evaluation metrics for arrival delay predictions.

    Parameters
    ----------
    y_true : np.ndarray
        Observed ground-truth arrival delay minutes.
    y_pred : np.ndarray
        Model predicted arrival delay minutes.

    Returns
    -------
    Dict[str, float]
        Dictionary containing MAE, MSE, RMSE, and R2.
    """
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()
    
    mae = float(mean_absolute_error(y_t, y_p))
    mse = float(mean_squared_error(y_t, y_p))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_t, y_p))
    
    return {
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 6),
    }


# ==============================================================================
# 2. First-Principles Gradient Descent Regressors
# ==============================================================================

class BatchGradientDescentRegressor:
    """
    Linear Regression fit using Batch Gradient Descent (BGD) from first principles.
    
    Objective Function:
      J(theta) = (1 / 2m) * ||X_b @ theta - y||^2
    Gradient:
      grad = (1 / m) * X_b^T @ (X_b @ theta - y)
    Update Rule:
      theta <- theta - alpha * grad
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.theta_: Optional[np.ndarray] = None
        self.loss_history_: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BatchGradientDescentRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        m, n = X.shape

        # Prepend bias column (ones)
        X_b = np.c_[np.ones((m, 1)), X]
        self.theta_ = np.zeros(n + 1, dtype=float)
        self.loss_history_ = []

        for i in range(self.max_iter):
            pred = X_b @ self.theta_
            error = pred - y
            loss = (1.0 / (2.0 * m)) * np.sum(error ** 2)
            self.loss_history_.append(loss)

            grad = (1.0 / m) * (X_b.T @ error)
            grad_norm = np.linalg.norm(grad)

            if grad_norm < self.tol:
                break

            self.theta_ -= self.learning_rate * grad

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.theta_ is None:
            raise RuntimeError("Model must be fitted before calling predict().")
        X = np.asarray(X, dtype=float)
        X_b = np.c_[np.ones((len(X), 1)), X]
        return X_b @ self.theta_

    @property
    def intercept_(self) -> float:
        return float(self.theta_[0])

    @property
    def coef_(self) -> np.ndarray:
        return self.theta_[1:]


class MiniBatchGradientDescentRegressor:
    """
    Linear Regression fit using Mini-Batch Gradient Descent (MBGD).
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        batch_size: int = 64,
        epochs: int = 50,
        random_state: int = RANDOM_STATE,
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state
        self.theta_: Optional[np.ndarray] = None
        self.loss_history_: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MiniBatchGradientDescentRegressor":
        np.random.seed(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        m, n = X.shape

        X_b = np.c_[np.ones((m, 1)), X]
        self.theta_ = np.zeros(n + 1, dtype=float)
        self.loss_history_ = []

        n_batches = int(np.ceil(m / self.batch_size))

        for epoch in range(self.epochs):
            indices = np.random.permutation(m)
            X_shuff = X_b[indices]
            y_shuff = y[indices]

            for b in range(n_batches):
                start = b * self.batch_size
                end = min(start + self.batch_size, m)
                X_batch = X_shuff[start:end]
                y_batch = y_shuff[start:end]
                k = len(y_batch)

                pred_b = X_batch @ self.theta_
                grad_b = (1.0 / k) * (X_batch.T @ (pred_b - y_batch))
                self.theta_ -= self.learning_rate * grad_b

            # Compute full training loss at end of epoch
            full_loss = (1.0 / (2.0 * m)) * np.sum((X_b @ self.theta_ - y) ** 2)
            self.loss_history_.append(full_loss)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.theta_ is None:
            raise RuntimeError("Model must be fitted before calling predict().")
        X = np.asarray(X, dtype=float)
        X_b = np.c_[np.ones((len(X), 1)), X]
        return X_b @ self.theta_

    @property
    def intercept_(self) -> float:
        return float(self.theta_[0])

    @property
    def coef_(self) -> np.ndarray:
        return self.theta_[1:]


class StochasticGradientDescentRegressor:
    """
    Linear Regression fit using Stochastic Gradient Descent (SGD) with single-sample updates
    and an inverse-scaling learning rate schedule.
    """

    def __init__(
        self,
        t0: float = 5.0,
        t1: float = 50.0,
        epochs: int = 50,
        random_state: int = RANDOM_STATE,
    ):
        self.t0 = t0
        self.t1 = t1
        self.epochs = epochs
        self.random_state = random_state
        self.theta_: Optional[np.ndarray] = None
        self.loss_history_: List[float] = []

    def _learning_schedule(self, t: int) -> float:
        return self.t0 / (t + self.t1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StochasticGradientDescentRegressor":
        np.random.seed(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        m, n = X.shape

        X_b = np.c_[np.ones((m, 1)), X]
        self.theta_ = np.zeros(n + 1, dtype=float)
        self.loss_history_ = []

        step = 0
        for epoch in range(self.epochs):
            indices = np.random.permutation(m)
            for idx in indices:
                x_i = X_b[idx:idx + 1]
                y_i = y[idx:idx + 1]
                grad_i = x_i.T @ (x_i @ self.theta_ - y_i)
                eta = self._learning_schedule(step)
                self.theta_ -= eta * grad_i.ravel()
                step += 1

            full_loss = (1.0 / (2.0 * m)) * np.sum((X_b @ self.theta_ - y) ** 2)
            self.loss_history_.append(full_loss)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.theta_ is None:
            raise RuntimeError("Model must be fitted before calling predict().")
        X = np.asarray(X, dtype=float)
        X_b = np.c_[np.ones((len(X), 1)), X]
        return X_b @ self.theta_

    @property
    def intercept_(self) -> float:
        return float(self.theta_[0])

    @property
    def coef_(self) -> np.ndarray:
        return self.theta_[1:]


# ==============================================================================
# 3. Linear Regression Assumption Diagnostics
# ==============================================================================

def compute_durbin_watson(residuals: np.ndarray) -> float:
    """
    Calculate the Durbin-Watson statistic to test for 1st-order serial autocorrelation:
      DW = sum_{t=2}^N (e_t - e_{t-1})^2 / sum_{t=1}^N e_t^2
    Values near 2 indicate zero serial correlation.
    """
    e = np.asarray(residuals, dtype=float).ravel()
    diff = np.diff(e)
    denom = np.sum(e ** 2)
    if denom == 0:
        return 2.0
    return float(np.sum(diff ** 2) / denom)


def compute_vif_dataframe(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor (VIF) for continuous features to diagnose multicollinearity:
      VIF_j = 1 / (1 - R_j^2)
    VIF > 5-10 indicates problematic collinearity.
    """
    from sklearn.linear_model import LinearRegression

    cols = list(df_features.columns)
    vif_records = []

    for target_col in cols:
        other_cols = [c for c in cols if c != target_col]
        X_other = df_features[other_cols].values
        y_target = df_features[target_col].values

        lr = LinearRegression().fit(X_other, y_target)
        r2_val = r2_score(y_target, lr.predict(X_other))
        vif = 1.0 / (1.0 - r2_val) if r2_val < 0.9999 else 9999.0

        vif_records.append({
            "Feature": target_col,
            "R2_from_others": round(r2_val, 4),
            "VIF": round(vif, 2),
        })

    return pd.DataFrame(vif_records).sort_values(by="VIF", ascending=False).reset_index(drop=True)


def evaluate_linear_assumptions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Assess classical Gauss-Markov linear regression assumptions on model residuals.
    """
    e = np.asarray(y_true).ravel() - np.asarray(y_pred).ravel()
    dw_stat = compute_durbin_watson(e)
    sk = float(skew(e))
    kt = float(kurtosis(e))
    mean_e = float(np.mean(e))
    std_e = float(np.std(e))

    diagnostics = {
        "mean_residual": round(mean_e, 6),
        "std_residual": round(std_e, 4),
        "durbin_watson": round(dw_stat, 4),
        "skewness": round(sk, 4),
        "kurtosis": round(kt, 4),
    }

    if feature_df is not None:
        vif_df = compute_vif_dataframe(feature_df)
        diagnostics["vif_summary"] = vif_df.to_dict(orient="records")

    return diagnostics


# ==============================================================================
# 4. Feature Selection Procedures
# ==============================================================================

def select_features_correlation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_k: int = 5,
) -> List[str]:
    """Filter method: select top_k features with highest absolute Pearson correlation."""
    corrs = X_train.apply(lambda col: np.abs(col.corr(y_train))).sort_values(ascending=False)
    return corrs.head(top_k).index.tolist()


def select_features_f_test(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_k: int = 5,
) -> List[str]:
    """Filter method: select top_k features based on univariate F-regression p-values."""
    selector = SelectKBest(score_func=f_regression, k=top_k)
    selector.fit(X_train, y_train)
    return [X_train.columns[i] for i in selector.get_support(indices=True)]


def select_features_rfe(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_features: int = 5,
) -> List[str]:
    """Wrapper method: Recursive Feature Elimination using an OLS linear estimator."""
    estimator = LinearRegression()
    rfe = RFE(estimator=estimator, n_features_to_select=n_features, step=1)
    rfe.fit(X_train, y_train)
    return [X_train.columns[i] for i in rfe.get_support(indices=True)]


# ==============================================================================
# 5. Regularization Tuning
# ==============================================================================

def tune_ridge_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    alphas: Optional[List[float]] = None,
) -> Tuple[float, float, pd.DataFrame]:
    """Tune Ridge L2 penalty across an alpha grid using validation RMSE."""
    if alphas is None:
        alphas = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

    records = []
    best_alpha = alphas[0]
    best_rmse = float("inf")

    for a in alphas:
        model = Ridge(alpha=a).fit(X_train, y_train)
        pred_va = model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, pred_va))
        val_mae = mean_absolute_error(y_val, pred_va)
        val_r2 = r2_score(y_val, pred_va)

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_alpha = a

        records.append({
            "alpha": a,
            "val_rmse": round(val_rmse, 4),
            "val_mae": round(val_mae, 4),
            "val_r2": round(val_r2, 6),
            "l2_norm_weights": round(float(np.linalg.norm(model.coef_)), 4),
        })

    return best_alpha, best_rmse, pd.DataFrame(records)


def tune_lasso_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    alphas: Optional[List[float]] = None,
) -> Tuple[float, float, pd.DataFrame]:
    """Tune Lasso L1 penalty across an alpha grid and track sparsity."""
    if alphas is None:
        alphas = [1e-4, 1e-3, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    records = []
    best_alpha = alphas[0]
    best_rmse = float("inf")
    n_features = X_train.shape[1]

    import warnings
    from sklearn.exceptions import ConvergenceWarning

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for a in alphas:
            model = Lasso(alpha=a, max_iter=5000, tol=1e-3, random_state=RANDOM_STATE).fit(X_train, y_train)
            pred_va = model.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, pred_va))
            val_mae = mean_absolute_error(y_val, pred_va)
            val_r2 = r2_score(y_val, pred_va)
            zero_count = int(np.sum(model.coef_ == 0.0))

            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_alpha = a

            records.append({
                "alpha": a,
                "val_rmse": round(val_rmse, 4),
                "val_mae": round(val_mae, 4),
                "val_r2": round(val_r2, 6),
                "zero_weights": zero_count,
                "active_weights": n_features - zero_count,
                "l1_norm_weights": round(float(np.sum(np.abs(model.coef_))), 4),
            })

    return best_alpha, best_rmse, pd.DataFrame(records)


# ==============================================================================
# 6. Command-Line Demonstration Runner
# ==============================================================================

def main():
    print("=" * 78)
    print("Flight Delay Prediction — Task 3: Regression Experiments Demonstration")
    print("=" * 78)

    features_path = PROCESSED_DATA_DIR / "flights_features.csv"
    if not features_path.exists():
        raise FileNotFoundError(f"Prepared features dataset missing at: {features_path}")

    print(f"Loading prepared features from: {features_path}")
    df = pd.read_csv(features_path)
    print(f"Loaded {len(df):,} records across {len(df.columns)} columns.")

    # Chronological partition extraction
    train = df[df["SPLIT_SET"] == "train"].copy()
    val = df[df["SPLIT_SET"] == "val"].copy()
    test = df[df["SPLIT_SET"] == "test"].copy()

    print(f"• Training Set:   {len(train):,} flights (Jan 1 – Aug 31, 2015)")
    print(f"• Validation Set: {len(val):,} flights (Sep 1 – Oct 31, 2015)")
    print(f"• Test Set:       {len(test):,} flights (Nov 1 – Dec 31, 2015)")

    target_col = "ARRIVAL_DELAY"
    y_train = train[target_col].values
    y_val = val[target_col].values
    y_test = test[target_col].values

    # 1. Simple OLS on DEP_HOUR
    print("\n--- 1. Simple Linear Regression (DEP_HOUR) ---")
    lr_simple = LinearRegression().fit(train[["DEP_HOUR"]], y_train)
    val_metrics_simple = compute_regression_metrics(y_val, lr_simple.predict(val[["DEP_HOUR"]]))
    print(f"Intercept: {lr_simple.intercept_:.4f} min | Slope: {lr_simple.coef_[0]:.4f} min/hour")
    print(f"Validation Metrics: {val_metrics_simple}")

    # 2. Multiple OLS on numeric features
    print("\n--- 2. Multiple Linear Regression (All Numerics) ---")
    scaler = StandardScaler()
    X_tr_num = scaler.fit_transform(train[FEATURE_COLUMNS_NUMERIC])
    X_va_num = scaler.transform(val[FEATURE_COLUMNS_NUMERIC])
    X_te_num = scaler.transform(test[FEATURE_COLUMNS_NUMERIC])

    lr_mult = LinearRegression().fit(X_tr_num, y_train)
    val_metrics_mult = compute_regression_metrics(y_val, lr_mult.predict(X_va_num))
    print(f"Validation Metrics: {val_metrics_mult}")

    # 3. Batch Gradient Descent
    print("\n--- 3. Batch Gradient Descent ---")
    bgd = BatchGradientDescentRegressor(learning_rate=0.05, max_iter=500)
    bgd.fit(X_tr_num, y_train)
    val_metrics_bgd = compute_regression_metrics(y_val, bgd.predict(X_va_num))
    print(f"BGD Final Loss: {bgd.loss_history_[-1]:.4f} | Validation Metrics: {val_metrics_bgd}")

    # 4. Regularization: Ridge & Lasso
    print("\n--- 4. Regularization Tuning ---")
    best_ridge_a, best_ridge_rmse, _ = tune_ridge_regression(X_tr_num, y_train, X_va_num, y_val)
    print(f"Best Ridge alpha: {best_ridge_a} (Val RMSE: {best_ridge_rmse:.4f})")

    best_lasso_a, best_lasso_rmse, _ = tune_lasso_regression(X_tr_num, y_train, X_va_num, y_val)
    print(f"Best Lasso alpha: {best_lasso_a} (Val RMSE: {best_lasso_rmse:.4f})")

    print("\n✓ Regression experiments demonstration executed successfully!")
    print("=" * 78)


if __name__ == "__main__":
    main()
