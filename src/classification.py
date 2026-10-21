"""
Classification Pipeline and Modeling Module
Project 21: Flight Delay Prediction and Rescheduling — AML Phase 1
Author: Member 4 (Classification, Logistic Regression, Gradient Descent, XGBoost & Evaluation)

This module implements:
1. Leakage-free classification preprocessing pipeline (StandardScaler for numerics, OneHotEncoder for categoricals).
2. Educational first-principles Gradient Descent for Logistic Regression (Binary Cross-Entropy / MLE).
3. Production Logistic Regression baseline (MLE, L1/L2 Regularization, Log-Odds & Odds Ratios).
4. XGBoost Classifier ensemble benchmark with validation tuning.
5. Rigorous classification metrics (Confusion Matrix, Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Log-Loss, Brier).
6. Operational probability threshold tuning and probability calibration diagnostics.
7. Multiclass extension for delay severity categories (On-time, Moderate 16-60m, Severe >60m).
8. Validated model serialization and artifact integrity checks.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from src.config import (
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
)
from src.features import (
    FEATURE_COLUMNS_CATEGORICAL,
    FEATURE_COLUMNS_NUMERIC,
    IDENTIFIER_COLUMNS,
    PROHIBITED_LEAKAGE_COLUMNS,
)

# Output directory for classification figures
CLASSIFICATION_FIGS_DIR = FIGURES_DIR / "classification"
CLASSIFICATION_FIGS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Canonical candidate feature list (36 pre-departure features)
ALL_CLASSIFICATION_FEATURES = FEATURE_COLUMNS_NUMERIC + FEATURE_COLUMNS_CATEGORICAL

# Multiclass definitions (Optional Extension)
MULTICLASS_EDGES = [-np.inf, 15.0, 60.0, np.inf]
MULTICLASS_LABELS = [0, 1, 2]
MULTICLASS_CLASS_NAMES = [
    "On-Time / Minor (<=15 min)",
    "Moderate Delay (16-60 min)",
    "Severe Delay (>60 min)",
]


# ==============================================================================
# 1. Pipeline Builder & Data Preparation
# ==============================================================================

def build_classification_pipeline(
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> ColumnTransformer:
    """
    Construct a reproducible Scikit-Learn ColumnTransformer for classification.
    
    Transforms:
      - Numeric features: Median Imputation -> StandardScaler.
      - Categorical features: Constant 'UNKNOWN' Imputation -> OneHotEncoder(handle_unknown='ignore').
      
    Guarantees:
      - Remainder columns (identifiers, leakage, targets) are dropped.
      - Fit strictly on training partitions; safe forward transform on val/test.
    """
    if numeric_features is None:
        numeric_features = FEATURE_COLUMNS_NUMERIC
    if categorical_features is None:
        categorical_features = FEATURE_COLUMNS_CATEGORICAL

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def load_and_partition_classification_data(
    features_csv_path: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Load data/processed/flights_features.csv, validate anti-leakage barriers,
    and extract non-overlapping chronological partitions.
    """
    if features_csv_path is None:
        features_csv_path = PROCESSED_DATA_DIR / "flights_features.csv"
    features_csv_path = Path(features_csv_path)

    if not features_csv_path.exists():
        raise FileNotFoundError(f"Prepared feature dataset missing: {features_csv_path}")

    df = pd.read_csv(features_csv_path)

    # Validate target columns
    assert "DELAYED" in df.columns, "Classification target 'DELAYED' missing from dataset."
    assert "SPLIT_SET" in df.columns, "Partition indicator 'SPLIT_SET' missing from dataset."

    train_df = df[df["SPLIT_SET"] == "train"].copy().reset_index(drop=True)
    val_df = df[df["SPLIT_SET"] == "val"].copy().reset_index(drop=True)
    test_df = df[df["SPLIT_SET"] == "test"].copy().reset_index(drop=True)

    metadata = {
        "total_rows": len(df),
        "total_cols": len(df.columns),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_delayed_rate": float(train_df["DELAYED"].mean()),
        "val_delayed_rate": float(val_df["DELAYED"].mean()),
        "test_delayed_rate": float(test_df["DELAYED"].mean()),
    }
    return train_df, val_df, test_df, metadata


def extract_feature_matrices(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Extract strictly pre-departure feature matrices X and binary targets y.
    Ensures ARRIVAL_DELAY, DELAYED, and identifiers are never present in X.
    """
    if feature_cols is None:
        feature_cols = ALL_CLASSIFICATION_FEATURES

    for col in feature_cols:
        assert col not in PROHIBITED_LEAKAGE_COLUMNS, f"Data leakage violation! Prohibited column in X: {col}"
        assert col not in IDENTIFIER_COLUMNS, f"Identifier violation! Column in X: {col}"

    X_train = train_df[feature_cols].copy()
    y_train = train_df["DELAYED"].values.astype(int)

    X_val = val_df[feature_cols].copy()
    y_val = val_df["DELAYED"].values.astype(int)

    X_test = test_df[feature_cols].copy()
    y_test = test_df["DELAYED"].values.astype(int)

    return X_train, y_train, X_val, y_val, X_test, y_test


# ==============================================================================
# 2. Educational First-Principles Gradient Descent for Logistic Regression
# ==============================================================================

class LogisticRegressionGD:
    """
    Educational First-Principles Logistic Regression using Gradient Descent.
    
    Demonstrates:
      - Sigmoid hypothesis function: h_theta(x) = 1 / (1 + exp(-theta^T x))
      - Binary Cross-Entropy (Log-Loss) objective function:
          J(theta) = -1/m * sum [ y*log(h) + (1-y)*log(1-h) ] + lambda/(2m) * ||theta_1:d||^2
      - Analytical gradient vector:
          grad_theta = 1/m * X^T (h_theta(X) - y) + lambda/m * theta (unregularized bias)
      - Supports Batch Gradient Descent (BGD) and Mini-Batch Gradient Descent (MBGD).
      
    NOTE: This is strictly an educational implementation for syllabus demonstration.
    Final production inference pipelines utilize Scikit-Learn's optimized L-BFGS solver.
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        n_epochs: int = 300,
        batch_size: Optional[int] = None,
        l2_reg: float = 0.01,
        fit_intercept: bool = True,
        random_state: int = RANDOM_STATE,
    ):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.l2_reg = l2_reg
        self.fit_intercept = fit_intercept
        self.random_state = random_state

        self.theta_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self.coef_: Optional[np.ndarray] = None
        self.loss_history_: List[float] = []
        self.val_loss_history_: List[float] = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid function."""
        z_clipped = np.clip(z, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def _compute_loss(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
        """Compute binary cross-entropy loss with L2 regularization."""
        m = len(y)
        probs = self._sigmoid(X @ theta)
        eps = 1e-15
        probs_clipped = np.clip(probs, eps, 1.0 - eps)

        bce = -np.mean(y * np.log(probs_clipped) + (1.0 - y) * np.log(1.0 - probs_clipped))
        if self.l2_reg > 0:
            reg_theta = theta[1:] if self.fit_intercept else theta
            reg_term = (self.l2_reg / (2.0 * m)) * np.sum(reg_theta ** 2)
            return float(bce + reg_term)
        return float(bce)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "LogisticRegressionGD":
        """
        Fit Logistic Regression via Gradient Descent.
        """
        rng = np.random.default_rng(self.random_state)
        m, n = X.shape

        if self.fit_intercept:
            X_b = np.c_[np.ones((m, 1)), X]
            n_features = n + 1
        else:
            X_b = X
            n_features = n

        if X_val is not None:
            if self.fit_intercept:
                X_val_b = np.c_[np.ones((len(X_val), 1)), X_val]
            else:
                X_val_b = X_val

        # Initialize weights with small random Gaussian values
        self.theta_ = rng.normal(loc=0.0, scale=0.01, size=n_features)
        self.loss_history_ = []
        self.val_loss_history_ = []

        batch_sz = m if (self.batch_size is None or self.batch_size >= m) else self.batch_size

        for epoch in range(self.n_epochs):
            indices = np.arange(m)
            rng.shuffle(indices)
            X_shuffled = X_b[indices]
            y_shuffled = y[indices]

            for start_idx in range(0, m, batch_sz):
                end_idx = min(start_idx + batch_sz, m)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                batch_m = len(y_batch)

                probs = self._sigmoid(X_batch @ self.theta_)
                errors = probs - y_batch
                grad = (1.0 / batch_m) * (X_batch.T @ errors)

                if self.l2_reg > 0:
                    reg_grad = np.copy(self.theta_)
                    if self.fit_intercept:
                        reg_grad[0] = 0.0  # Do not regularize bias
                    grad += (self.l2_reg / batch_m) * reg_grad

                self.theta_ -= self.learning_rate * grad

            epoch_loss = self._compute_loss(X_b, y, self.theta_)
            self.loss_history_.append(epoch_loss)

            if X_val is not None and y_val is not None:
                val_loss = self._compute_loss(X_val_b, y_val, self.theta_)
                self.val_loss_history_.append(val_loss)

        if self.fit_intercept:
            self.intercept_ = float(self.theta_[0])
            self.coef_ = self.theta_[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = self.theta_

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities: [P(y=0), P(y=1)]."""
        if self.theta_ is None:
            raise ValueError("Model is not fitted yet.")
        m = len(X)
        if self.fit_intercept:
            X_b = np.c_[np.ones((m, 1)), X]
        else:
            X_b = X
        p1 = self._sigmoid(X_b @ self.theta_)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels based on decision threshold."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


# ==============================================================================
# 3. Model Training & Evaluation Suite
# ==============================================================================

def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics for imbalanced evaluation.
    
    Returns:
      - Threshold, Accuracy, Precision, Recall, F1-Score
      - ROC-AUC, PR-AUC (Average Precision)
      - Log Loss (Binary Cross-Entropy), Brier Score
      - Confusion Matrix: TN, FP, FN, TP
    """
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Guard against zero divisions safely
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = float("nan")

    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = float("nan")

    try:
        bce = log_loss(y_true, y_prob, labels=[0, 1])
    except Exception:
        bce = float("nan")

    brier = brier_score_loss(y_true, y_prob)

    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "log_loss": float(bce),
        "brier_score": float(brier),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "support_total": int(len(y_true)),
        "support_positive": int(np.sum(y_true == 1)),
        "support_negative": int(np.sum(y_true == 0)),
        "positive_rate": float(np.mean(y_true)),
    }


def evaluate_threshold_grid(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    fn_cost_ratio: float = 2.5,
) -> pd.DataFrame:
    """
    Evaluate multiple probability thresholds on validation data.
    
    Includes an Operational Airline Utility Metric:
      In airline dispatch, a False Negative (unanticipated delayed flight) incurs higher operational costs
      (crew timeouts, stranded passengers, gate holds) than a False Positive (precautionary buffer alert).
      Custom Cost = (fn_cost_ratio * FN) + (1.0 * FP)
      Utility = 1 - (Custom Cost / Worst Case Cost)
    """
    if thresholds is None:
        thresholds = np.linspace(0.10, 0.90, 33)

    records = []
    total_pos = np.sum(y_true == 1)
    total_neg = np.sum(y_true == 0)
    worst_cost = (fn_cost_ratio * total_pos) + (1.0 * total_neg)

    for thresh in thresholds:
        metrics = compute_classification_metrics(y_true, y_prob, threshold=thresh)
        fn = metrics["confusion_matrix"]["fn"]
        fp = metrics["confusion_matrix"]["fp"]
        tp = metrics["confusion_matrix"]["tp"]
        tn = metrics["confusion_matrix"]["tn"]

        cost = (fn_cost_ratio * fn) + (1.0 * fp)
        utility = 1.0 - (cost / worst_cost) if worst_cost > 0 else 0.0

        records.append({
            "threshold": thresh,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "operational_cost": cost,
            "operational_utility": utility,
        })

    return pd.DataFrame(records)


def train_logistic_regression_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
    solver: str = "lbfgs",
    max_iter: int = 1000,
    random_state: int = RANDOM_STATE,
) -> LogisticRegression:
    """Train production-grade Scikit-Learn Logistic Regression baseline."""
    clf = LogisticRegression(
        C=C,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)
    return clf


def train_xgboost_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 150,
    max_depth: int = 4,
    learning_rate: float = 0.08,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = RANDOM_STATE,
) -> Any:
    """Train XGBoost gradient boosting classifier if available."""
    if not HAS_XGBOOST:
        raise RuntimeError("XGBoost is not installed or libomp runtime is unavailable.")

    clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


# ==============================================================================
# 4. Multiclass Extension Module (Optional Syllabus Extension)
# ==============================================================================

def create_multiclass_targets(
    df: pd.DataFrame,
    delay_col: str = "ARRIVAL_DELAY",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Construct 3-tier delay severity target:
      - Class 0: On-Time / Minor (<= 15 minutes)
      - Class 1: Moderate Delay (16 to 60 minutes)
      - Class 2: Severe Delay (> 60 minutes)
    """
    delays = df[delay_col].values
    y_multi = np.zeros(len(delays), dtype=int)
    y_multi[(delays > 15.0) & (delays <= 60.0)] = 1
    y_multi[delays > 60.0] = 2

    counts = pd.Series(y_multi).value_counts().sort_index().to_dict()
    proportions = (pd.Series(y_multi).value_counts(normalize=True).sort_index()).to_dict()

    metadata = {
        "class_counts": {MULTICLASS_CLASS_NAMES[k]: int(v) for k, v in counts.items()},
        "class_proportions": {MULTICLASS_CLASS_NAMES[k]: float(v) for k, v in proportions.items()},
    }
    return y_multi, metadata


def train_multiclass_logistic(
    X_train: np.ndarray,
    y_train_multi: np.ndarray,
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = RANDOM_STATE,
) -> LogisticRegression:
    """Train Multinomial Logistic Regression model for 3-tier delay classification."""
    clf = LogisticRegression(
        solver="lbfgs",
        C=C,
        max_iter=max_iter,
        random_state=random_state,
    )
    clf.fit(X_train, y_train_multi)
    return clf


def compute_multiclass_metrics(
    y_true_multi: np.ndarray,
    y_prob_multi: np.ndarray,
) -> Dict[str, Any]:
    """Compute overall accuracy, macro-F1, per-class F1, and log loss."""
    y_pred = np.argmax(y_prob_multi, axis=1)
    acc = accuracy_score(y_true_multi, y_pred)
    macro_f1 = f1_score(y_true_multi, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true_multi, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true_multi, y_pred, labels=[0, 1, 2])
    m_log_loss = log_loss(y_true_multi, y_prob_multi, labels=[0, 1, 2])

    report = classification_report(
        y_true_multi,
        y_pred,
        target_names=MULTICLASS_CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "multiclass_log_loss": float(m_log_loss),
        "confusion_matrix": cm.tolist(),
        "per_class_report": report,
    }


# ==============================================================================
# 5. Visualization Generation Suite
# ==============================================================================

def generate_classification_plots(
    y_val: np.ndarray,
    val_probs_dict: Dict[str, np.ndarray],
    threshold_df: pd.DataFrame,
    optimal_threshold: float,
    gd_model: Optional[LogisticRegressionGD] = None,
    lr_model: Optional[LogisticRegression] = None,
    xgb_model: Optional[Any] = None,
    feature_names: Optional[List[str]] = None,
    y_val_multi: Optional[np.ndarray] = None,
    multi_probs: Optional[np.ndarray] = None,
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """
    Generate and save publication-grade classification figures to results/figures/classification/.
    """
    if output_dir is None:
        output_dir = CLASSIFICATION_FIGS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_plots = []

    sns.set_theme(style="whitegrid", font="sans-serif")
    palette = sns.color_palette("deep")

    # --------------------------------------------------------------------------
    # Figure 1: ROC Curves Comparison
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, probs in val_probs_dict.items():
        fpr, tpr, _ = roc_curve(y_val, probs)
        score = roc_auc_score(y_val, probs)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (ROC-AUC = {score:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.5000)")
    ax.set_title("Validation ROC Curves Comparison — Flight Delay Classification", fontsize=13, weight="bold")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    fig_path = output_dir / "fig1_roc_curves.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    generated_plots.append(fig_path)

    # --------------------------------------------------------------------------
    # Figure 2: Precision-Recall Curves Comparison
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    base_pos_rate = float(np.mean(y_val))
    ax.axhline(base_pos_rate, color="gray", linestyle="--", lw=1.5, label=f"No-Skill Baseline ({base_pos_rate:.1%})")

    for name, probs in val_probs_dict.items():
        prec, rec, _ = precision_recall_curve(y_val, probs)
        score = average_precision_score(y_val, probs)
        ax.plot(rec, prec, lw=2, label=f"{name} (PR-AUC = {score:.4f})")

    ax.set_title("Validation Precision-Recall Curves (Class Imbalance Evaluation)", fontsize=13, weight="bold")
    ax.set_xlabel("Recall (Fraction of Actual Delays Caught)", fontsize=11)
    ax.set_ylabel("Precision (Fraction of Delay Alerts Correct)", fontsize=11)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([0.20, 0.85])
    fig_path = output_dir / "fig2_precision_recall_curves.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    generated_plots.append(fig_path)

    # --------------------------------------------------------------------------
    # Figure 3: Operating Threshold Optimization Curves
    # --------------------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.plot(threshold_df["threshold"], threshold_df["accuracy"], label="Accuracy", color="gray", lw=1.5, linestyle=":")
    ax1.plot(threshold_df["threshold"], threshold_df["precision"], label="Precision", color=palette[0], lw=2)
    ax1.plot(threshold_df["threshold"], threshold_df["recall"], label="Recall", color=palette[2], lw=2)
    ax1.plot(threshold_df["threshold"], threshold_df["f1"], label="F1-Score", color=palette[3], lw=2.5)

    ax1.axvline(0.5, color="black", linestyle="--", alpha=0.7, label="Default Threshold (0.50)")
    ax1.axvline(optimal_threshold, color="crimson", linestyle="-", lw=2, label=f"Selected Threshold ({optimal_threshold:.2f})")
    ax1.set_title("Probability Decision Threshold Optimization (Validation Set)", fontsize=13, weight="bold")
    ax1.set_xlabel("Decision Threshold (tau)", fontsize=11)
    ax1.set_ylabel("Metric Value", fontsize=11)
    ax1.set_xlim([0.15, 0.85])
    ax1.set_ylim([0.0, 1.0])
    ax1.legend(loc="center right", frameon=True, fontsize=10)
    fig_path = output_dir / "fig3_threshold_tradeoffs.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    generated_plots.append(fig_path)

    # --------------------------------------------------------------------------
    # Figure 4: Confusion Matrix Comparison (tau = 0.50 vs tau = optimal)
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    selected_probs = val_probs_dict.get("Logistic Regression (MLE)", list(val_probs_dict.values())[0])

    for idx, (thresh, title) in enumerate([
        (0.50, "Default Operating Threshold (tau = 0.50)"),
        (optimal_threshold, f"Optimized Operating Threshold (tau = {optimal_threshold:.2f})"),
    ]):
        y_pred = (selected_probs >= thresh).astype(int)
        cm = confusion_matrix(y_val, y_pred)
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[idx],
            cbar=False,
            annot_kws={"size": 13, "weight": "bold"},
            xticklabels=["Predicted On-Time", "Predicted Delayed"],
            yticklabels=["Actual On-Time", "Actual Delayed"],
        )
        # Overlay normalized percentages
        for i in range(2):
            for j in range(2):
                axes[idx].text(
                    j + 0.5,
                    i + 0.72,
                    f"({cm_norm[i, j]:.1%})",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="gray" if cm[i, j] < (cm.max() / 2) else "lightyellow",
                )
        axes[idx].set_title(title, fontsize=11, weight="bold")

    fig.suptitle("Logistic Regression Validation Confusion Matrices", fontsize=13, weight="bold", y=1.02)
    fig_path = output_dir / "fig4_confusion_matrices.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    generated_plots.append(fig_path)

    # --------------------------------------------------------------------------
    # Figure 5: Probability Calibration Diagnostics (Reliability Diagram)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, probs in val_probs_dict.items():
        prob_true, prob_pred = calibration_curve(y_val, probs, n_bins=10)
        brier = brier_score_loss(y_val, probs)
        ax.plot(prob_pred, prob_true, marker="s", lw=2, label=f"{name} (Brier = {brier:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfectly Calibrated")
    ax.set_title("Probability Calibration Curves (Reliability Diagram)", fontsize=13, weight="bold")
    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Fraction of True Delayed Flights", fontsize=11)
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    fig_path = output_dir / "fig5_calibration_curves.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    generated_plots.append(fig_path)

    # --------------------------------------------------------------------------
    # Figure 6: Gradient Descent Loss Convergence Curve
    # --------------------------------------------------------------------------
    if gd_model is not None and len(gd_model.loss_history_) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(gd_model.loss_history_, lw=2, color="steelblue", label="Training BCE Loss")
        if len(gd_model.val_loss_history_) > 0:
            ax.plot(gd_model.val_loss_history_, lw=2, color="coral", linestyle="--", label="Validation BCE Loss")
        ax.set_title("First-Principles Gradient Descent Loss Convergence", fontsize=13, weight="bold")
        ax.set_xlabel("Optimization Epoch", fontsize=11)
        ax.set_ylabel("Binary Cross-Entropy Loss J(theta)", fontsize=11)
        ax.legend(frameon=True, fontsize=10)
        fig_path = output_dir / "fig6_gd_loss_convergence.png"
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()
        generated_plots.append(fig_path)

    # --------------------------------------------------------------------------
    # Figure 7: Top Logistic Regression Odds Ratios & XGBoost Feature Importances
    # --------------------------------------------------------------------------
    if lr_model is not None and feature_names is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 7A: Logistic Regression Odds Ratios: exp(beta)
        coefs = lr_model.coef_[0]
        odds_ratios = np.exp(coefs)
        # Select top 10 highest and top 5 lowest
        sorted_indices = np.argsort(odds_ratios)
        top_indices = np.concatenate([sorted_indices[:5], sorted_indices[-10:]])
        names_sub = [feature_names[i] for i in top_indices]
        or_sub = odds_ratios[top_indices]

        colors = ["forestgreen" if val >= 1.0 else "firebrick" for val in or_sub]
        axes[0].barh(range(len(names_sub)), or_sub, color=colors, alpha=0.85)
        axes[0].axvline(1.0, color="black", linestyle="--", lw=1.2)
        axes[0].set_yticks(range(len(names_sub)))
        axes[0].set_yticklabels(names_sub, fontsize=9)
        axes[0].set_xlabel("Odds Ratio: exp(beta)", fontsize=10)
        axes[0].set_title("Logistic Regression: Key Feature Odds Ratios\n(exp(beta) > 1 increases delay odds)", fontsize=11, weight="bold")

        # 7B: XGBoost Feature Importance
        if xgb_model is not None:
            importances = xgb_model.feature_importances_
            top_xgb_idx = np.argsort(importances)[-15:]
            axes[1].barh(range(len(top_xgb_idx)), importances[top_xgb_idx], color="darkorange", alpha=0.85)
            axes[1].set_yticks(range(len(top_xgb_idx)))
            axes[1].set_yticklabels([feature_names[i] for i in top_xgb_idx], fontsize=9)
            axes[1].set_xlabel("Feature Importance (Gini Gain)", fontsize=10)
            axes[1].set_title("XGBoost: Top 15 Feature Importances", fontsize=11, weight="bold")
        else:
            axes[1].text(0.5, 0.5, "XGBoost Not Available", ha="center", va="center")

        fig_path = output_dir / "fig7_feature_interpretability.png"
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()
        generated_plots.append(fig_path)

    # --------------------------------------------------------------------------
    # Figure 8: Multiclass Delay Category Confusion Matrix (Optional Extension)
    # --------------------------------------------------------------------------
    if y_val_multi is not None and multi_probs is not None:
        y_multi_pred = np.argmax(multi_probs, axis=1)
        cm_m = confusion_matrix(y_val_multi, y_multi_pred, labels=[0, 1, 2])
        cm_m_norm = cm_m.astype("float") / cm_m.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(8, 6.5))
        sns.heatmap(
            cm_m,
            annot=True,
            fmt="d",
            cmap="Purples",
            ax=ax,
            cbar=False,
            annot_kws={"size": 12, "weight": "bold"},
            xticklabels=MULTICLASS_CLASS_NAMES,
            yticklabels=MULTICLASS_CLASS_NAMES,
        )
        for i in range(3):
            for j in range(3):
                ax.text(
                    j + 0.5,
                    i + 0.72,
                    f"({cm_m_norm[i, j]:.1%})",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="gray" if cm_m[i, j] < (cm_m.max() / 2) else "lightyellow",
                )
        ax.set_title("Multiclass Extension: Delay Severity Confusion Matrix\n(3-Tier Classification: On-Time, Moderate, Severe)", fontsize=12, weight="bold")
        ax.set_xlabel("Predicted Delay Severity Tier", fontsize=11)
        ax.set_ylabel("Actual Delay Severity Tier", fontsize=11)
        fig_path = output_dir / "fig8_multiclass_confusion.png"
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()
        generated_plots.append(fig_path)

    return generated_plots


# ==============================================================================
# 6. Artifact Serialization & Validation
# ==============================================================================

def save_classification_artifact(
    pipeline: ColumnTransformer,
    model: Any,
    metadata: Dict[str, Any],
    file_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Serialize the complete trained classification pipeline (preprocessing + classifier)
    and metadata to a joblib artifact under models/.
    """
    if file_path is None:
        file_path = MODELS_DIR / "classification_pipeline.joblib"
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "pipeline": pipeline,
        "model": model,
        "metadata": metadata,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "expected_features": ALL_CLASSIFICATION_FEATURES,
    }

    joblib.dump(artifact, file_path, compress=3)
    return file_path


def load_and_validate_pipeline(
    file_path: Union[str, Path],
    sample_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Validate saved model artifact:
      1. Confirms file existence and readability.
      2. Validates metadata and expected feature schema.
      3. Executes inference test on sample input to verify shapes and probability ranges [0, 1].
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {file_path}")

    artifact = joblib.load(file_path)
    assert "pipeline" in artifact, "Corrupt artifact: 'pipeline' key missing."
    assert "model" in artifact, "Corrupt artifact: 'model' key missing."
    assert "metadata" in artifact, "Corrupt artifact: 'metadata' key missing."

    pipeline = artifact["pipeline"]
    model = artifact["model"]
    expected_feats = artifact.get("expected_features", ALL_CLASSIFICATION_FEATURES)

    # Validate test inference on sample data
    X_sample = sample_df[expected_feats].copy()
    X_trans = pipeline.transform(X_sample)
    probs = model.predict_proba(X_trans)[:, 1]

    assert len(probs) == len(sample_df), f"Output length mismatch: expected {len(sample_df)}, got {len(probs)}"
    assert np.all((probs >= 0.0) & (probs <= 1.0)), "Probabilities violated [0, 1] range."

    return {
        "status": "VALIDATED",
        "file_path": str(file_path),
        "file_size_bytes": file_path.stat().st_size,
        "expected_features_count": len(expected_feats),
        "sample_inference_count": len(sample_df),
        "mean_sample_prob": float(np.mean(probs)),
        "metadata": artifact["metadata"],
    }


# ==============================================================================
# 7. Main Workflow Execution
# ==============================================================================

def main():
    print("=" * 80)
    print("Flight Delay Prediction — Task 4: Classification Models & Benchmarking")
    print("=" * 80)

    # 1. Load Data
    print("\n[Step 1] Loading prepared feature dataset and verifying chronological splits...")
    train_df, val_df, test_df, meta = load_and_partition_classification_data()
    print(f"Loaded: Train={meta['train_rows']:,} | Val={meta['val_rows']:,} | Test={meta['test_rows']:,}")
    print(f"Target Delay Rates: Train={meta['train_delayed_rate']:.2%} | Val={meta['val_delayed_rate']:.2%} | Test={meta['test_delayed_rate']:.2%}")

    # 2. Extract Features
    X_train_df, y_train, X_val_df, y_val, X_test_df, y_test = extract_feature_matrices(
        train_df, val_df, test_df, ALL_CLASSIFICATION_FEATURES
    )

    # 3. Fit Preprocessing Pipeline strictly on Train
    print("\n[Step 2] Fitting leakage-free ColumnTransformer strictly on training partition...")
    preprocessor = build_classification_pipeline(FEATURE_COLUMNS_NUMERIC, FEATURE_COLUMNS_CATEGORICAL)
    X_train_trans = preprocessor.fit_transform(X_train_df)
    X_val_trans = preprocessor.transform(X_val_df)
    X_test_trans = preprocessor.transform(X_test_df)

    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = cat_encoder.get_feature_names_out(FEATURE_COLUMNS_CATEGORICAL).tolist()
    all_transformed_feature_names = FEATURE_COLUMNS_NUMERIC + cat_feature_names
    print(f"Preprocessed Feature Matrix Dimensions: {X_train_trans.shape[1]} features (31 numeric + {len(cat_feature_names)} one-hot)")

    # 4. Train Dummy Baseline
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train_trans, y_train)
    dummy_val_probs = dummy.predict_proba(X_val_trans)[:, 1]
    dummy_metrics = compute_classification_metrics(y_val, dummy_val_probs, threshold=0.5)

    # 5. Train Educational First-Principles Gradient Descent
    print("\n[Step 3] Training Educational First-Principles Logistic Regression (BCE / MLE)...")
    gd_clf = LogisticRegressionGD(learning_rate=0.08, n_epochs=200, batch_size=256, l2_reg=0.01, random_state=RANDOM_STATE)
    gd_clf.fit(X_train_trans, y_train, X_val=X_val_trans, y_val=y_val)
    gd_val_probs = gd_clf.predict_proba(X_val_trans)[:, 1]
    gd_metrics = compute_classification_metrics(y_val, gd_val_probs, threshold=0.5)
    print(f"GD Epoch 1 Loss: {gd_clf.loss_history_[0]:.4f} -> Final Loss: {gd_clf.loss_history_[-1]:.4f}")
    print(f"GD Validation ROC-AUC: {gd_metrics['roc_auc']:.4f} | PR-AUC: {gd_metrics['pr_auc']:.4f}")

    # 6. Train Production Scikit-Learn Logistic Regression Baseline
    print("\n[Step 4] Training Production Scikit-Learn Logistic Regression (L-BFGS, L2)...")
    lr_clf = train_logistic_regression_baseline(X_train_trans, y_train, C=1.0, max_iter=1000, random_state=RANDOM_STATE)
    lr_val_probs = lr_clf.predict_proba(X_val_trans)[:, 1]
    lr_metrics = compute_classification_metrics(y_val, lr_val_probs, threshold=0.5)
    print(f"Logistic Regression Val ROC-AUC: {lr_metrics['roc_auc']:.4f} | PR-AUC: {lr_metrics['pr_auc']:.4f} | Brier: {lr_metrics['brier_score']:.4f}")

    # 7. Train XGBoost Classifier
    print("\n[Step 5] Training XGBoost Classifier Benchmark...")
    xgb_metrics = None
    xgb_clf = None
    xgb_val_probs = None
    if HAS_XGBOOST:
        xgb_clf = train_xgboost_classifier(X_train_trans, y_train, n_estimators=120, max_depth=3, learning_rate=0.06, random_state=RANDOM_STATE)
        xgb_val_probs = xgb_clf.predict_proba(X_val_trans)[:, 1]
        xgb_metrics = compute_classification_metrics(y_val, xgb_val_probs, threshold=0.5)
        print(f"XGBoost Val ROC-AUC: {xgb_metrics['roc_auc']:.4f} | PR-AUC: {xgb_metrics['pr_auc']:.4f} | Brier: {xgb_metrics['brier_score']:.4f}")
    else:
        print("XGBoost unavailable. Skipping tree ensemble benchmark.")

    # 8. Model Selection & Probability Threshold Optimization on Validation Data
    print("\n[Step 6] Running Probability Threshold Optimization on Validation Set...")
    val_probs_dict = {
        "Majority Dummy": dummy_val_probs,
        "First-Principles GD": gd_val_probs,
        "Logistic Regression (MLE)": lr_val_probs,
    }
    if xgb_val_probs is not None:
        val_probs_dict["XGBoost Classifier"] = xgb_val_probs

    # We evaluate thresholds for the primary candidate (Logistic Regression)
    threshold_df = evaluate_threshold_grid(y_val, lr_val_probs, fn_cost_ratio=2.5)
    # Optimal threshold selected by maximizing operational airline utility (balanced recall & precision)
    best_row = threshold_df.loc[threshold_df["operational_utility"].idxmax()]
    best_threshold = float(best_row["threshold"])
    print(f"Default (tau = 0.50): Acc={lr_metrics['accuracy']:.4f}, Prec={lr_metrics['precision']:.4f}, Rec={lr_metrics['recall']:.4f}, F1={lr_metrics['f1']:.4f}")
    print(f"Selected Optimal Threshold: tau = {best_threshold:.2f}")
    print(f"At tau = {best_threshold:.2f}: Acc={best_row['accuracy']:.4f}, Prec={best_row['precision']:.4f}, Rec={best_row['recall']:.4f}, F1={best_row['f1']:.4f}")

    # 9. FROZEN Test Set Evaluation (Conducted ONLY on Selected Model & Frozen Threshold)
    print("\n[Step 7] Evaluating Selected Model on Held-Out Test Set (FROZEN EVALUATION)...")
    lr_test_probs = lr_clf.predict_proba(X_test_trans)[:, 1]
    test_metrics_default = compute_classification_metrics(y_test, lr_test_probs, threshold=0.50)
    test_metrics_optimal = compute_classification_metrics(y_test, lr_test_probs, threshold=best_threshold)
    print(f"Test Set Default (tau=0.50): ROC-AUC={test_metrics_default['roc_auc']:.4f}, PR-AUC={test_metrics_default['pr_auc']:.4f}, F1={test_metrics_default['f1']:.4f}, Rec={test_metrics_default['recall']:.4f}")
    print(f"Test Set Frozen (tau={best_threshold:.2f}): ROC-AUC={test_metrics_optimal['roc_auc']:.4f}, PR-AUC={test_metrics_optimal['pr_auc']:.4f}, F1={test_metrics_optimal['f1']:.4f}, Rec={test_metrics_optimal['recall']:.4f}")

    # 10. Optional Multiclass Extension
    print("\n[Step 8] Evaluating Optional Multiclass Extension (3-Tier Delay Severity)...")
    y_train_multi, multi_meta_train = create_multiclass_targets(train_df)
    y_val_multi, multi_meta_val = create_multiclass_targets(val_df)
    y_test_multi, multi_meta_test = create_multiclass_targets(test_df)

    multi_lr = train_multiclass_logistic(X_train_trans, y_train_multi, C=1.0, max_iter=1000)
    multi_val_probs = multi_lr.predict_proba(X_val_trans)
    multi_metrics = compute_multiclass_metrics(y_val_multi, multi_val_probs)
    print(f"Multiclass Validation Accuracy: {multi_metrics['accuracy']:.4f} | Macro-F1: {multi_metrics['macro_f1']:.4f} | Log Loss: {multi_metrics['multiclass_log_loss']:.4f}")

    # 11. Generate Publication Visualizations
    print("\n[Step 9] Generating Publication Figures to results/figures/classification/...")
    plots = generate_classification_plots(
        y_val=y_val,
        val_probs_dict=val_probs_dict,
        threshold_df=threshold_df,
        optimal_threshold=best_threshold,
        gd_model=gd_clf,
        lr_model=lr_clf,
        xgb_model=xgb_clf,
        feature_names=all_transformed_feature_names,
        y_val_multi=y_val_multi,
        multi_probs=multi_val_probs,
    )
    for p in plots:
        print(f"  Saved figure: {p.name}")

    # 12. Save and Validate Model Artifact
    print("\n[Step 10] Serializing and Validating Final Model Artifact...")
    artifact_metadata = {
        "model_name": "LogisticRegression_MLE_L2",
        "objective": "Pre-departure flight delay binary classification (arrival delay > 15 min)",
        "optimal_threshold": best_threshold,
        "input_features": ALL_CLASSIFICATION_FEATURES,
        "validation_metrics": {
            "default_0_50": lr_metrics,
            "optimal_threshold": compute_classification_metrics(y_val, lr_val_probs, threshold=best_threshold),
        },
        "test_metrics": {
            "default_0_50": test_metrics_default,
            "optimal_threshold": test_metrics_optimal,
        },
        "xgb_benchmark_metrics": xgb_metrics if xgb_metrics else {},
        "multiclass_metrics": multi_metrics,
    }

    artifact_path = save_classification_artifact(
        pipeline=preprocessor,
        model=lr_clf,
        metadata=artifact_metadata,
        file_path=MODELS_DIR / "classification_pipeline.joblib",
    )
    print(f"Saved artifact to: {artifact_path}")

    validation_result = load_and_validate_pipeline(artifact_path, test_df.head(20))
    print(f"Model Artifact Validation Status: {validation_result['status']}")
    print("=" * 80)
    print("Task 4 Workflow Executed Successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
