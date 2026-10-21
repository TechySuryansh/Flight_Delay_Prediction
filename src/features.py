"""
Feature Engineering and Dataset Preparation Module
Project 21: Flight Delay Prediction and Rescheduling — AML Phase 1
Author: Member 2B (Feature Engineering & Dataset Preparation)

This module implements:
1. Operational Prediction Point Definition ($T - 2$ hours before scheduled departure).
2. Leakage-free pre-departure feature engineering:
   - Temporal cyclical encodings (sin/cos for hour, minute, day of week, month).
   - Operational time-of-day bins and peak evening delay rush indicators.
   - Distance transformations (log-scale, categorical bands) and scheduled speed buffer.
   - Route interactions and categorical identifiers.
3. Chronological historical delay priors (Empirical Bayes target encoding):
   - Out-of-fold (OOF) cross-validation on the training set to prevent self-target leakage.
   - Train-only statistics mapped strictly forward to validation and test partitions.
   - Unseen categories mapped to the global training baseline.
4. Strict anti-leakage barriers and validation checks.
5. Reusable Scikit-Learn preprocessing pipeline (ColumnTransformer) fit ONLY on training data.
6. Chronological dataset partitioning (Train: Jan-Aug, Val: Sep-Oct, Test: Nov-Dec 2015).
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    DATA_DIR,
    LEAKAGE_COLUMNS,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
)

# ==============================================================================
# 1. Operational Prediction Point and Column Segregation
# ==============================================================================

PREDICTION_POINT_DEFINITION = (
    "Prediction Point: Exactly T - 2 hours (120 minutes) prior to scheduled gate departure. "
    "At this operational juncture, only scheduled timetable attributes, calendar variables, "
    "and strictly historical outcomes (completed prior to dispatch) are legally available. "
    "All post-departure fields (actual departure/arrival times, taxi-out/in, airborne duration, "
    "wheels-off/on, actual delays, and delay breakdown causes) are strictly quarantined."
)

TARGET_COLUMNS: List[str] = [
    "DELAYED",          # Binary classification label: 1 if ARRIVAL_DELAY > 15 min, else 0
    "ARRIVAL_DELAY",    # Continuous regression target in minutes (auditing & Member 3)
]

IDENTIFIER_COLUMNS: List[str] = [
    "FLIGHT_DATE",      # Chronological flight date (YYYY-MM-DD)
    "FLIGHT_NUMBER",    # Carrier flight number
    "TAIL_NUMBER",      # Aircraft registration identifier (imputed 'UNKNOWN' if missing)
    "ROUTE",            # Origin-Destination pair identifier
    "SPLIT_SET",        # Partition indicator ('train', 'val', 'test')
]

PROHIBITED_LEAKAGE_COLUMNS: List[str] = list(set(LEAKAGE_COLUMNS + [
    "ARRIVAL_DELAY",
    "DELAYED",
    "DEPARTURE_TIME",
    "DEPARTURE_DELAY",
    "TAXI_OUT",
    "WHEELS_OFF",
    "ELAPSED_TIME",
    "AIR_TIME",
    "WHEELS_ON",
    "TAXI_IN",
    "ARRIVAL_TIME",
    "CANCELLED",
    "DIVERTED",
    "CANCELLATION_REASON",
    "AIR_SYSTEM_DELAY",
    "SECURITY_DELAY",
    "AIRLINE_DELAY",
    "LATE_AIRCRAFT_DELAY",
    "WEATHER_DELAY",
]))

FEATURE_COLUMNS_NUMERIC: List[str] = [
    # Scheduled Departure Encodings
    "DEP_HOUR",
    "DEP_MINUTE",
    "DEP_TIME_MINUTES",
    "DEP_HOUR_SIN",
    "DEP_HOUR_COS",
    "DEP_TIME_SIN",
    "DEP_TIME_COS",
    "DEP_HOUR_PEAK",
    
    # Scheduled Arrival Encodings
    "ARR_HOUR",
    "ARR_MINUTE",
    "ARR_TIME_MINUTES",
    "ARR_HOUR_SIN",
    "ARR_HOUR_COS",
    "ARR_NIGHT_CURFEW",
    
    # Calendar & Cyclical Temporal Encodings
    "DAY_OF_WEEK",
    "DOW_SIN",
    "DOW_COS",
    "IS_WEEKEND",
    "MONTH",
    "MONTH_SIN",
    "MONTH_COS",
    "DAY",
    "QUARTER",
    "IS_MONTH_END",
    
    # Distance & Scheduled Speed Buffer
    "DISTANCE",
    "DISTANCE_LOG",
    "SCHEDULED_TIME",
    "PLANNED_SPEED",
    
    # Chronological Historical Delay Priors (Empirical Bayes Smoothed)
    "CARRIER_HIST_DELAY_RATE",
    "ORIGIN_HIST_DELAY_RATE",
    "DEST_HIST_DELAY_RATE",
]

FEATURE_COLUMNS_CATEGORICAL: List[str] = [
    "AIRLINE",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",
    "DEP_TIME_OF_DAY",
    "DISTANCE_CATEGORY",
]

FEATURE_COLUMNS_ALL: List[str] = FEATURE_COLUMNS_NUMERIC + FEATURE_COLUMNS_CATEGORICAL


# ==============================================================================
# 2. Chronological Splitting Logic
# ==============================================================================

def assign_chronological_split(
    df: pd.DataFrame,
    train_end_date: str = "2015-08-31",
    val_end_date: str = "2015-10-31",
) -> pd.DataFrame:
    """
    Assign chronological partitions strictly based on FLIGHT_DATE without random shuffling:
      - Train:      FLIGHT_DATE <= train_end_date (Months 1–8: Jan 1 to Aug 31, ~67%)
      - Validation: train_end_date < FLIGHT_DATE <= val_end_date (Months 9–10: Sep 1 to Oct 31, ~17%)
      - Test:       FLIGHT_DATE > val_end_date (Months 11–12: Nov 1 to Dec 31, ~16%)

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing FLIGHT_DATE.
    train_end_date : str
        Upper inclusive date boundary for training partition.
    val_end_date : str
        Upper inclusive date boundary for validation partition.

    Returns
    -------
    pd.DataFrame
        Dataframe with standardized 'SPLIT_SET' column ('train', 'val', 'test').
    """
    df = df.copy()
    dates = pd.to_datetime(df["FLIGHT_DATE"])
    
    df["SPLIT_SET"] = "train"
    val_mask = (dates > pd.to_datetime(train_end_date)) & (dates <= pd.to_datetime(val_end_date))
    test_mask = dates > pd.to_datetime(val_end_date)
    
    df.loc[val_mask, "SPLIT_SET"] = "val"
    df.loc[test_mask, "SPLIT_SET"] = "test"
    
    return df


# ==============================================================================
# 3. Feature Engineering Functions
# ==============================================================================

def get_departure_time_of_day(hour: int) -> str:
    """Classify 24-hour departure hour into standard operational periods."""
    if 4 <= hour < 8:
        return "Early Morning"
    elif 8 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 22:
        return "Evening"
    else:
        return "Night"


def get_distance_category(dist: float) -> str:
    """Classify statute flight distance into standard FAA operational tiers."""
    if dist < 500:
        return "Short"
    elif dist < 1500:
        return "Medium"
    else:
        return "Long"


def compute_empirical_bayes_priors(
    train_df: pd.DataFrame,
    group_col: str,
    target_col: str = "DELAYED",
    smoothing_m: float = 10.0,
) -> Tuple[Dict[Any, float], float]:
    """
    Calculate Empirical Bayes smoothed historical target rate on the training split:
      prior(c) = (sum(target) + m * global_mean) / (count + m)

    Parameters
    ----------
    train_df : pd.DataFrame
        Training set rows only.
    group_col : str
        Categorical column (e.g. 'AIRLINE', 'ORIGIN_AIRPORT').
    target_col : str
        Binary target column.
    smoothing_m : float
        Bayesian pseudo-count smoothing parameter (default: 10.0).

    Returns
    -------
    Tuple[Dict[Any, float], float]
        Dictionary mapping category to smoothed prior rate, and global training mean.
    """
    global_mean = float(train_df[target_col].mean())
    stats = train_df.groupby(group_col)[target_col].agg(["count", "sum"])
    smoothed = (stats["sum"] + smoothing_m * global_mean) / (stats["count"] + smoothing_m)
    return smoothed.to_dict(), global_mean


def compute_oof_target_encoding(
    train_df: pd.DataFrame,
    group_col: str,
    target_col: str = "DELAYED",
    smoothing_m: float = 10.0,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> pd.Series:
    """
    Compute out-of-fold (OOF) target encoding within the training partition.
    Ensures that no training row uses its own target label in its historical prior feature.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training split only.
    group_col : str
        Categorical grouping column.
    target_col : str
        Target column name.
    smoothing_m : float
        Smoothing pseudo-count.
    n_splits : int
        Number of cross-validation folds.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    pd.Series
        OOF smoothed delay prior series indexed to train_df.
    """
    global_mean = float(train_df[target_col].mean())
    oof_series = pd.Series(index=train_df.index, dtype=float)
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fit_idx, oof_idx in kf.split(train_df):
        fold_train = train_df.iloc[fit_idx]
        fold_val = train_df.iloc[oof_idx]
        
        stats = fold_train.groupby(group_col)[target_col].agg(["count", "sum"])
        smoothed = (stats["sum"] + smoothing_m * global_mean) / (stats["count"] + smoothing_m)
        priors_dict = smoothed.to_dict()
        
        oof_series.iloc[oof_idx] = fold_val[group_col].map(priors_dict).fillna(global_mean)
        
    return oof_series


def engineer_flight_features(
    df: pd.DataFrame,
    train_end_date: str = "2015-08-31",
    val_end_date: str = "2015-10-31",
    smoothing_m: float = 10.0,
) -> pd.DataFrame:
    """
    Full leakage-safe feature engineering pipeline for cleaned flight datasets.

    Steps:
    1. Assign chronological train/validation/test split.
    2. Extract and encode scheduled departure temporal features (hour, min, continuous sin/cos, peak flag).
    3. Extract and encode scheduled arrival temporal features (hour, min, sin/cos, night curfew flag).
    4. Construct calendar and seasonal cyclic features (day of week, month, quarter, weekend, month-end).
    5. Construct distance representations (log-distance, FAA category tiers, planned speed buffer).
    6. Construct route interactions and impute missing tail identifiers ('UNKNOWN').
    7. Compute Empirical Bayes historical delay priors strictly on training data:
       - 5-fold out-of-fold encoding for training records.
       - Training-fitted priors mapped forward to validation and test records.
       - Fallback to global training mean for unseen categories.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned flight records from Member 1B.
    train_end_date : str
        Upper boundary for train partition.
    val_end_date : str
        Upper boundary for validation partition.
    smoothing_m : float
        Bayesian smoothing weight.

    Returns
    -------
    pd.DataFrame
        Enriched feature dataset containing identifiers, targets, and engineered features.
    """
    out = df.copy()
    
    # 1. Chronological Partitioning
    out = assign_chronological_split(out, train_end_date=train_end_date, val_end_date=val_end_date)
    
    # 2. Scheduled Departure Encodings
    out["DEP_HOUR"] = (out["SCHEDULED_DEPARTURE"] // 100).astype(int)
    out["DEP_MINUTE"] = (out["SCHEDULED_DEPARTURE"] % 100).astype(int)
    out["DEP_TIME_MINUTES"] = (out["DEP_HOUR"] * 60 + out["DEP_MINUTE"]).astype(int)
    
    # Hourly cyclical representation (24-hour cycle)
    out["DEP_HOUR_SIN"] = np.sin(2.0 * np.pi * out["DEP_HOUR"] / 24.0)
    out["DEP_HOUR_COS"] = np.cos(2.0 * np.pi * out["DEP_HOUR"] / 24.0)
    
    # Continuous high-resolution cyclical representation (1440-minute day cycle)
    out["DEP_TIME_SIN"] = np.sin(2.0 * np.pi * out["DEP_TIME_MINUTES"] / 1440.0)
    out["DEP_TIME_COS"] = np.cos(2.0 * np.pi * out["DEP_TIME_MINUTES"] / 1440.0)
    
    # Member 2A empirical finding: Diurnal surge in evening peak hours (17:00–22:00)
    out["DEP_HOUR_PEAK"] = out["DEP_HOUR"].isin([17, 18, 19, 20, 21, 22]).astype(int)
    out["DEP_TIME_OF_DAY"] = out["DEP_HOUR"].apply(get_departure_time_of_day)
    
    # 3. Scheduled Arrival Encodings
    out["ARR_HOUR"] = (out["SCHEDULED_ARRIVAL"] // 100).astype(int)
    out["ARR_MINUTE"] = (out["SCHEDULED_ARRIVAL"] % 100).astype(int)
    out["ARR_TIME_MINUTES"] = (out["ARR_HOUR"] * 60 + out["ARR_MINUTE"]).astype(int)
    out["ARR_HOUR_SIN"] = np.sin(2.0 * np.pi * out["ARR_HOUR"] / 24.0)
    out["ARR_HOUR_COS"] = np.cos(2.0 * np.pi * out["ARR_HOUR"] / 24.0)
    out["ARR_NIGHT_CURFEW"] = ((out["ARR_HOUR"] >= 23) | (out["ARR_HOUR"] < 6)).astype(int)
    
    # 4. Calendar and Seasonality Encodings
    out["DOW_SIN"] = np.sin(2.0 * np.pi * (out["DAY_OF_WEEK"] - 1) / 7.0)
    out["DOW_COS"] = np.cos(2.0 * np.pi * (out["DAY_OF_WEEK"] - 1) / 7.0)
    out["IS_WEEKEND"] = out["DAY_OF_WEEK"].isin([6, 7]).astype(int)
    
    out["MONTH_SIN"] = np.sin(2.0 * np.pi * (out["MONTH"] - 1) / 12.0)
    out["MONTH_COS"] = np.cos(2.0 * np.pi * (out["MONTH"] - 1) / 12.0)
    out["QUARTER"] = ((out["MONTH"] - 1) // 3 + 1).astype(int)
    out["IS_MONTH_END"] = (out["DAY"] >= 28).astype(int)
    
    # 5. Distance & Operational Flight Geometry
    out["DISTANCE_LOG"] = np.log1p(out["DISTANCE"].astype(float))
    out["DISTANCE_CATEGORY"] = out["DISTANCE"].apply(get_distance_category)
    
    # Planned speed proxy (miles per minute). Lower values reflect schedule padding / buffer.
    # Clip scheduled time at minimum 1.0 to avoid division by zero
    sched_time_safe = out["SCHEDULED_TIME"].clip(lower=1.0)
    out["PLANNED_SPEED"] = (out["DISTANCE"] / sched_time_safe).astype(float)
    
    # 6. Route and Missing Identifiers
    out["ROUTE"] = out["ORIGIN_AIRPORT"].astype(str) + "_" + out["DESTINATION_AIRPORT"].astype(str)
    out["TAIL_NUMBER"] = out["TAIL_NUMBER"].fillna("UNKNOWN")
    
    # 7. Chronological Historical Delay Priors (Train-Fit Only)
    train_mask = out["SPLIT_SET"] == "train"
    val_mask = out["SPLIT_SET"] == "val"
    test_mask = out["SPLIT_SET"] == "test"
    
    train_df = out[train_mask]
    
    prior_targets = [
        ("AIRLINE", "CARRIER_HIST_DELAY_RATE"),
        ("ORIGIN_AIRPORT", "ORIGIN_HIST_DELAY_RATE"),
        ("DESTINATION_AIRPORT", "DEST_HIST_DELAY_RATE"),
    ]
    
    for cat_col, feat_name in prior_targets:
        # Fit priors strictly on train split
        priors_dict, global_train_mean = compute_empirical_bayes_priors(
            train_df=train_df,
            group_col=cat_col,
            target_col="DELAYED",
            smoothing_m=smoothing_m,
        )
        
        # Map onto validation and test splits with fallback to global train mean
        out.loc[val_mask, feat_name] = out.loc[val_mask, cat_col].map(priors_dict).fillna(global_train_mean)
        out.loc[test_mask, feat_name] = out.loc[test_mask, cat_col].map(priors_dict).fillna(global_train_mean)
        
        # Compute 5-fold OOF target encoding on train split to prevent self-target leakage
        oof_train_priors = compute_oof_target_encoding(
            train_df=train_df,
            group_col=cat_col,
            target_col="DELAYED",
            smoothing_m=smoothing_m,
            n_splits=5,
            random_state=RANDOM_STATE,
        )
        out.loc[train_mask, feat_name] = oof_train_priors
        
    return out


# ==============================================================================
# 4. Scikit-Learn Preprocessing Pipeline
# ==============================================================================

def build_feature_pipeline(
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> ColumnTransformer:
    """
    Build a reusable Scikit-Learn ColumnTransformer for linear/logistic regression.
    
    Transformations:
      - Numeric: SimpleImputer(median) -> StandardScaler()
      - Categorical: SimpleImputer(constant 'UNKNOWN') -> OneHotEncoder(drop='first' optional, handle_unknown='ignore')

    Rules:
      - Must be fit ONLY on X_train.
      - Transforms X_train, X_val, and X_test without label leakage.

    Parameters
    ----------
    numeric_features : Optional[List[str]]
        List of numeric column names. Defaults to FEATURE_COLUMNS_NUMERIC.
    categorical_features : Optional[List[str]]
        List of categorical column names. Defaults to FEATURE_COLUMNS_CATEGORICAL.

    Returns
    -------
    ColumnTransformer
        Unfitted scikit-learn ColumnTransformer.
    """
    if numeric_features is None:
        numeric_features = FEATURE_COLUMNS_NUMERIC
    if categorical_features is None:
        categorical_features = FEATURE_COLUMNS_CATEGORICAL
        
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )
    
    return preprocessor


# ==============================================================================
# 5. Validation Suite
# ==============================================================================

def validate_feature_dataset(
    df: pd.DataFrame,
    source_row_count: Optional[int] = None,
    feature_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Execute rigorous validation assertions on the prepared feature dataset:
    1. Row count fidelity against source input.
    2. Zero duplicate rows.
    3. Complete absence of prohibited data leakage columns in feature list.
    4. Target variable separation and non-null validity.
    5. Chronological boundary separation (zero date overlap across splits).
    6. Historical prior bounds and non-null integrity.

    Parameters
    ----------
    df : pd.DataFrame
        Engineered feature dataframe.
    source_row_count : Optional[int]
        Expected row count from cleaned input.
    feature_cols : Optional[List[str]]
        List of candidate features to audit. Defaults to FEATURE_COLUMNS_ALL.

    Returns
    -------
    Dict[str, Any]
        Dictionary of audit results and validation metrics.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS_ALL
        
    audit: Dict[str, Any] = {}
    
    # 1. Row Count
    audit["row_count"] = len(df)
    if source_row_count is not None:
        assert len(df) == source_row_count, (
            f"Row count mismatch: expected {source_row_count}, got {len(df)}"
        )
    audit["row_count_passed"] = True
    
    # 2. Duplicate Check
    dups = int(df.duplicated().sum())
    audit["duplicate_rows"] = dups
    assert dups == 0, f"Found {dups} duplicate rows in feature dataset."
    audit["duplicate_check_passed"] = True
    
    # 3. Leakage Guardrail Check
    leakage_in_features = [col for col in feature_cols if col in PROHIBITED_LEAKAGE_COLUMNS]
    audit["leakage_features_found"] = leakage_in_features
    assert len(leakage_in_features) == 0, (
        f"CRITICAL DATA LEAKAGE: Prohibited columns found in feature list: {leakage_in_features}"
    )
    audit["leakage_check_passed"] = True
    
    # 4. Target Isolation
    for target in TARGET_COLUMNS:
        assert target in df.columns, f"Target column '{target}' missing from dataset."
        assert df[target].isnull().sum() == 0, f"Null values detected in target '{target}'."
        assert target not in feature_cols, f"Target '{target}' mistakenly included in feature list!"
    audit["target_isolation_passed"] = True
    
    # 5. Chronological Boundary Check
    assert "SPLIT_SET" in df.columns, "Partition indicator 'SPLIT_SET' missing."
    train_dates = pd.to_datetime(df[df["SPLIT_SET"] == "train"]["FLIGHT_DATE"])
    val_dates = pd.to_datetime(df[df["SPLIT_SET"] == "val"]["FLIGHT_DATE"])
    test_dates = pd.to_datetime(df[df["SPLIT_SET"] == "test"]["FLIGHT_DATE"])
    
    assert train_dates.max() < val_dates.min(), (
        f"Train/Val date overlap: Train max ({train_dates.max()}) >= Val min ({val_dates.min()})"
    )
    assert val_dates.max() < test_dates.min(), (
        f"Val/Test date overlap: Val max ({val_dates.max()}) >= Test min ({test_dates.min()})"
    )
    audit["chronological_splits"] = {
        "train": {"count": len(train_dates), "min_date": str(train_dates.min().date()), "max_date": str(train_dates.max().date())},
        "val": {"count": len(val_dates), "min_date": str(val_dates.min().date()), "max_date": str(val_dates.max().date())},
        "test": {"count": len(test_dates), "min_date": str(test_dates.min().date()), "max_date": str(test_dates.max().date())},
    }
    audit["chronological_splits_passed"] = True
    
    # 6. Feature Null & Infinite Value Check
    feature_nulls = df[feature_cols].isnull().sum().to_dict()
    unresolved_nulls = {k: v for k, v in feature_nulls.items() if v > 0}
    assert len(unresolved_nulls) == 0, f"Unresolved nulls detected in features: {unresolved_nulls}"
    
    for num_col in FEATURE_COLUMNS_NUMERIC:
        assert not np.isinf(df[num_col]).any(), f"Infinite values detected in numeric feature: {num_col}"
    audit["feature_integrity_passed"] = True
    
    return audit


# ==============================================================================
# 6. Command-Line Execution
# ==============================================================================

def main():
    print("=" * 78)
    print("Flight Delay Prediction — Task 2B: Feature Engineering Pipeline")
    print("=" * 78)
    
    clean_path = PROCESSED_DATA_DIR / "flights_cleaned.csv"
    output_path = PROCESSED_DATA_DIR / "flights_features.csv"
    
    if not clean_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at: {clean_path}")
        
    print(f"Loading cleaned dataset from: {clean_path}")
    df_clean = pd.read_csv(clean_path)
    print(f"Loaded {len(df_clean):,} rows and {len(df_clean.columns)} columns.")
    
    print("\nEngineering pre-departure features and computing train-only historical priors...")
    df_features = engineer_flight_features(df_clean)
    print(f"Engineered dataset shape: {df_features.shape}")
    
    print("\nRunning rigorous anti-leakage and schema validations...")
    audit = validate_feature_dataset(df_features, source_row_count=len(df_clean))
    print("Validation passed successfully!")
    print(f"Chronological Splits: {audit['chronological_splits']}")
    
    print(f"\nSaving prepared feature dataset to: {output_path}")
    df_features.to_csv(output_path, index=False)
    print(f"Successfully saved {len(df_features):,} rows and {len(df_features.columns)} columns to {output_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
