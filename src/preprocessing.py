"""
Preprocessing and Feature Engineering Module
Flight Delay Prediction - Phase 1: Machine Learning

This module implements:
1. Operational filtering: Excluding cancelled, diverted, and missing-delay records with row audit.
2. Chronological train/validation/test splitting based strictly on flight dates.
3. Pre-departure feature engineering:
   - Scheduled departure hour, minute, cyclical sin/cos encodings.
   - Day of week, weekend indicator, month, cyclical calendar encodings.
   - Route combinations (ORIGIN_DEST).
   - Historical delay priors computed EXCLUSIVELY on training data.
4. Scikit-learn ColumnTransformer and Pipeline:
   - Numerical: Median imputation + Standard scaling.
   - Categorical: Constant imputation + One-Hot Encoding (handling unknown categories).
5. Strict anti-leakage and chronological assertions.
6. Data dictionary generation.
"""

import json
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    COLUMN_ALIASES,
    DELAY_THRESHOLD_MINUTES,
    LEAKAGE_COLUMNS,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    TARGET_COLUMN,
)
from src.utils import parse_hhmm_hour


# ==============================================================================
# 1. Operational Filtering and Target Creation
# ==============================================================================
def prepare_classification_dataset(
    df: pd.DataFrame,
    mapping: Dict[str, str],
    filter_cancelled: bool = True,
    filter_diverted: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Filter out non-operational flights and construct the binary target.

    Scientific & Operational Rationale:
    1. Cancelled flights: Did not operate; arrival delay is undefined (NaN).
    2. Diverted flights: Landed at alternate airport; scheduled arrival dynamics voided.
    3. Missing target: Arrival delay unrecorded; cannot train/evaluate.

    Parameters
    ----------
    df : pd.DataFrame
        Raw flight dataframe.
    mapping : dict
        Detected canonical column mapping.
    filter_cancelled : bool
        Whether to exclude cancelled flights.
    filter_diverted : bool
        Whether to exclude diverted flights.

    Returns
    -------
    tuple of (DataFrame, dict)
        Cleaned dataframe and a detailed audit report tracking dropped rows.
    """
    initial_count = len(df)
    clean_df = df.copy()
    audit = {"initial_rows": initial_count}

    # Step 1: Filter cancelled flights
    if filter_cancelled and "cancellation" in mapping:
        canc_col = mapping["cancellation"]
        canc_mask = (clean_df[canc_col] == 1)
        audit["cancelled_rows_removed"] = int(canc_mask.sum())
        clean_df = clean_df[~canc_mask]
    else:
        audit["cancelled_rows_removed"] = 0

    # Step 2: Filter diverted flights
    if filter_diverted and "diversion" in mapping:
        div_col = mapping["diversion"]
        div_mask = (clean_df[div_col] == 1)
        audit["diverted_rows_removed"] = int(div_mask.sum())
        clean_df = clean_df[~div_mask]
    else:
        audit["diverted_rows_removed"] = 0

    # Step 3: Remove records missing arrival delay
    if "arrival_delay" in mapping:
        arr_col = mapping["arrival_delay"]
        missing_arr_mask = clean_df[arr_col].isnull()
        audit["missing_target_rows_removed"] = int(missing_arr_mask.sum())
        clean_df = clean_df[~missing_arr_mask].copy()

        # Step 4: Construct binary target (1 = delayed >= 15 min, 0 = on-time)
        clean_df[TARGET_COLUMN] = (clean_df[arr_col] >= DELAY_THRESHOLD_MINUTES).astype(int)
    else:
        raise ValueError("Cannot prepare classification dataset: 'arrival_delay' column is missing!")

    audit["final_rows"] = len(clean_df)
    audit["retention_pct"] = (len(clean_df) / initial_count * 100) if initial_count > 0 else 0.0
    audit["target_positive_count"] = int((clean_df[TARGET_COLUMN] == 1).sum())
    audit["target_positive_pct"] = (
        (clean_df[TARGET_COLUMN].mean() * 100) if len(clean_df) > 0 else 0.0
    )

    return clean_df, audit


# ==============================================================================
# 2. Chronological Train / Validation / Test Split
# ==============================================================================
def resolve_flight_dates(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.Series:
    """
    Extract or construct standardized datetime series for each flight record.
    """
    if "date" in mapping and mapping["date"] in df.columns:
        return pd.to_datetime(df[mapping["date"]], errors="coerce")
    elif {"year", "month", "day"}.issubset(mapping.keys()):
        y = df[mapping["year"]]
        m = df[mapping["month"]]
        d = df[mapping["day"]]
        return pd.to_datetime(pd.DataFrame({"year": y, "month": m, "day": d}), errors="coerce")
    else:
        raise ValueError("Cannot resolve flight dates: neither date nor year/month/day columns found!")


def chronological_train_val_test_split(
    df: pd.DataFrame,
    mapping: Dict[str, str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Split flight data strictly chronologically by date.
    Earlier dates -> Training
    Intermediate dates -> Validation
    Latest dates -> Testing

    Guarantees zero future-information leakage into training.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned flight records.
    mapping : dict
        Canonical column mapping.
    train_ratio : float
        Proportion of date range / rows for training.
    val_ratio : float
        Proportion for validation.
    test_ratio : float
        Proportion for testing.

    Returns
    -------
    tuple of (DataFrame, DataFrame, DataFrame, dict)
        train_df, val_df, test_df, split_report
    """
    df = df.copy()
    flight_dates = resolve_flight_dates(df, mapping)
    df["_flight_date_dt"] = flight_dates

    # Drop records with unparseable dates and sort chronologically
    valid_dates_df = df.dropna(subset=["_flight_date_dt"]).sort_values(by="_flight_date_dt").reset_index(drop=True)

    # Extract unique sorted calendar dates to guarantee strictly zero date overlap
    unique_dates = valid_dates_df["_flight_date_dt"].drop_duplicates().sort_values().reset_index(drop=True)
    n_dates = len(unique_dates)

    if n_dates < 3:
        raise ValueError(f"Need at least 3 distinct calendar dates for train/val/test split, found {n_dates}.")

    # Compute date index boundaries
    train_date_end_idx = int(n_dates * train_ratio)
    val_date_end_idx = int(n_dates * (train_ratio + val_ratio))

    # Guard bounds so each set has at least 1 date
    train_date_end_idx = max(1, min(train_date_end_idx, n_dates - 2))
    val_date_end_idx = max(train_date_end_idx + 1, min(val_date_end_idx, n_dates - 1))

    train_cutoff_date = unique_dates.iloc[train_date_end_idx - 1]
    val_start_date = unique_dates.iloc[train_date_end_idx]
    val_cutoff_date = unique_dates.iloc[val_date_end_idx - 1]
    test_start_date = unique_dates.iloc[val_date_end_idx]

    train_df = valid_dates_df[valid_dates_df["_flight_date_dt"] <= train_cutoff_date].copy()
    val_df = valid_dates_df[
        (valid_dates_df["_flight_date_dt"] >= val_start_date)
        & (valid_dates_df["_flight_date_dt"] <= val_cutoff_date)
    ].copy()
    test_df = valid_dates_df[valid_dates_df["_flight_date_dt"] >= test_start_date].copy()

    split_report = {
        "train_rows": len(train_df),
        "train_start_date": train_df["_flight_date_dt"].min().strftime("%Y-%m-%d"),
        "train_end_date": train_df["_flight_date_dt"].max().strftime("%Y-%m-%d"),
        "train_delay_rate_pct": float(train_df[TARGET_COLUMN].mean() * 100),
        "val_rows": len(val_df),
        "val_start_date": val_df["_flight_date_dt"].min().strftime("%Y-%m-%d"),
        "val_end_date": val_df["_flight_date_dt"].max().strftime("%Y-%m-%d"),
        "val_delay_rate_pct": float(val_df[TARGET_COLUMN].mean() * 100),
        "test_rows": len(test_df),
        "test_start_date": test_df["_flight_date_dt"].min().strftime("%Y-%m-%d"),
        "test_end_date": test_df["_flight_date_dt"].max().strftime("%Y-%m-%d"),
        "test_delay_rate_pct": float(test_df[TARGET_COLUMN].mean() * 100),
    }

    # Verify chronological integrity assertion (strictly zero date overlap)
    assert_chronological_integrity(
        train_df["_flight_date_dt"],
        val_df["_flight_date_dt"],
        test_df["_flight_date_dt"],
    )

    return train_df, val_df, test_df, split_report


# ==============================================================================
# 3. Pre-Departure Feature Engineering
# ==============================================================================
def engineer_pre_departure_features(
    df: pd.DataFrame,
    mapping: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract features strictly available prior to scheduled departure.

    Permitted features:
    - Scheduled departure hour (0-23) and minute (0-59)
    - Cyclic time encodings: dep_hour_sin, dep_hour_cos, dep_min_sin, dep_min_cos
    - Calendar features: day_of_week (1-7), is_weekend (0/1), month (1-12)
    - Cyclic calendar encodings: day_sin, day_cos, month_sin, month_cos
    - Route identifier: ORIGIN + '_' + DEST
    - Planned duration & distance: distance, scheduled_elapsed_time (if present)
    - Categorical carrier: airline, origin, destination

    Excluded features (STRICT ANTI-LEAKAGE):
    - Actual departure time / delay, taxi, wheels, arrival time, arrival delay.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned flight records.
    mapping : dict
        Canonical column mapping.

    Returns
    -------
    tuple of (DataFrame, Series)
        X: Feature matrix containing only pre-departure attributes.
        y: Binary target series (is_delayed_15).
    """
    features = pd.DataFrame(index=df.index)

    # 1. Scheduled Departure Time & Cyclical Encodings
    if "scheduled_departure" in mapping:
        sched_dep = df[mapping["scheduled_departure"]]
        dep_hour = parse_hhmm_hour(sched_dep)
        
        # Numeric time for minute extraction
        numeric_dep = pd.to_numeric(sched_dep, errors="coerce")
        dep_min = (numeric_dep % 100).clip(0, 59).fillna(0)

        features["departure_hour"] = dep_hour
        features["departure_minute"] = dep_min

        # 24-hour cycle encoding
        features["dep_hour_sin"] = np.sin(2 * np.pi * dep_hour / 24.0)
        features["dep_hour_cos"] = np.cos(2 * np.pi * dep_hour / 24.0)

        # Minute-of-day cycle encoding (1440 minutes in 24 hours)
        minute_of_day = dep_hour * 60 + dep_min
        features["dep_minute_of_day_sin"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
        features["dep_minute_of_day_cos"] = np.cos(2 * np.pi * minute_of_day / 1440.0)

    # 2. Calendar & Cyclical Day/Month Encodings
    # Day of week
    if "day_of_week" in mapping and mapping["day_of_week"] in df.columns:
        dow = df[mapping["day_of_week"]].astype(int)
    elif "_flight_date_dt" in df.columns:
        # Day of week in pandas: Mon=0, Sun=6 -> shift to 1-7
        dow = df["_flight_date_dt"].dt.dayofweek + 1
    else:
        dow = pd.Series(1, index=df.index)

    features["day_of_week"] = dow
    features["is_weekend"] = dow.isin([6, 7]).astype(int)
    features["day_of_week_sin"] = np.sin(2 * np.pi * dow / 7.0)
    features["day_of_week_cos"] = np.cos(2 * np.pi * dow / 7.0)

    # Month
    if "month" in mapping and mapping["month"] in df.columns:
        month = df[mapping["month"]].astype(int)
    elif "_flight_date_dt" in df.columns:
        month = df["_flight_date_dt"].dt.month
    else:
        month = pd.Series(1, index=df.index)

    features["month"] = month
    features["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    features["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    # 3. Distance & Planned Flight Duration
    if "distance" in mapping and mapping["distance"] in df.columns:
        features["distance"] = pd.to_numeric(df[mapping["distance"]], errors="coerce")

    if "scheduled_elapsed_time" in mapping and mapping["scheduled_elapsed_time"] in df.columns:
        features["scheduled_elapsed_time"] = pd.to_numeric(df[mapping["scheduled_elapsed_time"]], errors="coerce")

    # 4. Categorical Identifiers
    if "airline" in mapping and mapping["airline"] in df.columns:
        features["airline"] = df[mapping["airline"]].astype(str)

    if "origin" in mapping and mapping["origin"] in df.columns:
        features["origin"] = df[mapping["origin"]].astype(str)

    if "destination" in mapping and mapping["destination"] in df.columns:
        features["destination"] = df[mapping["destination"]].astype(str)

    # Route feature (combination of ORIGIN and DEST)
    if "origin" in features.columns and "destination" in features.columns:
        features["route"] = features["origin"] + "_" + features["destination"]

    # 5. Extract target y
    if TARGET_COLUMN in df.columns:
        y = df[TARGET_COLUMN].copy()
    else:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in input dataframe.")

    # Run anti-leakage verification on constructed feature matrix
    assert_no_target_leakage(features.columns, LEAKAGE_COLUMNS, TARGET_COLUMN)

    return features, y


# ==============================================================================
# 4. Strict Training-Only Historical Prior Encoding (No Lookahead Bias)
# ==============================================================================
def fit_training_historical_priors(
    train_df: pd.DataFrame,
    group_cols: List[str] = ["airline", "origin", "destination", "route"],
    target_col: str = TARGET_COLUMN,
    smoothing: float = 20.0,
) -> Tuple[Dict[str, Dict[str, float]], float]:
    """
    Calculate empirical Bayesian smoothed historical delay rates EXCLUSIVELY on training data.

    Formula:
        prior = (n_delayed + smoothing * global_mean) / (n_total + smoothing)

    Parameters
    ----------
    train_df : pd.DataFrame
        Training set containing feature columns and the target.
    group_cols : list of str
        Categorical columns to compute priors for.
    target_col : str
        Target column name.
    smoothing : float
        M-estimate smoothing parameter to regularize small-sample categories.

    Returns
    -------
    tuple of (dict, float)
        priors_map: {col: {category: smoothed_prior}}
        global_mean: The overall training delay rate used for unseen categories.
    """
    global_mean = float(train_df[target_col].mean())
    priors_map: Dict[str, Dict[str, float]] = {}

    for col in group_cols:
        if col in train_df.columns:
            stats = train_df.groupby(col)[target_col].agg(["count", "sum"])
            smoothed_priors = (stats["sum"] + smoothing * global_mean) / (stats["count"] + smoothing)
            priors_map[col] = smoothed_priors.to_dict()

    return priors_map, global_mean


def apply_historical_priors(
    features_df: pd.DataFrame,
    priors_map: Dict[str, Dict[str, float]],
    global_mean: float,
) -> pd.DataFrame:
    """
    Apply training-derived historical priors onto a feature set.
    Unseen categories in validation or test data receive global_mean.
    """
    df_out = features_df.copy()
    for col, prior_dict in priors_map.items():
        if col in df_out.columns:
            prior_col_name = f"{col}_delay_prior"
            # Map known categories, filling unknown categories with global_mean
            df_out[prior_col_name] = df_out[col].map(prior_dict).fillna(global_mean).astype(float)

    return df_out


# ==============================================================================
# 5. Scikit-Learn Pipeline and ColumnTransformer
# ==============================================================================
def build_preprocessing_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """
    Build a scikit-learn ColumnTransformer for numerical and categorical pipelines.

    - Numeric: Median Imputer (+ Standard Scaler if requested).
    - Categorical: Constant Imputer ('UNKNOWN') + OneHotEncoder (handle_unknown='ignore').

    Parameters
    ----------
    numeric_features : list of str
        List of continuous and integer column names.
    categorical_features : list of str
        List of categorical column names.
    scale_numeric : bool
        Whether to apply StandardScaler to numerical features.

    Returns
    -------
    ColumnTransformer
        Configured scikit-learn preprocessor.
    """
    # Numeric Pipeline
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps=num_steps)

    # Categorical Pipeline (Strictly handles unseen categories at test time)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    min_frequency=25,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


# Alias for flexible import
build_column_transformer = build_preprocessing_pipeline


# ==============================================================================
# 6. Strict Anti-Leakage & Chronological Assertions
# ==============================================================================
def assert_no_target_leakage(
    feature_columns: List[str],
    leakage_columns: List[str],
    target_column: str = TARGET_COLUMN,
) -> None:
    """
    Verify that neither the target nor post-departure variables exist in feature columns.
    """
    feature_set = {str(c).strip().lower() for c in feature_columns}
    leakage_set = {str(c).strip().lower() for c in leakage_columns}

    # Test 1: Target cannot be in X
    if target_column.lower() in feature_set:
        raise AssertionError(f"CRITICAL TARGET LEAKAGE: Target '{target_column}' found in feature matrix X!")

    # Test 2: Banned leakage columns cannot be in X
    leaks_found = feature_set.intersection(leakage_set)
    if leaks_found:
        raise AssertionError(f"CRITICAL LEAKAGE DETECTED: Post-departure columns found in features: {leaks_found}")

    # Test 3: Banned substrings
    banned_substrings = ["actual", "arr_delay", "dep_delay", "arrival_delay", "departure_delay"]
    for feat in feature_set:
        for sub in banned_substrings:
            if sub in feat and "prior" not in feat:  # allow prior statistics (e.g. carrier_delay_prior)
                raise AssertionError(f"SUSPECT LEAKAGE: Feature '{feat}' contains prohibited substring '{sub}'!")


def assert_chronological_integrity(
    train_dates: pd.Series,
    val_dates: pd.Series,
    test_dates: pd.Series,
) -> None:
    """
    Assert that train, validation, and test sets are strictly ordered in time without overlap.
    """
    max_train = train_dates.max()
    min_val = val_dates.min()
    max_val = val_dates.max()
    min_test = test_dates.min()

    if max_train >= min_val:
        raise AssertionError(
            f"CHRONOLOGICAL LEAKAGE: Training dates ({max_train}) overlap with or exceed validation dates ({min_val})!"
        )

    if max_val >= min_test:
        raise AssertionError(
            f"CHRONOLOGICAL LEAKAGE: Validation dates ({max_val}) overlap with or exceed test dates ({min_test})!"
        )


def assert_no_null_targets(y_train: pd.Series, y_val: pd.Series, y_test: pd.Series) -> None:
    """
    Assert that ground truth targets contain no null values.
    """
    for name, s in [("train", y_train), ("val", y_val), ("test", y_test)]:
        null_count = s.isnull().sum()
        if null_count > 0:
            raise AssertionError(f"TARGET INTEGRITY ERROR: {name} target has {null_count} null entries!")


# ==============================================================================
# 7. Data Dictionary Generator
# ==============================================================================
def generate_data_dictionary(
    numeric_features: List[str],
    categorical_features: List[str],
    output_dir: Path = RESULTS_DIR,
) -> Dict[str, Any]:
    """
    Generate and persist comprehensive data dictionary for all features.
    """
    feature_meta = {
        # Numerical & Cyclical
        "departure_hour": {
            "type": "Integer (0-23)",
            "available_at_prediction": "Yes (Scheduled)",
            "description": "Hour of scheduled departure extracted from CRSDepTime.",
        },
        "departure_minute": {
            "type": "Integer (0-59)",
            "available_at_prediction": "Yes (Scheduled)",
            "description": "Minute of scheduled departure.",
        },
        "dep_hour_sin": {
            "type": "Float [-1, 1]",
            "available_at_prediction": "Yes (Scheduled)",
            "description": "Cyclic sine encoding of scheduled departure hour: sin(2*pi*hour/24).",
        },
        "dep_hour_cos": {
            "type": "Float [-1, 1]",
            "available_at_prediction": "Yes (Scheduled)",
            "description": "Cyclic cosine encoding of scheduled departure hour: cos(2*pi*hour/24).",
        },
        "dep_minute_of_day_sin": {
            "type": "Float [-1, 1]",
            "available_at_prediction": "Yes (Scheduled)",
            "description": "Cyclic sine encoding of minute-of-day: sin(2*pi*min/1440).",
        },
        "dep_minute_of_day_cos": {
            "type": "Float [-1, 1]",
            "available_at_prediction": "Yes (Scheduled)",
            "description": "Cyclic cosine encoding of minute-of-day: cos(2*pi*min/1440).",
        },
        "day_of_week": {
            "type": "Integer (1-7)",
            "available_at_prediction": "Yes (Calendar)",
            "description": "Day of week (1=Monday, 7=Sunday).",
        },
        "is_weekend": {
            "type": "Binary (0 or 1)",
            "available_at_prediction": "Yes (Calendar)",
            "description": "Flag indicating if flight is scheduled on Saturday or Sunday.",
        },
        "day_of_week_sin": {
            "type": "Float [-1, 1]",
            "available_at_prediction": "Yes (Calendar)",
            "description": "Cyclic sine encoding of day of week: sin(2*pi*dow/7).",
        },
        "day_of_week_cos": {
            "type": "Float [-1, 1]",
            "available_at_prediction": "Yes (Calendar)",
            "description": "Cyclic cosine encoding of day of week: cos(2*pi*dow/7).",
        },
        "month": {
            "type": "Integer (1-12)",
            "available_at_prediction": "Yes (Calendar)",
            "description": "Scheduled month of flight.",
        },
        "month_sin": {
            "type": "Float [-1, 1]",
            "available_at_prediction": "Yes (Calendar)",
            "description": "Cyclic sine encoding of month: sin(2*pi*month/12).",
        },
        "month_cos": {
            "type": "Float [-1, 1]",
            "available_at_prediction": "Yes (Calendar)",
            "description": "Cyclic cosine encoding of month: cos(2*pi*month/12).",
        },
        "distance": {
            "type": "Float / Miles",
            "available_at_prediction": "Yes (Planned)",
            "description": "Flight distance between origin and destination airports.",
        },
        "scheduled_elapsed_time": {
            "type": "Float / Minutes",
            "available_at_prediction": "Yes (Planned)",
            "description": "Scheduled planned flight duration from gate to gate.",
        },
        "airline_delay_prior": {
            "type": "Float [0, 1]",
            "available_at_prediction": "Yes (Historical prior from training set only)",
            "description": "Smoothed historical delay rate of the carrier calculated exclusively on earlier training flights.",
        },
        "origin_delay_prior": {
            "type": "Float [0, 1]",
            "available_at_prediction": "Yes (Historical prior from training set only)",
            "description": "Smoothed historical delay rate of the origin airport calculated exclusively on earlier training flights.",
        },
        "destination_delay_prior": {
            "type": "Float [0, 1]",
            "available_at_prediction": "Yes (Historical prior from training set only)",
            "description": "Smoothed historical delay rate of the destination airport calculated exclusively on earlier training flights.",
        },
        "route_delay_prior": {
            "type": "Float [0, 1]",
            "available_at_prediction": "Yes (Historical prior from training set only)",
            "description": "Smoothed historical delay rate of the route calculated exclusively on earlier training flights.",
        },
        # Categorical
        "airline": {
            "type": "Categorical String",
            "available_at_prediction": "Yes (Scheduled)",
            "description": "Operating airline code (e.g. AA, DL, UA, WN). One-hot encoded.",
        },
        "origin": {
            "type": "Categorical String",
            "available_at_prediction": "Yes (Scheduled)",
            "description": "Origin airport IATA code (e.g. ORD, ATL, DFW). One-hot encoded.",
        },
        "destination": {
            "type": "Categorical String",
            "available_at_prediction": "Yes (Scheduled)",
            "description": "Destination airport IATA code (e.g. LAX, JFK, SFO). One-hot encoded.",
        },
        "route": {
            "type": "Categorical String",
            "available_at_prediction": "Yes (Scheduled)",
            "description": "Route origin_dest string (e.g. ORD_LAX). High cardinality.",
        },
        # Target
        "is_delayed_15": {
            "type": "Binary (0 or 1)",
            "available_at_prediction": "NO (Target variable)",
            "description": "Supervised learning target: 1 if flight arrival delay >= 15 min, 0 otherwise.",
        },
    }

    # Save JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "data_dictionary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(feature_meta, f, indent=2)

    # Save Markdown table
    md_path = output_dir / "data_dictionary.md"
    rows = []
    rows.append("# Flight Delay Prediction: Data Dictionary\n")
    rows.append("| Feature Name | Data Type | Available at Prediction Time | Description |")
    rows.append("| :--- | :--- | :--- | :--- |")
    for feat, info in feature_meta.items():
        rows.append(f"| **`{feat}`** | {info['type']} | {info['available_at_prediction']} | {info['description']} |")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows))

    return feature_meta


# ==============================================================================
# 8. Diagnostic Benchmark Flight Data Generator
# Enables immediate verification of pipelines, encoders, and assertions
# ==============================================================================
def generate_diagnostic_flight_data(
    n_samples: int = 5000,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Generate authentic-format diagnostic flight data for verification.
    Spans Jan 1 to Mar 31 with realistic dates, carriers, and delay patterns.
    """
    rng = np.random.default_rng(random_state)

    carriers = ["AA", "DL", "UA", "WN", "B6", "AS", "NK"]
    airports = ["ORD", "ATL", "DFW", "DEN", "LAX", "JFK", "SFO", "SEA", "MIA", "BOS"]

    dates = pd.date_range("2022-01-01", "2022-03-31", periods=n_samples)

    # Scheduled times: 600 to 2200
    sched_dep_hours = rng.integers(6, 23, size=n_samples)
    sched_dep_mins = rng.integers(0, 60, size=n_samples)
    sched_dep = sched_dep_hours * 100 + sched_dep_mins

    # Distance and scheduled duration
    distances = rng.integers(200, 2600, size=n_samples)
    sched_times = (distances / 7.5 + rng.integers(40, 80, size=n_samples)).astype(int)

    # Generate delays with cascading evening increase
    evening_risk = (sched_dep_hours - 6) * 1.5
    raw_delays = rng.normal(loc=-3 + evening_risk, scale=28, size=n_samples).astype(int)

    # Cancellations (1.5%) and diversions (0.3%)
    cancelled = (rng.uniform(0, 1, size=n_samples) < 0.015).astype(int)
    diverted = (rng.uniform(0, 1, size=n_samples) < 0.003).astype(int)

    # Cancelled/diverted flights have NaN arrival delays
    arrival_delays = raw_delays.astype(float)
    arrival_delays[cancelled == 1] = np.nan
    arrival_delays[diverted == 1] = np.nan

    origin_sample = rng.choice(airports, size=n_samples)
    dest_sample = []
    for orig in origin_sample:
        available = [a for a in airports if a != orig]
        dest_sample.append(rng.choice(available))

    df = pd.DataFrame({
        "FlightDate": dates.strftime("%Y-%m-%d"),
        "Year": dates.year,
        "Month": dates.month,
        "Day": dates.day,
        "DayOfWeek": dates.dayofweek + 1,
        "Reporting_Airline": rng.choice(carriers, size=n_samples),
        "Origin": origin_sample,
        "Dest": dest_sample,
        "CRSDepTime": sched_dep,
        "CRSArrTime": (sched_dep + sched_times // 60 * 100) % 2400,
        "CRSElapsedTime": sched_times,
        "Distance": distances,
        "Cancelled": cancelled,
        "Diverted": diverted,
        "ArrDelay": arrival_delays,
        # Leakage columns to verify they get filtered out
        "DepTime": sched_dep + np.maximum(0, raw_delays),
        "DepDelay": raw_delays,
        "ArrTime": (sched_dep + sched_times + arrival_delays) % 2400,
        "TaxiOut": rng.integers(10, 30, size=n_samples),
        "AirTime": sched_times - 20,
    })

    mapping = {
        "date": "FlightDate",
        "year": "Year",
        "month": "Month",
        "day": "Day",
        "day_of_week": "DayOfWeek",
        "airline": "Reporting_Airline",
        "origin": "Origin",
        "destination": "Dest",
        "scheduled_departure": "CRSDepTime",
        "scheduled_arrival": "CRSArrTime",
        "scheduled_elapsed_time": "CRSElapsedTime",
        "distance": "Distance",
        "cancellation": "Cancelled",
        "diversion": "Diverted",
        "arrival_delay": "ArrDelay",
    }

    return df, mapping
