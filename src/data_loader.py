"""
Flight Delay Dataset Loader and Inspection Module
Phase 1: Traditional Machine Learning

This module provides tools to:
1. Discover flight delay datasets placed in `data/` or `data/raw/`.
2. Inspect dataset schemas without making assumptions about column names.
3. Map source-specific column names to canonical concepts.
4. Detect missing columns and flag data leakage features.
5. Provide detailed summary statistics (rows, missing values, duplicates, date ranges).
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Ensure project root is in sys.path so src can be imported both directly and as module
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

# Import project configuration
from src.config import (
    DATA_DIR,
    RAW_DATA_DIR,
    COLUMN_ALIASES,
    LEAKAGE_COLUMNS,
    TARGET_COLUMN,
    DELAY_THRESHOLD_MINUTES,
)


def find_dataset_files(search_paths: Optional[List[Path]] = None) -> List[Path]:
    """
    Search for dataset files (.csv, .parquet, .zip) in the data directories.

    Parameters
    ----------
    search_paths : list of Path, optional
        Custom list of directories to search. Defaults to RAW_DATA_DIR and DATA_DIR.

    Returns
    -------
    list of Path
        List of found dataset file paths.
    """
    if search_paths is None:
        search_paths = [RAW_DATA_DIR, DATA_DIR]

    supported_extensions = [".csv", ".parquet", ".zip"]
    found_files = []

    for directory in search_paths:
        if directory.exists():
            for ext in supported_extensions:
                found_files.extend(directory.glob(f"*{ext}"))

    # Return unique sorted file paths
    return sorted(list(set(found_files)))


def detect_column_mapping(columns: List[str]) -> Tuple[Dict[str, str], List[str]]:
    """
    Map dataset column headers to canonical concepts using COLUMN_ALIASES.

    Parameters
    ----------
    columns : list of str
        The actual column names present in the dataset.

    Returns
    -------
    tuple of (dict, list)
        detected_mapping: {canonical_concept: actual_column_name}
        missing_concepts: [canonical_concept, ...] that could not be matched.
    """
    column_lookup = {col.strip().lower(): col for col in columns}
    detected_mapping: Dict[str, str] = {}
    missing_concepts: List[str] = []

    for canonical_name, aliases in COLUMN_ALIASES.items():
        matched_col = None
        for alias in aliases:
            # Check exact match
            if alias in columns:
                matched_col = alias
                break
            # Check case-insensitive match
            elif alias.lower() in column_lookup:
                matched_col = column_lookup[alias.lower()]
                break

        if matched_col is not None:
            detected_mapping[canonical_name] = matched_col
        else:
            missing_concepts.append(canonical_name)

    return detected_mapping, missing_concepts


def audit_feature_leakage(columns: List[str]) -> Tuple[List[str], List[str]]:
    """
    Audit dataset columns to identify data leakage vs. safe predictive features.

    Rule: Never use actual departure or arrival information as a prediction feature.
    Any feature occurring after scheduled departure time constitutes data leakage.

    Parameters
    ----------
    columns : list of str
        List of columns in the dataset.

    Returns
    -------
    tuple of (list, list)
        leakage_detected: List of columns flagged as data leakage.
        safe_candidates: List of columns eligible for predictive modeling.
    """
    leakage_lookup = {col.lower() for col in LEAKAGE_COLUMNS}
    leakage_detected = []
    safe_candidates = []

    for col in columns:
        col_clean = col.strip()
        if col_clean.lower() in leakage_lookup or any(leak in col_clean.lower() for leak in [
            "actual", "dep_delay", "arr_delay", "taxi", "wheels", "air_time", "elapsed_time"
        ]):
            leakage_detected.append(col_clean)
        else:
            safe_candidates.append(col_clean)

    return leakage_detected, safe_candidates


def inspect_dataset(
    file_path: Path,
    sample_rows: Optional[int] = None
) -> Dict[str, Any]:
    """
    Comprehensive dataset inspection:
    - Row count, column count, memory usage
    - Data types
    - Missing values and percentage per column
    - Duplicate rows
    - Date range detection
    - Canonical concept verification

    Parameters
    ----------
    file_path : Path
        Path to the dataset file.
    sample_rows : int, optional
        Limit rows read for very large datasets (e.g. 5M+ rows). If None, reads full file.

    Returns
    -------
    dict
        Inspection results dictionary containing all statistics.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    # Determine file reading strategy
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(file_path, nrows=sample_rows, low_memory=False)
    elif suffix == ".parquet":
        df = pd.read_parquet(file_path)
        if sample_rows is not None and len(df) > sample_rows:
            df = df.head(sample_rows)
    elif suffix == ".zip":
        df = pd.read_csv(file_path, nrows=sample_rows, compression="zip", low_memory=False)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    total_rows, total_cols = df.shape
    columns = list(df.columns)

    # 1. Missing values
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / total_rows) * 100
    missing_summary = pd.DataFrame({
        "column": columns,
        "dtype": [str(df[col].dtype) for col in columns],
        "missing_count": missing_counts.values,
        "missing_pct": missing_pct.values
    }).sort_values(by="missing_pct", ascending=False).reset_index(drop=True)

    # 2. Duplicate rows
    duplicate_count = int(df.duplicated().sum())

    # 3. Canonical concept verification
    detected_mapping, missing_concepts = detect_column_mapping(columns)

    # 4. Data leakage audit
    leakage_cols, safe_cols = audit_feature_leakage(columns)

    # 5. Date range inspection
    date_range = None
    if "date" in detected_mapping:
        date_col = detected_mapping["date"]
        try:
            parsed_dates = pd.to_datetime(df[date_col].dropna(), errors="coerce")
            date_range = (parsed_dates.min().strftime("%Y-%m-%d"), parsed_dates.max().strftime("%Y-%m-%d"))
        except Exception:
            date_range = f"Could not parse dates from column {date_col}"
    elif {"year", "month", "day"}.issubset(detected_mapping.keys()):
        try:
            y = df[detected_mapping["year"]]
            m = df[detected_mapping["month"]]
            d = df[detected_mapping["day"]]
            sample_dates = pd.to_datetime(pd.DataFrame({"year": y, "month": m, "day": d}).dropna(), errors="coerce")
            date_range = (sample_dates.min().strftime("%Y-%m-%d"), sample_dates.max().strftime("%Y-%m-%d"))
        except Exception:
            date_range = f"Year: [{df[detected_mapping['year']].min()}, {df[detected_mapping['year']].max()}]"

    # 6. Cancellation and Diversion summary
    cancellation_summary = {}
    if "cancellation" in detected_mapping:
        canc_col = detected_mapping["cancellation"]
        cancellation_summary["total_cancelled"] = int((df[canc_col] == 1).sum())
        cancellation_summary["cancellation_rate_pct"] = float((df[canc_col] == 1).mean() * 100)

    if "diversion" in detected_mapping:
        div_col = detected_mapping["diversion"]
        cancellation_summary["total_diverted"] = int((df[div_col] == 1).sum())
        cancellation_summary["diversion_rate_pct"] = float((df[div_col] == 1).mean() * 100)

    # 7. Arrival delay summary & target definition check
    arrival_delay_summary = {}
    if "arrival_delay" in detected_mapping:
        arr_col = detected_mapping["arrival_delay"]
        valid_delays = df[arr_col].dropna()
        delayed_count = int((valid_delays >= DELAY_THRESHOLD_MINUTES).sum())
        arrival_delay_summary = {
            "mean_delay_minutes": float(valid_delays.mean()),
            "median_delay_minutes": float(valid_delays.median()),
            "min_delay_minutes": float(valid_delays.min()),
            "max_delay_minutes": float(valid_delays.max()),
            "delayed_15_count": delayed_count,
            "delayed_15_rate_pct": float((delayed_count / len(valid_delays)) * 100) if len(valid_delays) > 0 else 0.0,
        }

    return {
        "file_name": file_path.name,
        "file_path": str(file_path),
        "total_rows": total_rows,
        "total_columns": total_cols,
        "memory_mb": float(df.memory_usage(deep=True).sum() / (1024 * 1024)),
        "duplicate_rows": duplicate_count,
        "date_range": date_range,
        "column_mapping": detected_mapping,
        "missing_concepts": missing_concepts,
        "leakage_columns": leakage_cols,
        "safe_columns": safe_cols,
        "cancellation_summary": cancellation_summary,
        "arrival_delay_summary": arrival_delay_summary,
        "missing_summary_df": missing_summary,
        "sample_head": df.head(5)
    }


def load_clean_flight_data(
    file_path: Path,
    sample_rows: Optional[int] = None,
    filter_cancelled: bool = True,
    filter_diverted: bool = True
) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, Any]]:
    """
    Load dataset and apply documented operational filters for arrival delay modeling.

    Filtering rationale (anti-leakage and operational validity):
    1. Cancelled flights: Never departed/arrived; arrival delay is undefined (NaN).
       Cannot predict arrival delay for a flight that did not operate.
    2. Diverted flights: Arrived at an alternate airport; scheduled arrival dynamics do not apply.
    3. Missing arrival delay: Ground-truth target is missing, cannot train or evaluate.

    Parameters
    ----------
    file_path : Path
        Path to the dataset CSV/parquet.
    sample_rows : int, optional
        Sample limit for large datasets.
    filter_cancelled : bool
        Whether to exclude cancelled flights.
    filter_diverted : bool
        Whether to exclude diverted flights.

    Returns
    -------
    tuple of (DataFrame, dict, dict)
        df: Cleaned dataframe with canonical columns and binary target.
        mapping: Detected column mappings.
        audit_report: Counts of rows before and after each filter to prevent silent dropping.
    """
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        raw_df = pd.read_csv(file_path, nrows=sample_rows, low_memory=False)
    elif suffix == ".parquet":
        raw_df = pd.read_parquet(file_path)
        if sample_rows and len(raw_df) > sample_rows:
            raw_df = raw_df.head(sample_rows)
    else:
        raise ValueError(f"Unsupported format: {suffix}")

    mapping, missing = detect_column_mapping(raw_df.columns)
    audit = {"initial_rows": len(raw_df)}

    df = raw_df.copy()

    # Step 1: Filter cancelled flights
    if filter_cancelled and "cancellation" in mapping:
        canc_col = mapping["cancellation"]
        cancelled_mask = (df[canc_col] == 1)
        audit["cancelled_dropped"] = int(cancelled_mask.sum())
        df = df[~cancelled_mask]
    else:
        audit["cancelled_dropped"] = 0

    # Step 2: Filter diverted flights
    if filter_diverted and "diversion" in mapping:
        div_col = mapping["diversion"]
        diverted_mask = (df[div_col] == 1)
        audit["diverted_dropped"] = int(diverted_mask.sum())
        df = df[~diverted_mask]
    else:
        audit["diverted_dropped"] = 0

    # Step 3: Remove missing arrival delays (target ground truth)
    if "arrival_delay" in mapping:
        arr_col = mapping["arrival_delay"]
        missing_delay_mask = df[arr_col].isnull()
        audit["missing_arrival_delay_dropped"] = int(missing_delay_mask.sum())
        df = df[~missing_delay_mask].copy()

        # Step 4: Construct binary target is_delayed_15
        df[TARGET_COLUMN] = (df[arr_col] >= DELAY_THRESHOLD_MINUTES).astype(int)
    else:
        audit["missing_arrival_delay_dropped"] = 0

    audit["final_clean_rows"] = len(df)
    audit["retention_rate_pct"] = (len(df) / audit["initial_rows"]) * 100 if audit["initial_rows"] > 0 else 0.0

    return df, mapping, audit


def print_inspection_report(results: Dict[str, Any]) -> None:
    """
    Format and print a comprehensive inspection report to the console.
    """
    print("=" * 80)
    print(f" DATASET INSPECTION REPORT: {results['file_name']}")
    print("=" * 80)
    print(f"File Path        : {results['file_path']}")
    print(f"Total Rows       : {results['total_rows']:,}")
    print(f"Total Columns    : {results['total_columns']}")
    print(f"Memory Usage     : {results['memory_mb']:.2f} MB")
    print(f"Duplicate Rows   : {results['duplicate_rows']:,}")
    print(f"Date Range       : {results['date_range']}")

    print("\n" + "-" * 80)
    print(" 1. CANONICAL COLUMN MAPPING VERIFICATION")
    print("-" * 80)
    canonical_items = [
        ("Scheduled Departure", "scheduled_departure"),
        ("Scheduled Arrival", "scheduled_arrival"),
        ("Actual Departure (LEAKAGE)", "actual_departure"),
        ("Actual Arrival (LEAKAGE)", "actual_arrival"),
        ("Arrival Delay (TARGET SOURCE)", "arrival_delay"),
        ("Airline / Carrier", "airline"),
        ("Origin Airport", "origin"),
        ("Destination Airport", "destination"),
        ("Distance", "distance"),
        ("Cancellation Flag", "cancellation"),
        ("Diversion Flag", "diversion"),
    ]

    for label, concept in canonical_items:
        if concept in results["column_mapping"]:
            print(f"  [FOUND]    {label:<30} -> '{results['column_mapping'][concept]}'")
        else:
            print(f"  [MISSING]  {label:<30} -> Not detected in dataset")

    if results["missing_concepts"]:
        print(f"\nOther unmapped concepts: {', '.join(results['missing_concepts'])}")

    print("\n" + "-" * 80)
    print(" 2. DATA LEAKAGE AUDIT (Rule: Pre-departure features only)")
    print("-" * 80)
    print(f"Identified {len(results['leakage_columns'])} leakage/post-departure columns:")
    for leak_col in results["leakage_columns"]:
        print(f"  - ⚠️  {leak_col} (DO NOT USE AS FEATURE)")

    print(f"\nIdentified {len(results['safe_columns'])} pre-departure safe candidate columns:")
    for safe_col in results["safe_columns"][:10]:
        print(f"  - ✓  {safe_col}")
    if len(results["safe_columns"]) > 10:
        print(f"    ... and {len(results['safe_columns']) - 10} more.")

    if results["arrival_delay_summary"]:
        print("\n" + "-" * 80)
        print(" 3. TARGET VARIABLE DISTRIBUTION (Arrival Delay >= 15 min)")
        print("-" * 80)
        ads = results["arrival_delay_summary"]
        print(f"Mean Delay       : {ads['mean_delay_minutes']:.2f} minutes")
        print(f"Median Delay     : {ads['median_delay_minutes']:.2f} minutes")
        print(f"Delayed (>=15m)  : {ads['delayed_15_count']:,} flights ({ads['delayed_15_rate_pct']:.2f}%)")
        print(f"On-Time (<15m)   : {results['total_rows'] - ads['delayed_15_count']:,} flights ({100 - ads['delayed_15_rate_pct']:.2f}%)")

    print("\n" + "-" * 80)
    print(" 4. MISSING VALUES SUMMARY (Top 10)")
    print("-" * 80)
    print(results["missing_summary_df"].head(10).to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    print("Scanning data directories for dataset files...")
    dataset_files = find_dataset_files()

    if not dataset_files:
        print("\n" + "!" * 80)
        print(" NO DATASET FILES DETECTED IN data/ OR data/raw/")
        print("!" * 80)
        print("Please provide a flight dataset CSV file into:")
        print(f"  {RAW_DATA_DIR}/")
        print("\nCommon recommended options:")
        print("  1. Kaggle 2015 Flight Delays: flights.csv (and optional airlines.csv, airports.csv)")
        print("  2. US DOT BTS TranStats: Combined_Flights_2022.csv or similar")
        print("  3. Kaggle Flight Delay Dataset 2019-2023: flight_delay.csv")
        print("\nSee data/README.md for column specifications.")
        print("!" * 80)
        sys.exit(0)

    print(f"Found {len(dataset_files)} dataset file(s):")
    for f in dataset_files:
        print(f"  - {f.name} ({f.stat().st_size / (1024 * 1024):.1f} MB)")

    # Inspect the first / primary file
    primary_file = dataset_files[0]
    print(f"\nInspecting primary dataset: {primary_file.name}...")
    # Read sample of 100,000 rows if file is large (> 50 MB) for quick initial inspection
    file_size_mb = primary_file.stat().st_size / (1024 * 1024)
    sample_size = 100_000 if file_size_mb > 50 else None

    report = inspect_dataset(primary_file, sample_rows=sample_size)
    print_inspection_report(report)
