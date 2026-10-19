"""
Utility functions for Exploratory Data Analysis and Feature Preprocessing
Flight Delay Prediction - Phase 1: Machine Learning
"""

from pathlib import Path
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_hhmm_hour(time_series: pd.Series) -> pd.Series:
    """
    Convert scheduled departure/arrival time from HHMM format (int/float/str)
    into an integer hour (0-23).

    In aviation datasets (e.g. BTS, Kaggle):
      - 830 -> 8 (08:30)
      - 1745 -> 17 (17:45)
      - 45 -> 0 (00:45)
      - 2400 -> 0 (Midnight rollover)
      - Missing or invalid -> NaN

    Parameters
    ----------
    time_series : pd.Series
        Raw scheduled time values.

    Returns
    -------
    pd.Series
        Clean integer hours between 0 and 23.
    """
    # Convert to numeric, coercing errors to NaN
    numeric_time = pd.to_numeric(time_series, errors="coerce")
    
    # In integer division, 830 // 100 = 8, 1745 // 100 = 17
    hours = (numeric_time // 100).fillna(-1).astype(int)
    
    # Handle midnight rollover: 2400 -> 0
    hours = hours.replace(24, 0)
    
    # Mask invalid hours outside [0, 23] back to NaN
    return hours.where((hours >= 0) & (hours <= 23), other=np.nan)


def compute_group_delay_stats(
    df: pd.DataFrame,
    group_col: str,
    target_col: str = "is_delayed_15",
    min_volume: int = 50
) -> pd.DataFrame:
    """
    Compute total flight volume, delayed count, and delay rate (%) for groups.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the group column and binary delay target.
    group_col : str
        The categorical column to group by (e.g. 'AIRLINE', 'ORIGIN_AIRPORT').
    target_col : str
        The binary target column (1 = delayed >= 15 min, 0 = on-time).
    min_volume : int
        Minimum flights threshold to filter out low-sample anomalies.

    Returns
    -------
    pd.DataFrame
        Aggregated dataframe sorted by flight volume or delay rate.
    """
    if group_col not in df.columns or target_col not in df.columns:
        return pd.DataFrame()

    stats = df.groupby(group_col).agg(
        total_flights=(target_col, "count"),
        delayed_flights=(target_col, "sum"),
        delay_rate_pct=(target_col, lambda s: s.mean() * 100)
    ).reset_index()

    if min_volume > 0:
        stats = stats[stats["total_flights"] >= min_volume]

    return stats.sort_values(by="total_flights", ascending=False).reset_index(drop=True)


def save_eda_figure(
    fig: plt.Figure,
    file_name: str,
    output_dir: Path,
    dpi: int = 300
) -> Path:
    """
    Safely save a matplotlib figure to the results/figures directory.

    Parameters
    ----------
    fig : plt.Figure
        Matplotlib figure object.
    file_name : str
        File name (e.g. '02_delay_by_airline.png').
    output_dir : Path
        Target directory (e.g. FIGURES_DIR).
    dpi : int
        Resolution in dots per inch.

    Returns
    -------
    Path
        Absolute path of the saved figure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / file_name
    fig.tight_layout()
    fig.savefig(target_path, dpi=dpi, bbox_inches="tight")
    return target_path
