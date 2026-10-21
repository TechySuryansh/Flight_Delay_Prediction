"""
Offline Flight Rescheduling Simulator Module
Project 21: Flight Delay Prediction and Rescheduling — AML Phase 1
Author: Member 5 (PCA, Final Evaluation, Rescheduling Simulation, and Streamlit Frontend)

This module implements:
1. Operational pre-departure feature construction for hypothetical departure shifts.
2. Consistent time-derived feature updating (cyclical sine/cosine encodings, peak rush flags,
   time-of-day bins, arrival hour curfew, and duration calculations).
3. Seamless integration with Member 4's frozen classification pipeline (`models/classification_pipeline.joblib`).
4. Ranking of departure alternatives by predicted delay risk.
5. Strict anti-leakage compliance (uses only pre-departure attributes, zero post-departure fields).
6. Transparent operational disclaimers regarding associative vs. causal inferences.
"""

import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR, RANDOM_STATE
from src.features import (
    FEATURE_COLUMNS_CATEGORICAL,
    FEATURE_COLUMNS_NUMERIC,
    compute_empirical_bayes_priors,
    get_departure_time_of_day,
    get_distance_category,
)
from src.classification import ALL_CLASSIFICATION_FEATURES

# Suppress minor version pickle warnings for clean execution
warnings.filterwarnings("ignore", category=UserWarning)

# Operational Risk Tier Constants
RISK_TIER_LOW = "LOW RISK (On-Time / Minor)"
RISK_TIER_MODERATE = "MODERATE RISK (Monitor Turnaround)"
RISK_TIER_HIGH = "HIGH RISK (Delay Alert / Buffer Needed)"

DEFAULT_OPERATIONAL_THRESHOLD = 0.33

SIMULATION_DISCLAIMER = (
    "OPERATIONAL & SCIENTIFIC DISCLAIMER:\n"
    "1. Model-Based Scenario Estimate: All delay probabilities and rankings are offline "
    "statistical estimates derived from historical 2015 flight patterns. They do NOT guarantee "
    "actual on-time arrival or delay reduction.\n"
    "2. Non-Causal Association: The predictive model captures historical correlations (e.g., lower "
    "congestion in early morning hours) and does NOT establish that rescheduling a flight causes lower "
    "operational delay risk.\n"
    "3. No Operational Availability Guarantee: This simulation does NOT check live airline seat "
    "inventory, air traffic control slot clearance, aircraft tail routing, or airline scheduling approval."
)


class OfflineReschedulingSimulator:
    """
    Simulates hypothetical departure-time alternatives for scheduled flights
    using Member 4's frozen classification pipeline.
    """

    def __init__(
        self,
        artifact_path: Optional[Union[str, Path]] = None,
        data_path: Optional[Union[str, Path]] = None,
        operational_threshold: float = DEFAULT_OPERATIONAL_THRESHOLD,
    ):
        """
        Initialize the simulator with the saved classification pipeline and historical priors.

        Parameters
        ----------
        artifact_path : str or Path, optional
            Path to `models/classification_pipeline.joblib`.
        data_path : str or Path, optional
            Path to `data/processed/flights_features.csv` for computing historical priors.
        operational_threshold : float
            Classification operating threshold (default: 0.33).
        """
        if artifact_path is None:
            artifact_path = PROJECT_ROOT / "models" / "classification_pipeline.joblib"
        self.artifact_path = Path(artifact_path)

        if data_path is None:
            data_path = PROCESSED_DATA_DIR / "flights_features.csv"
        self.data_path = Path(data_path)

        self.operational_threshold = operational_threshold

        # Load model artifact
        self._load_artifact()

        # Load or compute historical priors
        self._load_priors()

    def _load_artifact(self) -> None:
        """Load and validate the serialized classification artifact."""
        if not self.artifact_path.exists():
            raise FileNotFoundError(
                f"Classification pipeline artifact not found at: {self.artifact_path}\n"
                "Please run Member 4's notebook (05_classification_models.ipynb) or run "
                "`python -m src.classification` to build the artifact."
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            artifact = joblib.load(self.artifact_path)

        self.pipeline = artifact["pipeline"]
        self.model = artifact["model"]
        self.expected_features = artifact.get("expected_features", ALL_CLASSIFICATION_FEATURES)
        self.metadata = artifact.get("metadata", {})

    def _load_priors(self) -> None:
        """Compute training set Empirical Bayes historical delay priors for airlines and airports."""
        self.carrier_priors: Dict[str, float] = {}
        self.origin_priors: Dict[str, float] = {}
        self.dest_priors: Dict[str, float] = {}
        self.global_delay_mean: float = 0.3863

        if self.data_path.exists():
            try:
                df = pd.read_csv(self.data_path)
                train_df = df[df["SPLIT_SET"] == "train"]
                if len(train_df) > 0:
                    self.carrier_priors, self.global_delay_mean = compute_empirical_bayes_priors(
                        train_df, "AIRLINE"
                    )
                    self.origin_priors, _ = compute_empirical_bayes_priors(train_df, "ORIGIN_AIRPORT")
                    self.dest_priors, _ = compute_empirical_bayes_priors(train_df, "DESTINATION_AIRPORT")
                    return
            except Exception as e:
                warnings.warn(f"Could not compute priors from {self.data_path}: {e}")

        # Fallback default historical priors if dataset is not on disk
        self.global_delay_mean = 0.3863
        self.carrier_priors = {
            "AA": 0.3878, "AS": 0.3692, "B6": 0.3757, "DL": 0.4088,
            "EV": 0.4015, "F9": 0.3800, "HA": 0.3739, "MQ": 0.3793,
            "NK": 0.3683, "OO": 0.3866, "UA": 0.3883, "US": 0.4048,
            "VX": 0.4032, "WN": 0.3820,
        }

    def get_historical_priors(
        self, airline: str, origin: str, dest: str
    ) -> Tuple[float, float, float]:
        """Lookup historical delay rates with fallback to global training mean."""
        c_prior = self.carrier_priors.get(str(airline).upper(), self.global_delay_mean)
        o_prior = self.origin_priors.get(str(origin).upper(), self.global_delay_mean)
        d_prior = self.dest_priors.get(str(dest).upper(), self.global_delay_mean)
        return float(c_prior), float(o_prior), float(d_prior)

    def build_flight_feature_record(
        self,
        departure_dt: datetime,
        airline: str,
        origin: str,
        destination: str,
        distance: float,
        scheduled_time_minutes: float,
    ) -> Dict[str, Any]:
        """
        Construct a single complete feature dictionary containing all 36 required pre-departure features.

        Parameters
        ----------
        departure_dt : datetime
            Scheduled departure date and time.
        airline : str
            IATA airline code (e.g. 'AA', 'DL', 'UA').
        origin : str
            Origin airport 3-letter IATA code (e.g. 'ORD').
        destination : str
            Destination airport 3-letter IATA code (e.g. 'DFW').
        distance : float
            Planned flight distance in statute miles.
        scheduled_time_minutes : float
            Scheduled block duration in minutes.

        Returns
        -------
        dict
            Dictionary containing exact 36 model features.
        """
        # Temporal departure encodings
        dep_hour = departure_dt.hour
        dep_minute = departure_dt.minute
        dep_time_minutes = dep_hour * 60 + dep_minute

        dep_hour_sin = np.sin(2 * np.pi * dep_hour / 24.0)
        dep_hour_cos = np.cos(2 * np.pi * dep_hour / 24.0)
        dep_time_sin = np.sin(2 * np.pi * dep_time_minutes / 1440.0)
        dep_time_cos = np.cos(2 * np.pi * dep_time_minutes / 1440.0)
        dep_hour_peak = 1 if (17 <= dep_hour <= 20) else 0

        # Arrival timing calculation based on scheduled flight duration
        arrival_dt = departure_dt + timedelta(minutes=float(scheduled_time_minutes))
        arr_hour = arrival_dt.hour
        arr_minute = arrival_dt.minute
        arr_time_minutes = arr_hour * 60 + arr_minute

        arr_hour_sin = np.sin(2 * np.pi * arr_hour / 24.0)
        arr_hour_cos = np.cos(2 * np.pi * arr_hour / 24.0)
        arr_night_curfew = 1 if (arr_hour >= 23 or arr_hour < 6) else 0

        # Calendar features
        day_of_week = departure_dt.isoweekday()  # 1 = Mon, 7 = Sun
        dow_sin = np.sin(2 * np.pi * (day_of_week - 1) / 7.0)
        dow_cos = np.cos(2 * np.pi * (day_of_week - 1) / 7.0)
        is_weekend = 1 if day_of_week in [6, 7] else 0

        month = departure_dt.month
        month_sin = np.sin(2 * np.pi * (month - 1) / 12.0)
        month_cos = np.cos(2 * np.pi * (month - 1) / 12.0)
        day = departure_dt.day
        quarter = (month - 1) // 3 + 1
        is_month_end = 1 if day >= 28 else 0

        # Flight geometry & planned speed proxy
        distance = max(50.0, float(distance))
        distance_log = float(np.log1p(distance))
        scheduled_time_minutes = max(20.0, float(scheduled_time_minutes))
        planned_speed = (distance / scheduled_time_minutes) * 60.0

        # Empirical Bayes historical delay priors
        c_prior, o_prior, d_prior = self.get_historical_priors(airline, origin, destination)

        # Categoricals
        dep_time_of_day = get_departure_time_of_day(dep_hour)
        dist_category = get_distance_category(distance)

        record = {
            # 31 Numerics
            "DEP_HOUR": float(dep_hour),
            "DEP_MINUTE": float(dep_minute),
            "DEP_TIME_MINUTES": float(dep_time_minutes),
            "DEP_HOUR_SIN": float(dep_hour_sin),
            "DEP_HOUR_COS": float(dep_hour_cos),
            "DEP_TIME_SIN": float(dep_time_sin),
            "DEP_TIME_COS": float(dep_time_cos),
            "DEP_HOUR_PEAK": float(dep_hour_peak),
            "ARR_HOUR": float(arr_hour),
            "ARR_MINUTE": float(arr_minute),
            "ARR_TIME_MINUTES": float(arr_time_minutes),
            "ARR_HOUR_SIN": float(arr_hour_sin),
            "ARR_HOUR_COS": float(arr_hour_cos),
            "ARR_NIGHT_CURFEW": float(arr_night_curfew),
            "DAY_OF_WEEK": float(day_of_week),
            "DOW_SIN": float(dow_sin),
            "DOW_COS": float(dow_cos),
            "IS_WEEKEND": float(is_weekend),
            "MONTH": float(month),
            "MONTH_SIN": float(month_sin),
            "MONTH_COS": float(month_cos),
            "DAY": float(day),
            "QUARTER": float(quarter),
            "IS_MONTH_END": float(is_month_end),
            "DISTANCE": float(distance),
            "DISTANCE_LOG": float(distance_log),
            "SCHEDULED_TIME": float(scheduled_time_minutes),
            "PLANNED_SPEED": float(planned_speed),
            "CARRIER_HIST_DELAY_RATE": float(c_prior),
            "ORIGIN_HIST_DELAY_RATE": float(o_prior),
            "DEST_HIST_DELAY_RATE": float(d_prior),
            # 5 Categoricals
            "AIRLINE": str(airline).strip().upper(),
            "ORIGIN_AIRPORT": str(origin).strip().upper(),
            "DESTINATION_AIRPORT": str(destination).strip().upper(),
            "DEP_TIME_OF_DAY": str(dep_time_of_day),
            "DISTANCE_CATEGORY": str(dist_category),
        }
        return record

    def predict_delay_probability(self, record_or_df: Union[Dict[str, Any], pd.DataFrame]) -> np.ndarray:
        """
        Pass feature record(s) through Member 4's preprocessing pipeline and model to get P(Delayed).

        Parameters
        ----------
        record_or_df : dict or pd.DataFrame
            Single flight record or DataFrame of multiple flights.

        Returns
        -------
        np.ndarray
            Array of delay probabilities in [0.0, 1.0].
        """
        if isinstance(record_or_df, dict):
            df = pd.DataFrame([record_or_df])
        else:
            df = record_or_df.copy()

        # Align columns to exact expected features
        X_input = df[self.expected_features]

        # Execute transformation and prediction
        X_trans = self.pipeline.transform(X_input)
        probabilities = self.model.predict_proba(X_trans)[:, 1]
        return probabilities

    def simulate_rescheduling_options(
        self,
        base_departure_dt: datetime,
        airline: str,
        origin: str,
        destination: str,
        distance: float,
        scheduled_time_minutes: float,
        time_shifts_minutes: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Generate and rank hypothetical departure-time alternatives around a baseline scheduled flight.

        Parameters
        ----------
        base_departure_dt : datetime
            Original scheduled departure time.
        airline : str
            Airline carrier code (e.g., 'AA', 'DL').
        origin : str
            Origin airport code.
        destination : str
            Destination airport code.
        distance : float
            Planned distance in miles.
        scheduled_time_minutes : float
            Scheduled block duration in minutes.
        time_shifts_minutes : list of int, optional
            List of shift intervals in minutes (e.g. [-120, -60, -30, 0, 30, 60, 120]).
            Defaults to [-120, -90, -60, -30, 0, 30, 60, 90, 120].

        Returns
        -------
        pd.DataFrame
            Ranked simulation dataframe containing original option and alternatives.
        """
        if time_shifts_minutes is None:
            time_shifts_minutes = [-120, -90, -60, -30, 0, 30, 60, 90, 120]

        # Ensure 0 (baseline) is in the list
        if 0 not in time_shifts_minutes:
            time_shifts_minutes.append(0)
        time_shifts_minutes = sorted(list(set(time_shifts_minutes)))

        records = []
        metadata_rows = []

        for shift in time_shifts_minutes:
            hypo_dt = base_departure_dt + timedelta(minutes=int(shift))
            record = self.build_flight_feature_record(
                departure_dt=hypo_dt,
                airline=airline,
                origin=origin,
                destination=destination,
                distance=distance,
                scheduled_time_minutes=scheduled_time_minutes,
            )
            records.append(record)

            hypo_arr_dt = hypo_dt + timedelta(minutes=float(scheduled_time_minutes))
            is_baseline = (shift == 0)

            metadata_rows.append({
                "shift_minutes": shift,
                "is_baseline": is_baseline,
                "option_type": "BASELINE (Original)" if is_baseline else f"Shift {shift:+d} min",
                "scheduled_departure": hypo_dt.strftime("%Y-%m-%d %H:%M"),
                "scheduled_arrival": hypo_arr_dt.strftime("%Y-%m-%d %H:%M"),
                "dep_time_of_day": record["DEP_TIME_OF_DAY"],
                "is_peak_rush": bool(record["DEP_HOUR_PEAK"] == 1),
                "is_night_curfew": bool(record["ARR_NIGHT_CURFEW"] == 1),
                "airline": airline,
                "route": f"{origin} -> {destination}",
            })

        df_features = pd.DataFrame(records)
        probabilities = self.predict_delay_probability(df_features)

        results_df = pd.DataFrame(metadata_rows)
        results_df["delay_probability"] = probabilities
        results_df["predicted_delayed_flag"] = (probabilities >= self.operational_threshold).astype(int)

        # Risk tiering
        results_df["risk_tier"] = pd.cut(
            probabilities,
            bins=[-0.01, self.operational_threshold, 0.50, 1.01],
            labels=[RISK_TIER_LOW, RISK_TIER_MODERATE, RISK_TIER_HIGH],
        )

        # Baseline probability for delta computation
        baseline_mask = results_df["is_baseline"]
        baseline_prob = float(results_df.loc[baseline_mask, "delay_probability"].iloc[0])

        results_df["prob_delta_vs_baseline"] = results_df["delay_probability"] - baseline_prob
        results_df["prob_delta_pct_pts"] = results_df["prob_delta_vs_baseline"] * 100.0

        # Rank options (1 = lowest delay probability)
        results_df["rank"] = results_df["delay_probability"].rank(method="dense", ascending=True).astype(int)

        # Sort: baseline first or by rank
        results_df = results_df.sort_values(by=["rank", "shift_minutes"]).reset_index(drop=True)
        return results_df

    @staticmethod
    def get_simulation_disclaimer() -> str:
        """Return the official operational disclaimer for frontend and reports."""
        return SIMULATION_DISCLAIMER


# ==============================================================================
# Self-Test Execution
# ==============================================================================

if __name__ == "__main__":
    print("=" * 78)
    print("FLIGHT DELAY RESCHEDULING SIMULATION — MODULE SELF-TEST")
    print("=" * 78)

    simulator = OfflineReschedulingSimulator()
    print("✓ Rescheduling Simulator initialized successfully.")
    print(f"✓ Pipeline model loaded: {simulator.model.__class__.__name__}")
    print(f"✓ Operational alert threshold: {simulator.operational_threshold}")
    print(f"✓ Carrier priors loaded: {len(simulator.carrier_priors)} airlines")

    test_flight = {
        "base_departure_dt": datetime(2015, 10, 15, 18, 30),
        "airline": "AA",
        "origin": "ORD",
        "destination": "DFW",
        "distance": 802.0,
        "scheduled_time_minutes": 155.0,
        "time_shifts_minutes": [-120, -60, -30, 0, 30, 60, 120],
    }

    print(f"\nSimulating scenario for flight: {test_flight['airline']} {test_flight['origin']} -> {test_flight['destination']}")
    print(f"Baseline Scheduled Departure: {test_flight['base_departure_dt'].strftime('%Y-%m-%d %H:%M')}")

    results = simulator.simulate_rescheduling_options(**test_flight)

    print("\nSimulation Results (Ranked by Delay Probability):")
    cols_to_print = [
        "rank", "option_type", "scheduled_departure", "delay_probability",
        "prob_delta_pct_pts", "risk_tier", "predicted_delayed_flag"
    ]
    formatted_df = results[cols_to_print].copy()
    formatted_df["delay_probability"] = formatted_df["delay_probability"].map(lambda x: f"{x:.2%}")
    formatted_df["prob_delta_pct_pts"] = formatted_df["prob_delta_pct_pts"].map(lambda x: f"{x:+.2f}%")
    print(formatted_df.to_string(index=False))

    print("\n" + "-" * 78)
    print(simulator.get_simulation_disclaimer())
    print("=" * 78)
