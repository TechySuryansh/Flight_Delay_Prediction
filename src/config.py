"""
Project Configuration and Canonical Schema Mappings
Flight Delay Prediction - Phase 1: Machine Learning
"""

from pathlib import Path

# ==============================================================================
# 1. Project Directory Paths
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SRC_DIR = PROJECT_ROOT / "src"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

# Ensure output directories exist
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, METRICS_DIR]:
    path.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 2. Canonical Column Mapping Aliases
# Used by data_loader.py to automatically recognize column headers from
# various standard sources (e.g. BTS TranStats, Kaggle 2015, Kaggle 2019-2023)
# ==============================================================================
COLUMN_ALIASES = {
    "scheduled_departure": [
        "SCHEDULED_DEPARTURE", "CRSDepTime", "crs_dep_time", "CRS_DEP_TIME",
        "SchedDepTime", "scheduled_departure", "scheduled_dep_time"
    ],
    "scheduled_arrival": [
        "SCHEDULED_ARRIVAL", "CRSArrTime", "crs_arr_time", "CRS_ARR_TIME",
        "SchedArrTime", "scheduled_arrival", "scheduled_arr_time"
    ],
    "actual_departure": [
        "DEPARTURE_TIME", "DepTime", "dep_time", "DEP_TIME",
        "actual_departure", "actual_dep_time"
    ],
    "actual_arrival": [
        "ARRIVAL_TIME", "ArrTime", "arr_time", "ARR_TIME",
        "actual_arrival", "actual_arr_time"
    ],
    "arrival_delay": [
        "ARRIVAL_DELAY", "ArrDelay", "arr_delay", "ARR_DELAY",
        "ArrDelayMinutes", "arrival_delay", "arrival_delay_minutes"
    ],
    "departure_delay": [
        "DEPARTURE_DELAY", "DepDelay", "dep_delay", "DEP_DELAY",
        "DepDelayMinutes", "departure_delay"
    ],
    "cancellation": [
        "CANCELLED", "Cancelled", "cancelled", "is_cancelled", "CANCELLATION"
    ],
    "diversion": [
        "DIVERTED", "Diverted", "diverted", "is_diverted"
    ],
    "airline": [
        "AIRLINE", "OP_UNIQUE_CARRIER", "Reporting_Airline", "IATA_CODE_REPORTING_AIRLINE",
        "Carrier", "carrier", "airline", "UNIQUE_CARRIER"
    ],
    "origin": [
        "ORIGIN_AIRPORT", "ORIGIN", "Origin", "origin", "OriginAirport", "origin_airport"
    ],
    "destination": [
        "DESTINATION_AIRPORT", "DEST", "Dest", "dest", "destination",
        "DestAirport", "destination_airport"
    ],
    "distance": [
        "DISTANCE", "Distance", "distance"
    ],
    "scheduled_elapsed_time": [
        "SCHEDULED_TIME", "CRSElapsedTime", "crs_elapsed_time", "CRS_ELAPSED_TIME", "scheduled_time"
    ],
    "date": [
        "FL_DATE", "FlightDate", "flight_date", "Date", "date"
    ],
    "year": ["YEAR", "Year", "year"],
    "month": ["MONTH", "Month", "month"],
    "day": ["DAY", "Day", "day", "DayofMonth", "DAY_OF_MONTH"],
    "day_of_week": ["DAY_OF_WEEK", "DayOfWeek", "day_of_week"]
}

# ==============================================================================
# 3. Data Leakage Prevention Lists
# CRITICAL: These columns represent events occurring AFTER scheduled departure.
# They MUST NEVER be passed as input features into predictive models.
# ==============================================================================
LEAKAGE_COLUMNS = [
    # Actual departure occurrences
    "DEPARTURE_TIME", "DepTime", "dep_time", "DEP_TIME", "actual_departure",
    "DEPARTURE_DELAY", "DepDelay", "dep_delay", "DEP_DELAY", "DepDel15",
    "TAXI_OUT", "TaxiOut", "taxi_out",
    "WHEELS_OFF", "WheelsOff", "wheels_off",
    
    # In-flight and actual arrival occurrences
    "ARRIVAL_TIME", "ArrTime", "arr_time", "ARR_TIME", "actual_arrival",
    "ELAPSED_TIME", "ActualElapsedTime", "actual_elapsed_time",
    "AIR_TIME", "AirTime", "air_time",
    "WHEELS_ON", "WheelsOn", "wheels_on",
    "TAXI_IN", "TaxiIn", "taxi_in",
    
    # Post-hoc delay cause breakdowns
    "AIR_SYSTEM_DELAY", "NASDelay", "nas_delay",
    "SECURITY_DELAY", "SecurityDelay", "security_delay",
    "AIRLINE_DELAY", "CarrierDelay", "carrier_delay",
    "LATE_AIRCRAFT_DELAY", "LateAircraftDelay", "late_aircraft_delay",
    "WEATHER_DELAY", "WeatherDelay", "weather_delay",
    "CANCELLATION_REASON", "CancellationCode"
]

# ==============================================================================
# 4. Modeling Parameters (Phase 1)
# ==============================================================================
TARGET_COLUMN = "is_delayed_15"
DELAY_THRESHOLD_MINUTES = 15
RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.10
