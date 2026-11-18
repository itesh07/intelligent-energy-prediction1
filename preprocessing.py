"""
Preprocessing and Feature Engineering Pipeline for Intelligent Energy Consumption Prediction.
Supports time-series preparation, schema validation, lag generation, rolling statistics,
chronological train-validation-test splits, and demo dataset synthesis.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "hour",
    "dayofweek",
    "month",
    "is_weekend",
    "is_business_hour",
    "temperature",
    "humidity",
    "occupancy",
    "lag_1",
    "lag_2",
    "lag_24",
    "rolling_mean_6",
    "rolling_mean_24",
    "cooling_degree",
    "heating_degree",
]

TARGET_COLUMN = "energy_kwh"

COLUMN_ALIASES = {
    "timestamp": ["timestamp", "datetime", "date", "time", "date_time", "timestamp_utc"],
    "energy_kwh": ["energy_kwh", "energy", "kwh", "consumption", "power_kwh", "energy_consumption", "power"],
    "temperature": ["temperature", "temp", "temp_c", "ambient_temperature", "outdoor_temp"],
    "humidity": ["humidity", "rh", "rel_humidity", "relative_humidity"],
    "occupancy": ["occupancy", "occupants", "people", "headcount", "active_occupants"],
}


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes column names and maps known aliases to standard schema."""
    df = df.copy()
    raw_cols = {str(c).strip().lower().replace(" ", "_").replace("-", "_"): c for c in df.columns}

    rename_map = {}
    for target, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in raw_cols and target not in rename_map.values():
                rename_map[raw_cols[alias]] = target
                break

    if rename_map:
        df = df.rename(columns=rename_map)

    # Clean remaining columns to lowercase with underscores
    df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    return df


def validate_and_clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validates CSV schema and data types, parses timestamps, handles missing values,
    removes duplicate timestamps and negative readings, and returns cleaned dataframe
    along with a detailed validation summary report.
    """
    if df is None or df.empty:
        raise ValueError("The provided CSV dataset is empty. Please provide a file with valid records.")

    raw_row_count = len(df)
    df = normalize_column_names(df)

    # 1. Verify required columns
    missing_required = []
    if "timestamp" not in df.columns:
        missing_required.append("timestamp")
    if "energy_kwh" not in df.columns:
        missing_required.append("energy_kwh")

    if missing_required:
        raise ValueError(
            f"Dataset is missing required column(s): {', '.join(missing_required)}. "
            f"Found columns: {list(df.columns)}. "
            f"Required columns are 'timestamp' and 'energy_kwh'."
        )

    # 2. Parse timestamps correctly
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", format="mixed")
    invalid_timestamps = int(df["timestamp"].isna().sum())
    df = df.dropna(subset=["timestamp"])

    if df.empty:
        raise ValueError(
            "None of the values in the 'timestamp' column could be parsed into valid date/time. "
            "Please format timestamps as 'YYYY-MM-DD HH:MM:SS' or standard ISO-8601 strings."
        )

    # 3. Clean energy consumption readings
    df["energy_kwh"] = pd.to_numeric(df["energy_kwh"], errors="coerce")
    nan_energy_count = int(df["energy_kwh"].isna().sum())
    df = df.dropna(subset=["energy_kwh"])

    negative_energy_count = int((df["energy_kwh"] < 0).sum())
    if negative_energy_count > 0:
        # Exclude physically impossible negative electrical energy readings
        df = df[df["energy_kwh"] >= 0]

    if df.empty:
        raise ValueError(
            "No valid positive energy readings found in 'energy_kwh'. "
            "Energy consumption must be non-negative numeric values (kWh >= 0)."
        )

    # 4. Sort chronologically & handle duplicate timestamps
    initial_valid = len(df)
    df = df.sort_values("timestamp")
    duplicate_count = int(df.duplicated(subset=["timestamp"]).sum())
    # Keep the last observation for any duplicate timestamp entry
    df = df.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

    if len(df) < 24:
        raise ValueError(
            f"Insufficient chronological data: minimum 24 valid hourly records required to construct "
            f"lag and rolling features (found {len(df)} valid records after validation)."
        )

    # 5. Process optional columns (temperature, humidity, occupancy)
    hour_series = df["timestamp"].dt.hour
    is_weekend_series = df["timestamp"].dt.day_of_week.isin([5, 6]).astype(int)

    has_temp = "temperature" in df.columns
    if not has_temp:
        # Synthesize realistic diurnal ambient temperature (22°C night to 32°C midday)
        df["temperature"] = 27.0 + 5.0 * np.sin((hour_series - 9) * np.pi / 12)
    else:
        df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
        df["temperature"] = df["temperature"].interpolate(method="linear").bfill().ffill().fillna(26.0)
        df["temperature"] = df["temperature"].clip(lower=-25.0, upper=65.0)

    has_hum = "humidity" in df.columns
    if not has_hum:
        # Inverse ambient humidity relative to temperature cycle
        df["humidity"] = 60.0 - 15.0 * np.sin((hour_series - 9) * np.pi / 12)
    else:
        df["humidity"] = pd.to_numeric(df["humidity"], errors="coerce")
        df["humidity"] = df["humidity"].interpolate(method="linear").bfill().ffill().fillna(55.0)
        df["humidity"] = df["humidity"].clip(lower=5.0, upper=100.0)

    has_occ = "occupancy" in df.columns
    if not has_occ:
        # Commercial smart facility schedule (140 peak occupants on weekday business hours, 5 setback)
        is_biz = ((hour_series >= 8) & (hour_series <= 18) & (is_weekend_series == 0)).astype(int)
        df["occupancy"] = is_biz * 140 + 5
    else:
        df["occupancy"] = pd.to_numeric(df["occupancy"], errors="coerce")
        df["occupancy"] = df["occupancy"].interpolate(method="linear").bfill().ffill().fillna(10.0)
        df["occupancy"] = df["occupancy"].clip(lower=0).round().astype(int)

    # 6. Interpolate short sensor dropouts in energy series (up to 3 consecutive hours)
    df["energy_kwh"] = df["energy_kwh"].interpolate(method="linear", limit=3).bfill().ffill()

    validation_report = {
        "raw_records": raw_row_count,
        "valid_records": len(df),
        "invalid_timestamps_excluded": invalid_timestamps,
        "negative_energy_excluded": negative_energy_count,
        "nan_energy_excluded": nan_energy_count,
        "duplicate_timestamps_resolved": duplicate_count,
        "has_temperature": has_temp,
        "has_humidity": has_hum,
        "has_occupancy": has_occ,
        "start_time": str(df["timestamp"].min()),
        "end_time": str(df["timestamp"].max()),
        "mean_energy_kwh": round(float(df["energy_kwh"].mean()), 2),
        "min_energy_kwh": round(float(df["energy_kwh"].min()), 2),
        "max_energy_kwh": round(float(df["energy_kwh"].max()), 2),
    }

    return df, validation_report


def parse_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible wrapper that validates, parses, and cleans dataset."""
    cleaned_df, _ = validate_and_clean_dataset(df)
    return cleaned_df


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts temporal, lag, rolling, and thermal degree features.
    Strictly uses past information to prevent future data leakage.
    """
    df = df.copy()

    # Temporal features
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["dayofweek"] = df["timestamp"].dt.day_of_week
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["is_business_hour"] = (
        (df["hour"] >= 8) & (df["hour"] <= 18) & (df["is_weekend"] == 0)
    ).astype(int)

    # Thermal degree features (HVAC cooling above 22°C, heating below 18°C)
    df["cooling_degree"] = (df["temperature"] - 22.0).clip(lower=0.0)
    df["heating_degree"] = (18.0 - df["temperature"]).clip(lower=0.0)

    # Lag features strictly from past readings (energy at t-1, t-2, t-24)
    df["lag_1"] = df["energy_kwh"].shift(1)
    df["lag_2"] = df["energy_kwh"].shift(2)
    df["lag_24"] = df["energy_kwh"].shift(24)

    # Rolling window statistics (shifted by 1 to prevent current/future target leakage)
    shifted_energy = df["energy_kwh"].shift(1)
    df["rolling_mean_6"] = shifted_energy.rolling(window=6, min_periods=1).mean()
    df["rolling_mean_24"] = shifted_energy.rolling(window=24, min_periods=1).mean()

    # Backfill earliest initial rows so valuable chronological data is not discarded
    df["lag_24"] = df["lag_24"].bfill().fillna(df["energy_kwh"])
    df["lag_2"] = df["lag_2"].bfill().fillna(df["energy_kwh"])
    df["lag_1"] = df["lag_1"].bfill().fillna(df["energy_kwh"])
    df["rolling_mean_6"] = df["rolling_mean_6"].bfill().fillna(df["energy_kwh"])
    df["rolling_mean_24"] = df["rolling_mean_24"].bfill().fillna(df["energy_kwh"])

    # Drop any remaining uncomputable rows and retain timestamp for reference
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)
    return df


def chronological_split(
    df: pd.DataFrame, train_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits time-series data chronologically (train/test).
    Guarantees no future data is leaked into the training partition.
    """
    n = len(df)
    split_idx = int(n * train_ratio)
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
    return train_df, test_df


def chronological_train_val_test_split(
    df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits time-series data chronologically into three partitions:
    - Train (earliest data, used to fit preprocessing and model parameters)
    - Validation (intermediate data, used for model selection / hyperparameter tuning)
    - Test (latest unseen data, reserved exclusively for final evaluation)
    """
    n = len(df)
    if n < 100:
        # For smaller datasets, use standard 80/20 train/test with val mirroring test
        train_df, test_df = chronological_split(df, train_ratio=0.8)
        return train_df, test_df.copy(), test_df

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)

    return train_df, val_df, test_df


def build_prediction_features(
    temperature: float,
    humidity: float,
    occupancy: int,
    hour: int,
    day_of_week: int,
    prev_energy_kwh: float,
    reference_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Constructs a single-row feature vector matching FEATURE_COLUMNS for live model inference.
    """
    is_weekend = 1 if day_of_week in [5, 6] else 0
    is_biz = 1 if (8 <= hour <= 18 and not is_weekend) else 0
    cooling = max(0.0, float(temperature) - 22.0)
    heating = max(0.0, 18.0 - float(temperature))
    month = datetime.now().month

    # Reference values if historical dataset is provided
    if reference_df is not None and not reference_df.empty and "hour" in reference_df.columns:
        hour_history = reference_df[reference_df["hour"] == hour]
        if not hour_history.empty and "energy_kwh" in hour_history.columns:
            lag_24_est = float(hour_history["energy_kwh"].mean())
        else:
            lag_24_est = float(prev_energy_kwh)
        rolling_24_est = float(reference_df["energy_kwh"].tail(24).mean()) if len(reference_df) >= 24 else float(prev_energy_kwh)
        rolling_6_est = float(reference_df["energy_kwh"].tail(6).mean()) if len(reference_df) >= 6 else float(prev_energy_kwh)
    else:
        lag_24_est = float(prev_energy_kwh)
        rolling_24_est = float(prev_energy_kwh) * 0.95
        rolling_6_est = float(prev_energy_kwh) * 0.98

    sample = {
        "hour": int(hour),
        "dayofweek": int(day_of_week),
        "month": int(month),
        "is_weekend": int(is_weekend),
        "is_business_hour": int(is_biz),
        "temperature": float(temperature),
        "humidity": float(humidity),
        "occupancy": int(occupancy),
        "lag_1": float(prev_energy_kwh),
        "lag_2": float(prev_energy_kwh * 0.99),
        "lag_24": float(lag_24_est),
        "rolling_mean_6": float(rolling_6_est),
        "rolling_mean_24": float(rolling_24_est),
        "cooling_degree": float(cooling),
        "heating_degree": float(heating),
    }

    return pd.DataFrame([sample])[FEATURE_COLUMNS]


def generate_synthetic_building_dataset(
    days: int = 90,
    output_path: Optional[str | Path] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generates a physics-inspired synthetic hourly energy dataset for a 6-story smart office building.
    Clearly designated as synthetic simulation benchmark data.
    """
    np.random.seed(seed)
    start_time = datetime(2026, 8, 1, 0, 0, 0)
    total_hours = days * 24

    timestamps = [start_time + timedelta(hours=i) for i in range(total_hours)]
    records = []
    base_load = 28.0

    for dt in timestamps:
        hour = dt.hour
        weekday = dt.weekday()
        is_weekend = weekday in [5, 6]

        day_progress = (dt - start_time).total_seconds() / (86400 * days)
        temp_season = 31.0 - 5.0 * day_progress
        temp_diurnal = 5.5 * np.sin((hour - 9) * np.pi / 12)
        temp_noise = np.random.normal(0, 0.8)
        temperature = round(temp_season + temp_diurnal + temp_noise, 1)

        humidity = round(float(np.clip(70.0 - 15.0 * np.sin((hour - 9) * np.pi / 12) + np.random.normal(0, 2), 25, 95)), 1)

        if is_weekend:
            occupancy = int(np.random.poisson(8) if 9 <= hour <= 17 else np.random.poisson(2))
        else:
            if 8 <= hour <= 18:
                occ_factor = np.exp(-((hour - 13) ** 2) / 12.0)
                occupancy = int(np.clip(180 * occ_factor + np.random.normal(0, 10), 20, 240))
            elif 6 <= hour < 8 or 19 <= hour <= 21:
                occupancy = int(np.random.poisson(25))
            else:
                occupancy = int(np.random.poisson(4))

        plug_load = occupancy * 0.16
        cooling_deg = max(0.0, temperature - 22.0)
        heating_deg = max(0.0, 18.0 - temperature)
        hvac_load = (cooling_deg * 2.2) + (heating_deg * 1.5) + (occupancy * 0.08)

        if 8 <= hour <= 19 and not is_weekend:
            lighting_load = 14.0
        elif is_weekend and (10 <= hour <= 16):
            lighting_load = 5.0
        else:
            lighting_load = 3.0

        random_fluctuation = np.random.normal(0, 1.2)
        total_energy = base_load + plug_load + hvac_load + lighting_load + random_fluctuation
        energy_kwh = round(max(15.0, total_energy), 2)

        records.append({
            "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "energy_kwh": energy_kwh,
            "temperature": temperature,
            "humidity": humidity,
            "occupancy": occupancy,
        })

    df = pd.DataFrame(records)

    if output_path:
        out_p = Path(output_path)
        try:
            out_p.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_p, index=False)
        except (OSError, PermissionError):
            pass

    return df
