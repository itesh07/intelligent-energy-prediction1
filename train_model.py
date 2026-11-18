"""
Machine Learning Training Pipeline for Smart Building Energy Consumption Prediction.
Implements chronological train-validation-test splitting, data standardization,
model training for Linear Regression, Random Forest, and Gradient Boosting,
rigorous out-of-sample evaluation (MAE, RMSE, R²), automated model selection,
and robust serialization using Joblib with serverless read-only filesystem fallback.
"""

from __future__ import annotations

import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from preprocessing import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    chronological_train_val_test_split,
    generate_features,
    generate_synthetic_building_dataset,
    validate_and_clean_dataset,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DEFAULT_DEMO_PATH = DATA_DIR / "demo_smart_building_energy.csv"
MODEL_BUNDLE_PATH = MODELS_DIR / "energy_model_bundle.joblib"

TEMP_DIR = Path(tempfile.gettempdir())
TEMP_MODELS_DIR = TEMP_DIR / "wattwise_models"
TEMP_MODEL_BUNDLE_PATH = TEMP_MODELS_DIR / "energy_model_bundle.joblib"


def get_writable_model_path() -> Path:
    """
    Returns a writable path for saving the trained model bundle.
    Prefers the local workspace models directory; safely falls back to system /tmp
    if deployed in read-only serverless environments such as Vercel /var/task.
    """
    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        probe = MODELS_DIR / ".write_probe"
        probe.touch()
        probe.unlink(missing_ok=True)
        return MODEL_BUNDLE_PATH
    except (OSError, PermissionError):
        TEMP_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        return TEMP_MODEL_BUNDLE_PATH


def resolve_model_bundle_path() -> Optional[Path]:
    """
    Resolves the active model bundle path, checking workspace models and /tmp.
    Returns the path to the newest bundle if multiple exist.
    """
    paths_to_check = [MODEL_BUNDLE_PATH, TEMP_MODEL_BUNDLE_PATH]
    existing = [p for p in paths_to_check if p.exists()]
    if not existing:
        return None
    # If both exist, pick the one modified most recently
    return max(existing, key=lambda p: p.stat().st_mtime)


def ensure_demo_dataset() -> Path:
    """Ensures a baseline synthetic smart building dataset exists for testing and cold starts."""
    if DEFAULT_DEMO_PATH.exists():
        return DEFAULT_DEMO_PATH

    # Try creating demo in workspace data directory
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        target_path = DEFAULT_DEMO_PATH
    except (OSError, PermissionError):
        temp_data_dir = TEMP_DIR / "wattwise_data"
        temp_data_dir.mkdir(parents=True, exist_ok=True)
        target_path = temp_data_dir / "demo_smart_building_energy.csv"

    if not target_path.exists():
        print(f"[WattWise] Generating realistic synthetic smart building dataset at {target_path}...")
        generate_synthetic_building_dataset(days=90, output_path=target_path)

    return target_path


def train_pipeline(
    data_input: Union[pd.DataFrame, str, Path, None] = None,
) -> Dict[str, Any]:
    """
    Executes the end-to-end ML training and validation pipeline:
    1. Loads dataset from DataFrame or CSV path (supports in-memory processing)
    2. Validates schema and extracts chronological time-series features
    3. Chronologically splits data (70% train, 15% validation, 15% test)
    4. Fits StandardScaler strictly on training split to eliminate data leakage
    5. Trains Linear Regression, Random Forest, and Gradient Boosting
    6. Benchmarks using validation split for automated model selection
    7. Evaluates final performance on unseen chronological test split (MAE, RMSE, R²)
    8. Serializes model bundle to Joblib (with fallback for read-only serverless filesystems)
    """
    start_total_time = time.time()

    # 1. Load data
    if isinstance(data_input, pd.DataFrame):
        raw_df = data_input.copy()
        dataset_name = "in_memory_upload.csv"
        is_synthetic = False
    elif data_input is not None and Path(data_input).exists():
        csv_path = Path(data_input)
        dataset_name = csv_path.name
        is_synthetic = (dataset_name == DEFAULT_DEMO_PATH.name)
        print(f"[WattWise] Loading dataset from file: {csv_path}")
        raw_df = pd.read_csv(csv_path)
    else:
        demo_path = ensure_demo_dataset()
        dataset_name = demo_path.name
        is_synthetic = True
        print(f"[WattWise] Loading baseline demo dataset from: {demo_path}")
        raw_df = pd.read_csv(demo_path)

    # 2. Validation & Preprocessing
    cleaned_df, validation_report = validate_and_clean_dataset(raw_df)
    dataset_rows = len(cleaned_df)
    min_date = str(cleaned_df["timestamp"].min())
    max_date = str(cleaned_df["timestamp"].max())
    mean_kwh = float(cleaned_df["energy_kwh"].mean())
    current_kwh = float(cleaned_df["energy_kwh"].iloc[-1])

    # 3. Feature engineering
    feature_df = generate_features(cleaned_df)

    # 4. Chronological 3-way split (Train / Validation / Test)
    train_df, val_df, test_df = chronological_train_val_test_split(
        feature_df, train_ratio=0.70, val_ratio=0.15
    )

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN].values
    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df[TARGET_COLUMN].values
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN].values

    # 5. Fit Scaler STRICTLY on training split
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 6. Candidate model definitions
    candidate_models = {
        "Linear Regression": {
            "model": LinearRegression(),
            "use_scaled": True,
            "description": "Baseline ordinary least squares with L2 feature standardization",
        },
        "Random Forest Regressor": {
            "model": RandomForestRegressor(
                n_estimators=100,
                max_depth=12,
                min_samples_split=4,
                random_state=42,
                n_jobs=-1,
            ),
            "use_scaled": False,
            "description": "Ensemble of 100 decorrelated decision trees with bootstrap aggregation",
        },
        "Gradient Boosting Regressor": {
            "model": GradientBoostingRegressor(
                n_estimators=120,
                learning_rate=0.07,
                max_depth=4,
                subsample=0.85,
                random_state=42,
            ),
            "use_scaled": False,
            "description": "Sequential gradient boosted regression trees minimizing squared error",
        },
    }

    results = {}
    fitted_models = {}
    test_predictions = {}
    val_predictions = {}

    for name, config in candidate_models.items():
        print(f"[WattWise] Training {name}...")
        t0 = time.time()
        model = config["model"]

        X_tr = X_train_scaled if config["use_scaled"] else X_train
        X_v = X_val_scaled if config["use_scaled"] else X_val
        X_te = X_test_scaled if config["use_scaled"] else X_test

        # Fit model on training split
        model.fit(X_tr, y_train)
        train_duration = round(time.time() - t0, 3)

        # Validation predictions for model selection
        val_preds = model.predict(X_v)
        val_predictions[name] = val_preds
        val_mae = float(mean_absolute_error(y_val, val_preds))
        val_rmse = float(np.sqrt(mean_squared_error(y_val, val_preds)))
        val_r2 = float(r2_score(y_val, val_preds))

        # Test predictions for final evaluation
        test_preds = model.predict(X_te)
        test_predictions[name] = test_preds
        test_mae = float(mean_absolute_error(y_test, test_preds))
        test_rmse = float(np.sqrt(mean_squared_error(y_test, test_preds)))
        test_r2 = float(r2_score(y_test, test_preds))

        results[name] = {
            "mae": round(test_mae, 3),
            "rmse": round(test_rmse, 3),
            "r2_score": round(test_r2, 4),
            "val_mae": round(val_mae, 3),
            "val_rmse": round(val_rmse, 3),
            "val_r2_score": round(val_r2, 4),
            "training_time_seconds": train_duration,
            "description": config["description"],
            "use_scaled": config["use_scaled"],
        }
        fitted_models[name] = model
        print(
            f"  -> Validation RMSE: {val_rmse:.3f} kWh | Test MAE: {test_mae:.3f} kWh | "
            f"Test RMSE: {test_rmse:.3f} kWh | Test R²: {test_r2:.4f} (in {train_duration}s)"
        )

    # 7. Model selection strictly on validation split performance
    best_model_name = min(results.keys(), key=lambda k: results[k]["val_rmse"])
    best_model = fitted_models[best_model_name]
    best_metrics = results[best_model_name]
    print(
        f"[WattWise] Automated Best Model Selected: {best_model_name} "
        f"(Val RMSE: {best_metrics['val_rmse']} kWh, Test RMSE: {best_metrics['rmse']} kWh, Test R²: {best_metrics['r2_score']})"
    )

    # 8. Feature Importance
    feature_importance = {}
    if hasattr(best_model, "feature_importances_"):
        raw_imp = best_model.feature_importances_
        feature_importance = {col: round(float(imp), 4) for col, imp in zip(FEATURE_COLUMNS, raw_imp)}
    elif hasattr(best_model, "coef_"):
        coef_abs = np.abs(best_model.coef_)
        coef_sum = np.sum(coef_abs) if np.sum(coef_abs) > 0 else 1.0
        feature_importance = {col: round(float(c / coef_sum), 4) for col, c in zip(FEATURE_COLUMNS, coef_abs)}

    # 9. Test set actual vs predicted chart points (last 96 test hours for clear visualization)
    chart_window = min(96, len(test_df))
    chart_slice = test_df.tail(chart_window).copy()
    chart_actual = [round(float(v), 2) for v in chart_slice[TARGET_COLUMN].values]
    chart_timestamps = [dt.strftime("%d %b %H:%M") for dt in chart_slice["timestamp"]]
    chart_predicted = [
        round(float(p), 2)
        for p in test_predictions[best_model_name][-chart_window:]
    ]

    # Daily aggregation for dashboard
    cleaned_df["date_str"] = cleaned_df["timestamp"].dt.strftime("%Y-%m-%d")
    daily_stats = cleaned_df.groupby("date_str")["energy_kwh"].agg(["sum", "mean"]).reset_index()
    avg_daily_kwh = float(daily_stats["sum"].mean())

    # 10. Assemble complete model bundle
    bundle = {
        "best_model_name": best_model_name,
        "best_model": best_model,
        "scaler": scaler,
        "candidate_models": fitted_models,
        "all_metrics": results,
        "feature_columns": FEATURE_COLUMNS,
        "feature_importance": feature_importance,
        "dataset_info": {
            "rows": dataset_rows,
            "min_date": min_date,
            "max_date": max_date,
            "mean_kwh": round(mean_kwh, 2),
            "current_kwh": round(current_kwh, 2),
            "avg_daily_kwh": round(avg_daily_kwh, 2),
            "is_synthetic": is_synthetic,
            "dataset_filename": dataset_name,
            "validation_report": validation_report,
        },
        "test_chart_data": {
            "timestamps": chart_timestamps,
            "actual": chart_actual,
            "predicted": chart_predicted,
        },
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "total_pipeline_time_seconds": round(time.time() - start_total_time, 2),
    }

    # 11. Serialize bundle to writable path (local workspace or /tmp fallback)
    target_save_path = get_writable_model_path()
    try:
        target_save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, target_save_path)
        print(f"[WattWise] Model bundle saved successfully to {target_save_path}")
    except (OSError, PermissionError) as exc:
        print(f"[WattWise] Primary model save path not writable ({exc}), falling back to /tmp...")
        TEMP_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, TEMP_MODEL_BUNDLE_PATH)
        print(f"[WattWise] Model bundle saved to temporary fallback at {TEMP_MODEL_BUNDLE_PATH}")

    return bundle


if __name__ == "__main__":
    ensure_demo_dataset()
    train_pipeline()
