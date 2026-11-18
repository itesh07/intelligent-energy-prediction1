"""
WattWise - Smart Energy AI
Flask Application Server & REST API.

Provides endpoints for:
- Serving modern glassmorphic dashboard UI
- In-memory CSV dataset upload, preview, and schema validation
- Machine learning model training & chronological evaluation
- Real-time energy consumption inference
- Analytics telemetry for Chart.js visualizations
- CORS support for decoupled Vercel (frontend) + Render (backend) deployments
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from prediction import EnergyPredictor, TARIFF_INR_PER_KWH
from preprocessing import (
    generate_synthetic_building_dataset,
    parse_and_clean_data,
    validate_and_clean_dataset,
)
from train_model import (
    DEFAULT_DEMO_PATH,
    MODEL_BUNDLE_PATH,
    TEMP_MODEL_BUNDLE_PATH,
    ensure_demo_dataset,
    resolve_model_bundle_path,
    train_pipeline,
)

BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = Path(tempfile.gettempdir())
DATA_DIR = BASE_DIR / "data"

ACTIVE_DATASET_CONFIG = TEMP_DIR / "wattwise_active_dataset.json"
ACTIVE_UPLOADED_CSV = TEMP_DIR / "wattwise_active_dataset.csv"
ALLOWED_EXTENSIONS = {".csv"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

# In-memory session state for active uploaded dataset (serverless-friendly)
_ACTIVE_DATAFRAME: Optional[pd.DataFrame] = None
_ACTIVE_DATASET_NAME: str = "demo_smart_building_energy.csv"
_ACTIVE_VALIDATION_REPORT: Optional[Dict[str, Any]] = None
_IS_SYNTHETIC: bool = True

# Initialize predictor
predictor = EnergyPredictor(bundle_path=resolve_model_bundle_path())


# --------------------------------------------------------------------------
# CORS & Serverless Headers
# --------------------------------------------------------------------------
@app.after_request
def add_cors_headers(response):
    """Allows cross-origin requests for decoupled Vercel frontend + Render backend."""
    origin = request.headers.get("Origin", "*")
    response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, HEAD"
    response.headers["Access-Control-Allow-Headers"] = (
        "Content-Type, Authorization, X-Requested-With, Accept, Origin"
    )
    return response


@app.route("/api/<path:path>", methods=["OPTIONS"])
def api_options_handler(path):
    """Responds to browser preflight requests immediately."""
    return "", 204


# --------------------------------------------------------------------------
# Active Dataset Management
# --------------------------------------------------------------------------
def get_active_dataframe() -> Tuple[pd.DataFrame, str, bool]:
    """
    Returns the active DataFrame, its name, and whether it is synthetic simulation.
    Checks in-memory cache, then /tmp, and falls back to baseline demo data.
    """
    global _ACTIVE_DATAFRAME, _ACTIVE_DATASET_NAME, _IS_SYNTHETIC

    if _ACTIVE_DATAFRAME is not None and not _ACTIVE_DATAFRAME.empty:
        return _ACTIVE_DATAFRAME, _ACTIVE_DATASET_NAME, _IS_SYNTHETIC

    if ACTIVE_UPLOADED_CSV.exists():
        try:
            df = pd.read_csv(ACTIVE_UPLOADED_CSV)
            cleaned_df, report = validate_and_clean_dataset(df)
            _ACTIVE_DATAFRAME = cleaned_df
            _ACTIVE_DATASET_NAME = ACTIVE_UPLOADED_CSV.name
            _IS_SYNTHETIC = False
            return _ACTIVE_DATAFRAME, _ACTIVE_DATASET_NAME, _IS_SYNTHETIC
        except Exception:
            pass

    demo_path = ensure_demo_dataset()
    try:
        raw_df = pd.read_csv(demo_path)
        cleaned_df, report = validate_and_clean_dataset(raw_df)
        _ACTIVE_DATAFRAME = cleaned_df
        _ACTIVE_DATASET_NAME = demo_path.name
        _IS_SYNTHETIC = True
        return _ACTIVE_DATAFRAME, _ACTIVE_DATASET_NAME, _IS_SYNTHETIC
    except Exception as exc:
        app.logger.warning("Error loading demo dataset: %s", exc)
        empty_df = pd.DataFrame()
        return empty_df, "None", True


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


# --------------------------------------------------------------------------
# Page Routes
# --------------------------------------------------------------------------
@app.get("/")
def index():
    _, dataset_name, is_synthetic = get_active_dataframe()
    api_base_url = os.environ.get("API_BASE_URL", "").rstrip("/")
    return render_template(
        "index.html",
        model_ready=predictor.is_ready,
        is_synthetic=is_synthetic,
        tariff=TARIFF_INR_PER_KWH,
        api_base_url=api_base_url,
    )


# --------------------------------------------------------------------------
# REST API Endpoints
# --------------------------------------------------------------------------
@app.get("/api/health")
def health():
    _, dataset_name, is_synthetic = get_active_dataframe()
    return jsonify({
        "status": "healthy",
        "service": "WattWise Smart Energy AI",
        "version": "1.0.0",
        "model_ready": predictor.is_ready,
        "active_dataset": dataset_name,
        "is_synthetic": is_synthetic,
        "tariff_per_kwh_inr": TARIFF_INR_PER_KWH,
    })


@app.get("/api/model-info")
def model_info():
    if not predictor.is_ready:
        return jsonify({
            "status": "not_ready",
            "message": "Machine learning model has not been trained or loaded yet. Call POST /api/train or trigger training in the dashboard.",
        }), 503
    return jsonify(predictor.get_info())


@app.get("/api/metrics")
def get_metrics():
    if not predictor.is_ready:
        return jsonify({
            "error": "Machine learning model bundle not found. Please train the model using 'Train AI Model' or upload a custom CSV dataset."
        }), 503

    bundle = predictor.bundle
    return jsonify({
        "best_model_name": bundle["best_model_name"],
        "all_metrics": bundle["all_metrics"],
        "dataset_info": bundle["dataset_info"],
        "feature_importance": bundle.get("feature_importance", {}),
        "trained_at": bundle.get("trained_at"),
    })


@app.post("/api/predict")
def predict_energy():
    if not predictor.is_ready:
        return jsonify({
            "error": "The AI model is not ready. Train the model first.",
            "code": "MODEL_NOT_READY",
        }), 503

    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Invalid request: JSON body required."}), 400

    required_fields = ["temperature", "humidity", "occupancy", "hour", "day_of_week", "prev_energy_kwh"]
    missing = [f for f in required_fields if f not in payload]
    if missing:
        return jsonify({"error": f"Missing required input fields: {', '.join(missing)}"}), 400

    try:
        prediction_result = predictor.predict(
            temperature=payload["temperature"],
            humidity=payload["humidity"],
            occupancy=payload["occupancy"],
            hour=payload["hour"],
            day_of_week=payload["day_of_week"],
            prev_energy_kwh=payload["prev_energy_kwh"],
        )
        return jsonify(prediction_result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({"error": f"Internal prediction error: {str(exc)}"}), 500


@app.post("/api/upload")
def upload_dataset():
    """
    Handles CSV dataset upload with direct in-memory Pandas processing.
    Validates required columns, parses timestamps, handles duplicates,
    negative values, and missing values without touching read-only paths.
    """
    global _ACTIVE_DATAFRAME, _ACTIVE_DATASET_NAME, _ACTIVE_VALIDATION_REPORT, _IS_SYNTHETIC

    if "file" not in request.files:
        return jsonify({"error": "No file part in request. Please upload a CSV file."}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file was selected for upload."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file format. Please upload a valid CSV file (.csv)."}), 400

    orig_name = secure_filename(file.filename) or "uploaded_dataset.csv"

    # Read uploaded bytes directly in memory to avoid saving to read-only directories
    try:
        file_bytes = file.read()
    except Exception as exc:
        return jsonify({"error": f"Failed to read uploaded file: {str(exc)}"}), 400

    if not file_bytes:
        return jsonify({"error": "Uploaded file is empty (0 bytes)."}), 400

    # 1. Parse CSV in-memory with Pandas
    try:
        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding="latin1")
    except Exception as e:
        return jsonify({"error": f"Failed to parse CSV format: {str(e)}"}), 422

    # 2. Validate schema, timestamps, positive energy, and clean dataset
    try:
        cleaned_df, report = validate_and_clean_dataset(df)
    except ValueError as ve:
        return jsonify({
            "error": str(ve),
            "details": "Ensure your CSV has 'timestamp' (e.g. YYYY-MM-DD HH:MM:SS) and 'energy_kwh' (non-negative float) columns and at least 24 valid records.",
        }), 422
    except Exception as ge:
        return jsonify({"error": f"Validation failed: {str(ge)}"}), 422

    # 3. Store active dataset in memory (ephemeral/serverless safe)
    _ACTIVE_DATAFRAME = cleaned_df
    _ACTIVE_DATASET_NAME = orig_name
    _ACTIVE_VALIDATION_REPORT = report
    _IS_SYNTHETIC = False

    # 4. Safely persist to writable /tmp for multi-request reuse
    try:
        cleaned_df.to_csv(ACTIVE_UPLOADED_CSV, index=False)
    except (OSError, PermissionError) as exc:
        app.logger.warning("Could not persist active dataset to /tmp: %s", exc)

    # 5. Build preview of first 5 rows
    preview_rows = cleaned_df.head(5).to_dict(orient="records")
    for row in preview_rows:
        if "timestamp" in row and hasattr(row["timestamp"], "strftime"):
            row["timestamp"] = row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")

    summary = {
        "total_records": len(cleaned_df),
        "start_time": report["start_time"],
        "end_time": report["end_time"],
        "mean_energy_kwh": report["mean_energy_kwh"],
        "min_energy_kwh": report["min_energy_kwh"],
        "max_energy_kwh": report["max_energy_kwh"],
        "has_temperature": report["has_temperature"],
        "has_humidity": report["has_humidity"],
        "has_occupancy": report["has_occupancy"],
        "validation_report": report,
    }

    return jsonify({
        "message": "Dataset successfully uploaded, validated, and loaded into active memory.",
        "filename": orig_name,
        "summary": summary,
        "preview": preview_rows,
        "columns": list(cleaned_df.columns),
    })


@app.post("/api/train")
def train_models():
    """
    Triggers chronological model training and automated model selection
    on the currently active dataset (supports in-memory DataFrame).
    """
    global _ACTIVE_DATAFRAME, _ACTIVE_DATASET_NAME, _IS_SYNTHETIC

    try:
        if _ACTIVE_DATAFRAME is not None and not _ACTIVE_DATAFRAME.empty:
            training_input = _ACTIVE_DATAFRAME
        elif ACTIVE_UPLOADED_CSV.exists():
            training_input = ACTIVE_UPLOADED_CSV
        else:
            training_input = ensure_demo_dataset()

        # Run complete ML training pipeline
        bundle = train_pipeline(data_input=training_input)

        # Update in-memory predictor immediately
        predictor.set_bundle(bundle)

        return jsonify({
            "message": "Models successfully trained and evaluated.",
            "best_model_name": bundle["best_model_name"],
            "all_metrics": bundle["all_metrics"],
            "dataset_info": bundle["dataset_info"],
            "trained_at": bundle["trained_at"],
            "training_time_seconds": bundle["total_pipeline_time_seconds"],
        })
    except Exception as exc:
        app.logger.exception("Training pipeline failed")
        return jsonify({"error": f"Model training failed: {str(exc)}"}), 500


@app.post("/api/reset-demo")
def reset_demo():
    """Resets the active dataset back to the default synthetic smart building demo dataset."""
    global _ACTIVE_DATAFRAME, _ACTIVE_DATASET_NAME, _ACTIVE_VALIDATION_REPORT, _IS_SYNTHETIC

    demo_path = ensure_demo_dataset()
    _ACTIVE_DATAFRAME = None
    _ACTIVE_DATASET_NAME = demo_path.name
    _ACTIVE_VALIDATION_REPORT = None
    _IS_SYNTHETIC = True

    if ACTIVE_UPLOADED_CSV.exists():
        try:
            ACTIVE_UPLOADED_CSV.unlink(missing_ok=True)
        except OSError:
            pass

    try:
        bundle = train_pipeline(data_input=demo_path)
        predictor.set_bundle(bundle)
        return jsonify({
            "message": "Reset to synthetic smart building demo dataset completed.",
            "best_model_name": bundle["best_model_name"],
            "all_metrics": bundle["all_metrics"],
            "dataset_info": bundle["dataset_info"],
        })
    except Exception as exc:
        app.logger.exception("Reset demo failed")
        return jsonify({"error": f"Reset failed: {str(exc)}"}), 500


@app.get("/api/analytics")
def get_analytics():
    """
    Returns aggregated metrics and timeseries data for Chart.js:
    - 4 Top Metric Cards (Current kWh, Next-hour predicted kWh, Daily Average kWh, Estimated Cost ₹)
    - Hourly consumption (last 48h)
    - Daily consumption totals
    - Weekly averages (Mon-Sun)
    - Actual vs Predicted (from test evaluation)
    - Distribution histogram bins
    """
    df, dataset_name, is_synthetic = get_active_dataframe()
    if df.empty:
        return jsonify({"error": "No active dataset available."}), 500

    # 1. Metric Cards calculations
    current_kwh = round(float(df["energy_kwh"].iloc[-1]), 2)
    last_row = df.iloc[-1]

    # Predict next hour if model is ready
    if predictor.is_ready:
        next_hour = (int(last_row["timestamp"].hour) + 1) % 24
        next_day = int(last_row["timestamp"].weekday())
        if next_hour == 0:
            next_day = (next_day + 1) % 7
        try:
            pred_res = predictor.predict(
                temperature=float(last_row.get("temperature", 28.0)),
                humidity=float(last_row.get("humidity", 55.0)),
                occupancy=int(last_row.get("occupancy", 50)),
                hour=next_hour,
                day_of_week=next_day,
                prev_energy_kwh=current_kwh,
            )
            predicted_next_kwh = pred_res["predicted_kwh"]
        except Exception:
            predicted_next_kwh = current_kwh
    else:
        predicted_next_kwh = current_kwh

    # Daily aggregation
    df["date_str"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    daily_grp = df.groupby("date_str")["energy_kwh"].sum()
    avg_daily_kwh = round(float(daily_grp.mean()), 2)
    estimated_daily_cost_inr = round(avg_daily_kwh * TARIFF_INR_PER_KWH, 2)

    # 2. Hourly Timeseries (last 48 hours for crisp visualization)
    recent_48 = df.tail(48).copy()
    hourly_data = {
        "labels": [dt.strftime("%d %b %H:%M") for dt in recent_48["timestamp"]],
        "values": [round(float(v), 2) for v in recent_48["energy_kwh"]],
        "temperatures": [round(float(v), 1) for v in recent_48["temperature"]],
        "occupancies": [int(v) for v in recent_48["occupancy"]],
    }

    # 3. Daily Totals (last 14 days)
    recent_daily = daily_grp.tail(14)
    daily_data = {
        "labels": [datetime.strptime(d, "%Y-%m-%d").strftime("%d %b") for d in recent_daily.index],
        "values": [round(float(v), 2) for v in recent_daily.values],
    }

    # 4. Weekly Average by Day of Week (0=Monday ... 6=Sunday)
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    df["dow"] = df["timestamp"].dt.day_of_week
    dow_means = df.groupby("dow")["energy_kwh"].mean()
    weekly_data = {
        "labels": day_names,
        "values": [round(float(dow_means.get(i, 0.0)), 2) for i in range(7)],
    }

    # 5. Actual vs Predicted (from test split in model bundle)
    actual_vs_predicted = {"timestamps": [], "actual": [], "predicted": []}
    if predictor.is_ready and "test_chart_data" in predictor.bundle:
        actual_vs_predicted = predictor.bundle["test_chart_data"]

    # 6. Distribution Histogram (10 bins)
    hist_counts, bin_edges = np.histogram(df["energy_kwh"], bins=10)
    distribution_data = {
        "labels": [f"{round(bin_edges[i], 1)}-{round(bin_edges[i+1], 1)}" for i in range(len(hist_counts))],
        "counts": [int(c) for c in hist_counts],
    }

    return jsonify({
        "metrics": {
            "current_energy_kwh": current_kwh,
            "predicted_next_kwh": predicted_next_kwh,
            "average_daily_kwh": avg_daily_kwh,
            "estimated_daily_cost_inr": estimated_daily_cost_inr,
            "is_synthetic": is_synthetic,
            "dataset_name": dataset_name,
            "total_records": len(df),
        },
        "hourly": hourly_data,
        "daily": daily_data,
        "weekly": weekly_data,
        "actual_vs_predicted": actual_vs_predicted,
        "distribution": distribution_data,
    })


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({"error": "File size exceeds the 25 MB upload limit."}), 413


@app.errorhandler(404)
def not_found(_error):
    return jsonify({"error": "Resource not found."}), 404


@app.errorhandler(500)
def server_error(_error):
    return jsonify({"error": "An internal server error occurred."}), 500


if __name__ == "__main__":
    ensure_demo_dataset()
    if not predictor.is_ready:
        print("[WattWise] Initializing baseline model...")
        train_pipeline()
        predictor.load_model()

    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() in ["true", "1"]
    print(f"[WattWise] Server starting on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
