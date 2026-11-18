"""
Automated Test Suite for WattWise Smart Energy AI.
Validates all Flask REST API endpoints, model predictions, in-memory file uploads,
data schema validation, 1,000-row user dataset training lifecycle,
and analytics telemetry using standard Python unittest.
"""

from __future__ import annotations

import io
import json
import unittest
from pathlib import Path

from app import app, predictor

BASE_DIR = Path(__file__).resolve().parent
USER_DATASET_PATH = BASE_DIR / "data" / "user_1000_energy_dataset.csv"


class WattWiseApiTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_01_index_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.data.decode("utf-8")
        self.assertIn("WATTWISE", html)
        self.assertIn("Intelligent Energy Consumption", html)
        self.assertIn("chart.js", html.lower())

    def test_02_api_health(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_ready"])
        self.assertIn("tariff_per_kwh_inr", data)

    def test_03_api_metrics(self):
        response = self.client.get("/api/metrics")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("best_model_name", data)
        self.assertIn("all_metrics", data)
        self.assertIn("Linear Regression", data["all_metrics"])
        self.assertIn("Random Forest Regressor", data["all_metrics"])
        self.assertIn("Gradient Boosting Regressor", data["all_metrics"])

        rf_metrics = data["all_metrics"]["Random Forest Regressor"]
        self.assertGreater(rf_metrics["mae"], 0)
        self.assertGreater(rf_metrics["rmse"], 0)
        self.assertIn("r2_score", rf_metrics)

    def test_04_api_analytics(self):
        response = self.client.get("/api/analytics")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("metrics", data)
        self.assertIn("current_energy_kwh", data["metrics"])
        self.assertIn("predicted_next_kwh", data["metrics"])
        self.assertIn("average_daily_kwh", data["metrics"])
        self.assertIn("estimated_daily_cost_inr", data["metrics"])
        self.assertIn("hourly", data)
        self.assertGreater(len(data["hourly"]["values"]), 0)
        self.assertIn("daily", data)
        self.assertIn("weekly", data)
        self.assertIn("actual_vs_predicted", data)
        self.assertIn("distribution", data)

    def test_05_api_model_info(self):
        response = self.client.get("/api/model-info")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "ready")
        self.assertIn("best_model_name", data)

    def test_06_api_predict_success(self):
        payload = {
            "temperature": 30.5,
            "humidity": 65.0,
            "occupancy": 140,
            "hour": 14,
            "day_of_week": 2,
            "prev_energy_kwh": 62.4,
        }
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("predicted_kwh", data)
        self.assertGreater(data["predicted_kwh"], 5.0)
        self.assertIn("estimated_cost_inr", data)
        self.assertIn("load_category", data)
        self.assertIn("breakdown", data)
        self.assertIn("base_building_load", data["breakdown"])
        self.assertIn("hvac_thermal_load", data["breakdown"])
        self.assertIn("occupancy_plug_load", data["breakdown"])

    def test_07_api_predict_validation_errors(self):
        # Missing required field
        response = self.client.post("/api/predict", json={"temperature": 25.0})
        self.assertEqual(response.status_code, 400)

        # Invalid hour
        payload = {
            "temperature": 25.0,
            "humidity": 50.0,
            "occupancy": 50,
            "hour": 29,  # invalid!
            "day_of_week": 1,
            "prev_energy_kwh": 40.0,
        }
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_08_api_upload_and_train_and_reset(self):
        # 1. Create a mini valid CSV with 60 hourly rows
        from datetime import datetime, timedelta
        base = datetime(2026, 1, 1, 0, 0, 0)
        csv_content = "timestamp,energy_kwh,temperature,humidity,occupancy\n"
        for i in range(60):
            dt = base + timedelta(hours=i)
            csv_content += f"{dt.strftime('%Y-%m-%d %H:%M:%S')},{50.0 + (i % 15)},{25.0 + (i % 5)},55.0,{20 + i}\n"

        data = {
            "file": (io.BytesIO(csv_content.encode("utf-8")), "test_telemetry.csv")
        }
        upload_res = self.client.post("/api/upload", data=data, content_type="multipart/form-data")
        self.assertEqual(upload_res.status_code, 200)
        upload_json = upload_res.get_json()
        self.assertEqual(upload_json["filename"], "test_telemetry.csv")
        self.assertEqual(upload_json["summary"]["total_records"], 60)
        self.assertEqual(len(upload_json["preview"]), 5)

        # 2. Retrain models on this uploaded data
        train_res = self.client.post("/api/train")
        self.assertEqual(train_res.status_code, 200)
        train_json = train_res.get_json()
        self.assertIn("best_model_name", train_json)
        self.assertIn("all_metrics", train_json)

        # 3. Reset back to demo dataset
        reset_res = self.client.post("/api/reset-demo")
        self.assertEqual(reset_res.status_code, 200)
        reset_json = reset_res.get_json()
        self.assertTrue(reset_json["dataset_info"]["is_synthetic"])

    def test_09_user_1000_records_full_lifecycle(self):
        """
        Validates Section 9: Complete 1,000-row user dataset pipeline:
        1. Ingest 1,000-row CSV via /api/upload
        2. Validate schema & check 1,000 records
        3. Train Linear Regression, Random Forest, Gradient Boosting via /api/train
        4. Validate MAE, RMSE, R² scores and automated best model selection
        5. Verify inference via /api/predict
        6. Verify analytics and actual vs predicted data via /api/analytics
        """
        self.assertTrue(USER_DATASET_PATH.exists(), "User 1000-row dataset must exist")

        with open(USER_DATASET_PATH, "rb") as f:
            csv_bytes = f.read()

        # Step 1 & 2: Upload and validate schema
        data = {
            "file": (io.BytesIO(csv_bytes), "user_1000_smart_building.csv")
        }
        upload_res = self.client.post("/api/upload", data=data, content_type="multipart/form-data")
        self.assertEqual(upload_res.status_code, 200)
        upload_json = upload_res.get_json()
        self.assertEqual(upload_json["filename"], "user_1000_smart_building.csv")
        self.assertEqual(upload_json["summary"]["total_records"], 1000)
        self.assertTrue(upload_json["summary"]["has_temperature"])
        self.assertTrue(upload_json["summary"]["has_humidity"])
        self.assertTrue(upload_json["summary"]["has_occupancy"])
        self.assertEqual(len(upload_json["preview"]), 5)

        # Step 3 & 4: Train models on the 1,000-row dataset
        train_res = self.client.post("/api/train")
        self.assertEqual(train_res.status_code, 200)
        train_json = train_res.get_json()
        self.assertIn("best_model_name", train_json)
        self.assertIn("all_metrics", train_json)

        metrics = train_json["all_metrics"]
        self.assertIn("Linear Regression", metrics)
        self.assertIn("Random Forest Regressor", metrics)
        self.assertIn("Gradient Boosting Regressor", metrics)

        # Confirm valid non-zero regression metrics
        for model_name, m in metrics.items():
            self.assertGreater(m["mae"], 0.0, f"{model_name} MAE should be > 0")
            self.assertGreater(m["rmse"], 0.0, f"{model_name} RMSE should be > 0")
            self.assertIn("r2_score", m)
            self.assertIn("training_time_seconds", m)

        # Step 5: Verify inference
        predict_res = self.client.post("/api/predict", json={
            "temperature": 18.5,
            "humidity": 60.0,
            "occupancy": 175,
            "hour": 14,
            "day_of_week": 2,
            "prev_energy_kwh": 45.0,
        })
        self.assertEqual(predict_res.status_code, 200)
        pred_data = predict_res.get_json()
        self.assertGreater(pred_data["predicted_kwh"], 10.0)
        self.assertIn("estimated_cost_inr", pred_data)
        self.assertIn("load_category", pred_data)
        self.assertIn("breakdown", pred_data)

        # Step 6: Verify analytics telemetry
        analytics_res = self.client.get("/api/analytics")
        self.assertEqual(analytics_res.status_code, 200)
        analytics_data = analytics_res.get_json()
        self.assertEqual(analytics_data["metrics"]["total_records"], 1000)
        self.assertFalse(analytics_data["metrics"]["is_synthetic"])
        self.assertGreater(len(analytics_data["actual_vs_predicted"]["actual"]), 0)
        self.assertGreater(len(analytics_data["actual_vs_predicted"]["predicted"]), 0)

    def test_10_upload_validation_error_cases(self):
        """Validates robust handling and descriptive error messages for invalid uploads."""
        # Case A: Missing required energy_kwh column
        bad_csv_1 = "timestamp,temperature,humidity\n2026-01-01 00:00:00,25.0,60.0\n"
        res = self.client.post("/api/upload", data={"file": (io.BytesIO(bad_csv_1.encode()), "bad.csv")}, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 422)
        err_json = res.get_json()
        self.assertIn("error", err_json)
        self.assertIn("energy_kwh", err_json["error"].lower())

        # Case B: Unparseable timestamps
        bad_csv_2 = "timestamp,energy_kwh\nnot_a_time,50.0\ninvalid,45.0\n"
        res = self.client.post("/api/upload", data={"file": (io.BytesIO(bad_csv_2.encode()), "bad2.csv")}, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 422)
        err_json = res.get_json()
        self.assertIn("timestamp", err_json["error"].lower())

        # Case C: All negative energy values
        bad_csv_3 = "timestamp,energy_kwh\n2026-01-01 00:00:00,-10.0\n2026-01-01 01:00:00,-20.0\n"
        res = self.client.post("/api/upload", data={"file": (io.BytesIO(bad_csv_3.encode()), "bad3.csv")}, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 422)
        err_json = res.get_json()
        self.assertIn("negative", err_json["error"].lower())

        # Case D: Empty file
        res = self.client.post("/api/upload", data={"file": (io.BytesIO(b""), "empty.csv")}, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 400)


if __name__ == "__main__":
    unittest.main()
