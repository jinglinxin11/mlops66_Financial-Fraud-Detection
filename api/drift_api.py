"""Module for handling data drift detection and GCS logging."""

import json
from datetime import datetime

import pandas as pd
from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from google.cloud import storage

# --- Safety Block (Safe Mode) ---
try:
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    EVIDENTLY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Evidently is not available ({e}). Drift detection disabled.")
    EVIDENTLY_AVAILABLE = False
# --------------------------------

router = APIRouter()

BUCKET_NAME = "databucketmlops66"
LOG_FILE = "inference_logs.jsonl"

try:
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
except Exception:
    print("Warning: GCS client not initialized.")
    bucket = None


def log_prediction_to_gcs(input_data: dict, prediction: float):
    """Log prediction input and output to Google Cloud Storage."""
    if not bucket:
        return

    try:
        log_entry = input_data.copy()
        log_entry["prediction"] = float(prediction)
        log_entry["timestamp"] = datetime.now().isoformat()

        json_line = json.dumps(log_entry) + "\n"

        blob = bucket.blob(LOG_FILE)
        current_content = ""
        if blob.exists():
            current_content = blob.download_as_text()

        new_content = current_content + json_line
        blob.upload_from_string(new_content)
        print("Log saved to GCS")
    except Exception as e:
        print(f"Error logging to GCS: {e}")


@router.get("/drift", response_class=HTMLResponse, tags=["Monitoring"])
async def get_drift_report():
    """Generate and return an HTML data drift report."""
    if not EVIDENTLY_AVAILABLE:
        return """
        <h1>Drift Detection Disabled</h1>
        <p>Library 'evidently' could not be loaded on this server.</p>
        <p>But <b>Data Collection</b> (GCS Logging) is still working!</p>
        """

    if not bucket:
        return "<h1>GCS not configured</h1>"

    blob = bucket.blob(LOG_FILE)
    if not blob.exists():
        return "<h1>No data logged yet. Make predictions via /predict first!</h1>"

    try:
        content = blob.download_as_text()
        logs = [json.loads(line) for line in content.strip().split("\n")]
        df_current = pd.DataFrame(logs)

        numeric_cols = [c for c in df_current.columns if c not in ["timestamp", "prediction"]]
        df_current = df_current[numeric_cols]

        mid_point = len(df_current) // 2
        if mid_point < 2:
            return "<h1>Not enough data to calculate drift (need > 5 logs)</h1>"

        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=df_current.iloc[:mid_point], current_data=df_current.iloc[mid_point:]
        )

        return report.get_html()
    except Exception as e:
        return f"<h1>Error generating report: {e}</h1>"
