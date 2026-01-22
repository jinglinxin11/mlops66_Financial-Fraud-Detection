# System Monitoring & Alerting

## Overview
The Fraud Detection API uses a multi-layered monitoring approach:
1. **Application Metrics**: Prometheus metrics exposed at `/metrics`.
2. **Data Drift**: Evidently drift detection available via API and UI.
3. **Logs**: Request/Response logging to `logs/api_io.jsonl`.

## Metrics
The `prometheus-fastapi-instrumentator` exposes default Python/FastAPI metrics plus custom ones:
- `http_requests_total`: Total number of requests.
- `http_request_duration_seconds`: Histogram of request latencies.
- `http_request_size_bytes`: Size of requests.
- `http_response_size_bytes`: Size of responses.

## Access
- **Metrics Endpoint**: [http://localhost:8000/metrics](http://localhost:8000/metrics)
- **Gradio Dashboard**: [http://localhost:8000/ui](http://localhost:8000/ui) (Metrics Tab)

## Alerting Policy
Defined in `infra/alert_policy.json`. Keys alerts include:
- **High Latency**: 95th percentile latency > 500ms.
- **High Error Rate**: 5xx errors > 1% of total requests.
- **Drift Detected**: Drift score > 0.3.

## Automated Retraining
When drift is detected, the system triggers the `Retrain Model` GitHub Action.
- **Trigger**: `POST /check_drift` with `auto_retrain=True`.
- **Action**: Runs `src/train.py` and commits new model artifacts.
