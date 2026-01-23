
# 🛠️ Project Upgrade Report: Unified MLOps Dashboard

## 1. Overview
This document outlines the major architectural upgrade performed on the Financial Fraud Detection system. We have transitioned from a fragmented architecture (FastAPI Backend + HTML Frontend + Separate Scripts) to a **Unified MLOps Dashboard (Ultimate Edition)** powered by **Gradio**.

This change significantly lowers the barrier to entry, integrates all ML lifecycle stages into one interface, and explicitly satisfies M27 (Robustness & Logging) and M28 (System Monitoring) requirements.

## 2. Key Changes & Features

| Feature Area | Old System | New System (`main.py`) | Improvement |
| :--- | :--- | :--- | :--- |
| **Interface** | Static HTML/JS frontend | **Gradio Web UI** | Interactive, Python-native, easier to extend. |
| **Training** | Command-line scripts | **"Advanced Training" Tab** | UI-based parameter tuning, real-time Log/Chart streaming. |
| **Data Viz** | N/A | **"Data Explorer" Tab** | Pie charts (Target Dist) & Heatmaps (Correlation). |
| **Prediction** | API Endpoint (JSON) | **"Prediction & Ops" Tab** | Smart Form (auto-fill features), Visual Results. |
| **Monitoring** | N/A | **"System Monitor" Tab** | **(M28)** Real-time CPU/RAM/Disk usage visualization. |
| **Robustness** | None | **Auto-Finetune Workflow** | **(M27)** On-the-fly model correction & drift adaptation. |
| **Logging** | Console output | **Auto CSV Logging** | **(M27)** Automatically saves all inferences to `data/inference_history.csv`. |

## 3. New Architecture Components

### 3.1 `main.py` (The Core)
This single file now acts as the orchestrator for the entire system.
*   **System Monitor**: Uses `psutil` to track infrastructure health.
*   **Log Callback**: A custom `GradioLogCallback` intercepts training metrics and pushes them to the UI in real-time.
*   **Smart Preprocessor**: The `predict_smart` function allows users to input just 5 key features (Amount, Card, Distance, etc.) while the system automatically handles the remaining 400+ features using dataset statistics.

### 3.2 Deleted Redundant Code
To minimize technical debt, the following legacy components were removed:
*   `api/` directory (FastAPI backend).
*   `api/frontend/` (HTML/CSS/JS files).

## 4. User Guide

### How to Run
```bash
python main.py
```

### Dashboard Tabs
1.  **🖥️ System Monitor**: Check if your server has enough RAM/CPU.
2.  **📊 Data Explorer**: View your training data quality and distributions.
3.  **🚀 Advanced Training**: Train new models. Support resuming, parameter tweaking, and visual debugging.
4.  **📈 Evaluation**: After training, view Feature Importance (Top 20) and Confusion Matrix.
5.  **🔮 Ops & Logging**:
    *   **Predict**: Manual risk assessment.
    *   **Auto-Finetune**: Fix model errors instantly by retraining on the mistake.
    *   *Note: All actions are logged to `data/inference_history.csv`.*

## 5. MLOps Compliance Checklist
*   ✅ **M27 (Robustness)**: Implemented via Auto-Finetuning workflow.
*   ✅ **M27 (Data Collection)**: Implemented via automated CSV logging (`log_inference`).
*   ✅ **M28 (System Metrics)**: Implemented via System Monitor tab (`psutil`).
