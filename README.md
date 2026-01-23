# 🛡️ Financial Fraud Detection System (Ultimate Edition)

An enterprise-grade MLOps platform for detecting financial fraud. Now upgraded with a **Unified Gradio Dashboard** that integrates Training, Evaluation, Monitoring, and Operations into a single interface.

## 🌟 Key Features
*   **Unified Control Plane**: One `main.py` to rule them all. No separate frontend/backend processes.
*   **Interactive Training**: Adjust hyperparameters (Epochs, LR, Architecture) and watch real-time loss/AUC curves.
*   **Robustness (M27)**: Auto-Finetuning workflow to adapt to data drift on-the-fly.
*   **Observability (M28)**: Built-in System Monitor (CPU/RAM/Disk) and automated CSV inference logging.
*   **Explainability**: Feature Importance visualization and Confusion Matrices.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
# Ensure you have torch, gradio, plotly, psutil installed
```

### 2. Run the Dashboard
```bash
python main.py
```
*The dashboard will auto-launch in your browser at http://127.0.0.1:7860*

## 📚 Documentation
*   [📄 Project Upgrade Report](docs/PROJECT_UPDATE_REPORT.md): Detailed explanation of new features and architecture changes.
*   [📄 Monitoring Guide](docs/monitoring.md): Guide to drift detection and system metrics.

## 📂 Project Structure
*   `main.py`: **The Entry Point**. Contains the Gradio UI, Training Logic, and System Monitors.
*   `src/`: Core logic (Preprocessing, TabNet Model, Config).
*   `data/`: Data storage (Place your `train_transaction.csv` here).
*   `docs/`: Documentation.

## 🛠️ MLOps Compliance
This project satisfies key MLOps requirements:
*   **Data Collection (M27)**: All predictions are logged to `data/inference_history.csv`.
*   **Robustness (M27)**: "Correct & Learn" feature allows instant feedback loops.
*   **System Monitoring (M28)**: Real-time resource usage tab.

---
*Created by Antigravity*
