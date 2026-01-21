# IEEE-CIS Fraud Detection with TabNet

A financial fraud detection system using TabNet deep learning model for the [IEEE-CIS Fraud Detection Kaggle Competition](https://www.kaggle.com/c/ieee-fraud-detection).

## ✨ Features

- 🧠 **TabNet Model** - Attention-based interpretable deep learning
- 📊 **Complete Preprocessing** - Auto handling of missing values, rare categories, feature encoding
- 🔄 **Checkpoint Support** - Resume training from interruption
- 📈 **Uncertainty Analysis** - Prediction confidence stratification
- 🎯 **Class Imbalance Handling** - Automatic class weight calculation
- 🌐 **FastAPI Inference API (M22)** - Run inference through a simple API endpoint
- ⚙️ **Hydra Configuration** - Flexible YAML-based configuration management
- 📝 **Weights & Biases** - Experiment tracking and hyperparameter sweeps


## 📁 Project Structure
```bash
├── api/                       # FastAPI application
│   ├── main.py                # Entry point for the API application
│   └── schemas.py             # Pydantic models for data validation
├── reports/                   # Reporting modules
│   ├── figures/               # Generated plots and visualizations
│   └── report.py              # Script to generate performance reports
├── src/                       # Modular source code
│   ├── config/                # Configuration module (settings.py)
│   ├── data/                  # Data loading module (loader.py)
│   ├── evaluation/            # Evaluation module (metrics & uncertainty)
│   ├── features/              # Feature engineering (preprocessor, encoders, time_features)
│   ├── models/                # Model architecture & training (TabNet, callbacks)
│   └── utils/                 # Utility module (helpers.py)
├── tests/                     # Unit & Integration tests (Pytest)
├── data/                      # Dataset directory (Kaggle files go here)
├── checkpoints/               # Model checkpoints storage
├── Dockerfile                 # Docker configuration for containerization
├── docker-entrypoint.sh       # Entry script for Docker container
├── locustfile.py              # Load testing configuration (Locust)
├── train.py                   # Training entry point
├── predict.py                 # Prediction entry point (Kaggle submission)
├── preprocess.py              # Data preprocessing entry point
├── pyproject.toml             # Project configuration & dependencies
├── requirements.txt           # Production dependencies
├── requirements_tests.txt     # Development/Test dependencies
├── ieee_cis_preprocessor.pkl  # Serialized preprocessor object
└── tabnet_fraud_model.zip     # Compressed model artifact
```

## 🚀 Quick Start

### 1. 🛠️ Environment Setup

Start by installing the required dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. 🗄️ Data Preparation

Download the **IEEE-CIS Fraud Detection** dataset from Kaggle and place the following files in the `data/` directory:

* `train_transaction.csv`
* `train_identity.csv`
* `test_transaction.csv`
* `test_identity.csv`
* `sample_submission.csv`

### 3. 🚀 Run the Pipeline

You can run the different stages of the pipeline using the commands below:
```bash
# 1. Analyze data quality (optional)
python preprocess.py --analyze
```
```bash
# 2. Preprocess data (clean & engineer features)
python preprocess.py
```
```bash
# 3. Train the TabNet model
python train.py
```
```bash
# 4. Generate predictions for Kaggle submission
python predict.py
```

### 4. 🌐 FastAPI Inference API (M22)

Deploy the model as a REST API for real-time inference.
```bash
# Start the server
python -m uvicorn api.main:app --reload
```

📖 Swagger documentation: http://127.0.0.1:8000/docs

**Demo Inference Endpoint**

Runs preprocessing + TabNet inference on the Kaggle test set and returns the first `limit` predictions:
```http
POST /predict_test?limit=5
```

Example response fields:
- `TransactionID`
- `fraud_probability`
- `is_fraud`

### 5. 🐳 Docker Support

You can build and run the project inside a Docker container to ensure a consistent environment.
```bash
docker build -t fraud-detection-app .
```
```bash
# Runs the API on port 8000
docker run -p 8000:8000 fraud-detection-app
```

### 6. 🧪 Development & Testing

Run Unit Tests: Ensure the logic works correctly by running the test suite:
```bash
pytest tests/
```
Run Load Tests (Locust): Simulate user traffic to test API performance:
```bash
# 1. Start the API first
python -m uvicorn api.main:app --reload

# 2. In a separate terminal, run Locust
locust -f locustfile.py --host=[http://127.0.0.1:8000](http://127.0.0.1:8000)
```
Then open http://localhost:8089 in your browser.

## ⚙️ Configuration

You can modify training parameters in `src/config/settings.py`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_EPOCHS` | 100 | Maximum training epochs |
| `PATIENCE` | 10 | Early stopping patience |
| `BATCH_SIZE` | 8192 | Batch size |
| `CHECKPOINT_EVERY` | 10 | Checkpoint save interval |
| `RESUME_TRAINING` | True | Resume from checkpoint |

## 📊 Model Performance

**Test AUC:** ~0.81

**Top 5 Features:** V230, P_emaildomain, M6, id_11, V154

## 📝 License

MIT License
