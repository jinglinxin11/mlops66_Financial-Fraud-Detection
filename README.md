# IEEE-CIS Fraud Detection with TabNet

A financial fraud detection system using TabNet deep learning model for the [IEEE-CIS Fraud Detection Kaggle Competition](https://www.kaggle.com/c/ieee-fraud-detection).

## ✨ Features

- 🧠 **TabNet Model** - Attention-based interpretable deep learning
- 📊 **Complete Preprocessing** - Auto handling of missing values, rare categories, feature encoding
- 🔄 **Checkpoint Support** - Resume training from interruption
- 📈 **Uncertainty Analysis** - Prediction confidence stratification
- 🎯 **Class Imbalance Handling** - Automatic class weight calculation
- ⚙️ **Hydra Configuration** - Flexible YAML-based configuration management
- 📝 **Weights & Biases** - Experiment tracking and hyperparameter sweeps


## 📁 Project Structure

```
mlops66_Financial-Fraud-Detection/
├── train.py              # Training entry point (Hydra + W&B)
├── predict.py            # Prediction entry point
├── preprocess.py         # Data preprocessing entry point
├── profile_training.py   # Performance profiling script
├── configs/              # Hydra configuration files
│   ├── config.yaml       # Main configuration
│   ├── model/tabnet.yaml # Model hyperparameters
│   ├── training/default.yaml
│   └── sweep.yaml        # W&B hyperparameter sweep
├── src/                  # Modular source code
│   ├── config/           # Configuration module
│   ├── data/             # Data loading module
│   ├── features/         # Feature engineering module
│   ├── models/           # Model module
│   ├── evaluation/       # Evaluation module
│   └── utils/            # Utilities (logging, profiling, wandb)
├── data/                 # Dataset directory (DVC tracked)
├── checkpoints/          # Model checkpoints

└── data.dvc              # DVC data tracking
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
conda create -n mlops python=3.10
conda activate mlops
pip install -r requirements.txt
```

### 2. Data Preparation

Place IEEE-CIS dataset in `data/` directory, or use DVC:

```bash
dvc pull  # If remote is configured
```

### 3. Run Training

```bash
# View all configuration options
python train.py --help

# Train with default config
python train.py

# Train with custom parameters
python train.py training.max_epochs=50 model.n_steps=5

# Train with W&B logging
python train.py wandb.enabled=true
```

### 4. Data Preprocessing (Optional)

```bash
# Analyze data quality
python preprocess.py --analyze

# Preprocess training data
python preprocess.py

# Preprocess test data (for Kaggle submission)
python preprocess.py --test
```

### 5. Prediction

```bash
python predict.py
```


## ⚙️ Configuration

Configuration is managed via Hydra YAML files in `configs/`:

### Model Parameters (`configs/model/tabnet.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_d` | 48 | Decision layer width |
| `n_a` | 48 | Attention layer width |
| `n_steps` | 4 | Number of decision steps |
| `gamma` | 1.5 | Feature reuse coefficient |
| `lambda_sparse` | 0.001 | Sparsity regularization |

### Training Parameters (`configs/training/default.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_epochs` | 100 | Maximum training epochs |
| `patience` | 10 | Early stopping patience |
| `batch_size` | 8192 | Batch size |
| `learning_rate` | 0.005 | Learning rate |
| `checkpoint_every` | 10 | Checkpoint save interval |

### Override Examples

```bash
# Change epochs and batch size
python train.py training.max_epochs=200 training.batch_size=4096

# Use different model config
python train.py model=tabnet model.n_steps=6
```

## 📊 Experiment Tracking (W&B)

```bash
# Login to W&B
wandb login

# Run with experiment tracking
python train.py wandb.enabled=true wandb.project=my-fraud-detection

# Hyperparameter sweep
wandb sweep configs/sweep.yaml
wandb agent <sweep-id>
```

## 🔧 Data Version Control (DVC)

```bash
# Initialize DVC (already done)
dvc init

# Add data to DVC
dvc add data/

# Configure remote storage
dvc remote add -d myremote gs://your-bucket/dvc-storage

# Push/pull data
dvc push
dvc pull
```

## 📈 Profiling

```bash
# Quick profiling test
python profile_training.py --dry-run

# Full training profiling
python profile_training.py
```

## ☁️ Deployment & API

### 1. Docker (Recommended)

Build and run the fraud detection API in a container:

```bash
# Build image
docker build -t fraud-detection-api .

# Run container (mounts data and model from local host for development)
# Windows (PowerShell):
docker run -p 8000:8000 -v ${PWD}:/app fraud-detection-api

# Linux/Mac:
docker run -p 8000:8000 -v $(pwd):/app fraud-detection-api
```

### 2. Standard API

Run the FastAPI service locally:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

- **Health Check**: `http://localhost:8000/`
- **Predict**: `http://localhost:8000/predict_test`

### 3. ONNX High-Performance API

For faster inference using ONNX Runtime:

```bash
# 1. Export model to ONNX
python export_onnx.py

# 2. Run ONNX API
python onnx_api.py
```

- Server runs on `http://localhost:8001`
- **Predict**: `POST /predict` (accepts feature dictionary)
- **Batch Predict**: `POST /predict_batch`

## 🧪 Testing & Validation

```bash
# Run unit and integration tests
pytest

# Run tests with coverage report
pytest --cov=src tests/
```

## 📊 Model Performance

- **Test AUC**: ~0.81
- **Top 5 Features**: V230, P_emaildomain, M6, id_11, V154

## 📝 License

MIT License
