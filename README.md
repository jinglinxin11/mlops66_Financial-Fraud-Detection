<<<<<<< HEAD
# mlops66_Financial-Fraud-Detection
=======
# IEEE-CIS Fraud Detection with TabNet

A financial fraud detection system using TabNet deep learning model for the [IEEE-CIS Fraud Detection Kaggle Competition](https://www.kaggle.com/c/ieee-fraud-detection).

## ✨ Features

- 🧠 **TabNet Model** - Attention-based interpretable deep learning
- 📊 **Complete Preprocessing** - Auto handling of missing values, rare categories, feature encoding
- 🔄 **Checkpoint Support** - Resume training from interruption
- 📈 **Uncertainty Analysis** - Prediction confidence stratification
- 🎯 **Class Imbalance Handling** - Automatic class weight calculation

## 📁 Project Structure

```
mlops_66/
├── train.py              # Training entry point
├── predict.py            # Prediction entry point
├── preprocess.py         # Data preprocessing entry point
├── src/                  # Modular source code
│   ├── config/           # Configuration module
│   │   └── settings.py
│   ├── data/             # Data loading module
│   │   └── loader.py
│   ├── features/         # Feature engineering module
│   │   ├── preprocessor.py
│   │   ├── encoders.py
│   │   └── time_features.py
│   ├── models/           # Model module
│   │   ├── tabnet_trainer.py
│   │   └── callbacks.py
│   ├── evaluation/       # Evaluation module
│   │   ├── metrics.py
│   │   └── uncertainty.py
│   └── utils/            # Utility module
│       └── helpers.py
├── data/                 # Dataset directory
└── checkpoints/          # Model checkpoints
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
conda create -n mlops python=3.9
conda activate mlops
pip install pytorch-tabnet pandas numpy scikit-learn
```

### 2. Data Preparation

Place IEEE-CIS dataset in `data/` directory.

### 3. Run

```bash
# Analyze data quality
python preprocess.py --analyze

# Preprocess data
python preprocess.py

# Train model
python train.py

# Predict (Kaggle submission)
python predict.py
```

## 📖 Usage

### Data Preprocessing

```python
from src.config.settings import Config
from src.features.preprocessor import FraudPreprocessor

config = Config()
preprocessor = FraudPreprocessor(config)
data = preprocessor.fit_transform()
preprocessor.save()
```

### Model Training

```python
from src.models.tabnet_trainer import TabNetTrainer

trainer = TabNetTrainer(config, data)
model = trainer.train()
```

### Model Evaluation

```python
from src.evaluation.metrics import evaluate_model
from src.evaluation.uncertainty import UncertaintyAnalyzer

results = evaluate_model(model, X_test, y_test, feature_columns)
analyzer = UncertaintyAnalyzer()
uncertainty = analyzer.analyze(results['proba'], y_test)
```

## ⚙️ Configuration

Modify parameters in `src/config/settings.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_EPOCHS` | 100 | Maximum training epochs |
| `PATIENCE` | 10 | Early stopping patience |
| `BATCH_SIZE` | 1024 | Batch size |
| `CHECKPOINT_EVERY` | 10 | Checkpoint save interval |
| `RESUME_TRAINING` | True | Resume from checkpoint |

## 📊 Model Performance

- **Test AUC**: ~0.81
- **Top 5 Features**: V230, P_emaildomain, M6, id_11, V154

## 📝 License

MIT License
>>>>>>> feb34cc (initial commit)
