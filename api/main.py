"""FastAPI application for fraud detection predictions."""

from fastapi import FastAPI

from src.config.settings import Config
from src.features.preprocessor import FraudPreprocessor
from src.models.tabnet_trainer import TabNetTrainer

app = FastAPI(title="Fraud Detection API", version="1.0")


@app.get("/")
def root():
    """Return the API health status."""
    return {"status": "running"}


@app.post("/predict_test")
def predict_test(limit: int = 5):
    """Run preprocessing on the Kaggle test set and return the first N predictions."""
    config = Config()

    # Load preprocessor + model (same as predict.py)
    preprocessor = FraudPreprocessor(config)
    preprocessor.load()

    trainer = TabNetTrainer(config, data=None)
    model = trainer.load()

    # Transform test set
    test_data = preprocessor.transform()

    # Predict probabilities
    proba = model.predict_proba(test_data["X_test"])[:, 1]

    # Return first N predictions
    out = []
    for tid, p in zip(test_data["transaction_ids"][:limit], proba[:limit]):
        out.append(
            {
                "TransactionID": int(tid),
                "fraud_probability": float(p),
                "is_fraud": bool(p >= 0.5),
            }
        )

    return {"count": limit, "predictions": out}
