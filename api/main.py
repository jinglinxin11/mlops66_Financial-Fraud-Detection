"""FastAPI application for fraud detection predictions."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config.settings import Config
from src.features.preprocessor import FraudPreprocessor
from src.models.tabnet_trainer import TabNetTrainer

app = FastAPI(title="Fraud Detection API", version="1.0")

# Enable CORS (so browser frontend can call the API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals
config = Config()
preprocessor = None
model = None
test_data_cache = None


def ensure_loaded():
    """Ensure model + preprocessor + cached test data are loaded.

    This is needed because startup events may not run in some CI/pytest setups.
    """
    global preprocessor, model, test_data_cache

    if preprocessor is None:
        preprocessor = FraudPreprocessor(config)
        preprocessor.load()

    if model is None:
        trainer = TabNetTrainer(config, data=None)
        model = trainer.load()

    if test_data_cache is None:
        test_data_cache = preprocessor.transform()


@app.get("/")
def root():
    """Root endpoint returning service status."""
    return {"status": "running"}


@app.get("/health")
def health():
    """Health check endpoint used for monitoring."""
    return {"status": "ok"}


@app.post("/predict_test")
def predict_test(limit: int = 5):
    """Predict fraud probabilities on cached Kaggle test transactions.

    Args:
        limit: Number of predictions to return.

    Returns:
        Dict with count and a list of predictions.
    """
    ensure_loaded()

    proba = model.predict_proba(test_data_cache["X_test"])[:, 1]

    # Return first N predictions
    out = []
    for tid, p in zip(test_data_cache["transaction_ids"][:limit], proba[:limit]):
        out.append(
            {
                "TransactionID": int(tid),
                "fraud_probability": float(p),
                "is_fraud": bool(p >= 0.5),
            }
        )

    return {"count": limit, "predictions": out}
