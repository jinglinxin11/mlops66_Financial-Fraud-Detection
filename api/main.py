"""FastAPI application for fraud detection predictions."""

from typing import Annotated

from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.drift_api import log_prediction_to_gcs
from api.drift_api import router as drift_router
from src.config.settings import Config
from src.features.preprocessor import FraudPreprocessor
from src.models.tabnet_trainer import TabNetTrainer

app = FastAPI(title="Fraud Detection API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals
app.include_router(drift_router)
config = Config()
preprocessor = None
model = None
test_data_cache = None


def ensure_loaded():
    """Ensure model + preprocessor + cached test data are loaded."""
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
def predict_test(
    limit: int = 5, background_tasks: Annotated[BackgroundTasks, None] = BackgroundTasks()
):
    """Predict fraud probabilities on cached Kaggle test transactions."""
    ensure_loaded()

    proba = model.predict_proba(test_data_cache["X_test"])[:, 1]

    # Return first N predictions
    out = []
    for tid, p in zip(test_data_cache["transaction_ids"][:limit], proba[:limit]):
        pred = {
            "TransactionID": int(tid),
            "fraud_probability": float(p),
            "is_fraud": bool(p >= 0.5),
        }
        out.append(pred)

        # M27 logging (in background)
        if background_tasks is not None:
            background_tasks.add_task(log_prediction_to_gcs, pred, float(p))

    return {"count": limit, "predictions": out}
