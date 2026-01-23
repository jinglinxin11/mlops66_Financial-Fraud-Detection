"""BentoML Service for Fraud Detection.

Provides production-ready ML service with BentoML.

Usage:
    # Save model to BentoML
    python bentoml_service.py save

    # Serve the model
    bentoml serve bentoml_service:svc

    # Build a bento
    bentoml build
"""

import argparse
from pathlib import Path

import numpy as np

# Check if bentoml is installed
try:
    import bentoml
    from bentoml.io import JSON, NumpyNdarray

    BENTOML_AVAILABLE = True
except ImportError:
    BENTOML_AVAILABLE = False
    print("⚠️ BentoML not installed. Run: pip install bentoml")


def save_model_to_bentoml():
    """Save TabNet model to BentoML model store."""
    if not BENTOML_AVAILABLE:
        print("Please install BentoML: pip install bentoml")
        return

    from pytorch_tabnet.tab_model import TabNetClassifier

    from src.config.settings import Config

    config = Config()
    model_path = str(config.MODEL_PATH) + ".zip"

    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("   Run train.py first to train a model")
        return

    print(f"Loading model from: {model_path}")
    model = TabNetClassifier()
    model.load_model(model_path)

    # Save to BentoML
    print("Saving model to BentoML...")
    saved_model = bentoml.picklable_model.save_model(
        "fraud_detection_tabnet",
        model,
        signatures={"predict": {"batchable": True}, "predict_proba": {"batchable": True}},
        labels={"framework": "pytorch-tabnet", "task": "fraud-detection"},
        metadata={
            "input_dim": model.input_dim,
            "description": "IEEE-CIS Fraud Detection TabNet Model",
        },
    )

    print(f"✅ Model saved: {saved_model}")
    return saved_model


# BentoML Service Definition
if BENTOML_AVAILABLE:
    # Try to load the model
    try:
        fraud_model_runner = bentoml.picklable_model.get(
            "fraud_detection_tabnet:latest"
        ).to_runner()

        svc = bentoml.Service("fraud_detection_service", runners=[fraud_model_runner])

        @svc.api(input=NumpyNdarray(), output=JSON())
        async def predict(input_data: np.ndarray) -> dict:
            """Predict fraud probability.

            Input: numpy array of shape (n_samples, n_features)
            Output: predictions with probabilities
            """
            # Ensure float32
            input_data = input_data.astype(np.float32)

            # Get predictions
            predictions = await fraud_model_runner.predict.async_run(input_data)
            probas = await fraud_model_runner.predict_proba.async_run(input_data)

            results = []
            for _, (pred, proba) in enumerate(zip(predictions, probas)):
                fraud_prob = float(proba[1])
                results.append(
                    {
                        "prediction": int(pred),
                        "fraud_probability": fraud_prob,
                        "is_fraud": fraud_prob >= 0.5,
                        "risk_level": get_risk_level(fraud_prob),
                    }
                )

            return {"predictions": results, "count": len(results)}

        @svc.api(input=JSON(), output=JSON())
        async def health(input_data: dict) -> dict:
            """Health check endpoint."""
            return {
                "status": "healthy",
                "service": "fraud_detection_service",
                "framework": "BentoML",
            }

        def get_risk_level(prob: float) -> str:
            """Determine risk level based on fraud probability."""
            if prob >= 0.8:
                return "HIGH"
            elif prob >= 0.5:
                return "MEDIUM-HIGH"
            elif prob >= 0.3:
                return "MEDIUM"
            elif prob >= 0.1:
                return "LOW"
            else:
                return "VERY-LOW"

    except bentoml.exceptions.NotFound:
        print("⚠️ Model not found in BentoML store.")
        print("   Run: python bentoml_service.py save")
        svc = None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="BentoML Fraud Detection Service")
    parser.add_argument("action", choices=["save", "info"], help="Action to perform")
    args = parser.parse_args()

    if args.action == "save":
        save_model_to_bentoml()
    elif args.action == "info":
        if BENTOML_AVAILABLE:
            models = bentoml.models.list()
            print("Available models in BentoML store:")
            for m in models:
                print(f"  - {m.tag}")
        else:
            print("BentoML not installed")


if __name__ == "__main__":
    main()
