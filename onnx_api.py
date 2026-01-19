"""
ONNX Inference API
FastAPI application for fraud detection using ONNX Runtime

Usage:
    uvicorn onnx_api:app --host 0.0.0.0 --port 8001
"""

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from pathlib import Path
import pickle

app = FastAPI(
    title="Fraud Detection API (ONNX)",
    description="High-performance fraud detection using ONNX Runtime",
    version="1.0"
)

# Global variables for model and preprocessor
ort_session = None
preprocessor = None
feature_columns = None


class TransactionData(BaseModel):
    """Single transaction data for prediction."""
    features: Dict[str, Any]


class BatchTransactionData(BaseModel):
    """Batch transaction data for prediction."""
    transactions: List[Dict[str, Any]]


class PredictionResponse(BaseModel):
    """Prediction response."""
    fraud_probability: float
    is_fraud: bool
    risk_level: str


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResponse]
    count: int


def get_risk_level(prob: float) -> str:
    """Classify risk level based on probability."""
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


@app.on_event("startup")
async def load_model():
    """Load ONNX model and preprocessor on startup."""
    global ort_session, preprocessor, feature_columns
    
    onnx_path = Path("tabnet_fraud_model.onnx")
    preprocessor_path = Path("ieee_cis_preprocessor.pkl")
    
    if not onnx_path.exists():
        print(f"⚠️ ONNX model not found: {onnx_path}")
        print("   Run: python export_onnx.py")
        return
    
    # Load ONNX model
    print(f"Loading ONNX model from: {onnx_path}")
    ort_session = ort.InferenceSession(str(onnx_path))
    print("✅ ONNX model loaded")
    
    # Load preprocessor (optional, for feature processing)
    if preprocessor_path.exists():
        print(f"Loading preprocessor from: {preprocessor_path}")
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        print("✅ Preprocessor loaded")


@app.get("/")
def root():
    """Health check endpoint."""
    model_loaded = ort_session is not None
    return {
        "status": "running",
        "model_loaded": model_loaded,
        "backend": "ONNX Runtime"
    }


@app.get("/health")
def health():
    """Detailed health check."""
    if ort_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model": "tabnet_fraud_model.onnx",
        "backend": "ONNX Runtime",
        "providers": ort_session.get_providers()
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: TransactionData):
    """
    Predict fraud probability for a single transaction.
    
    Input: feature vector as dictionary
    Output: fraud probability and classification
    """
    if ort_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert features to numpy array
        # Expecting already preprocessed features
        features = np.array(list(data.features.values()), dtype=np.float32)
        features = features.reshape(1, -1)
        
        # Run inference
        input_name = ort_session.get_inputs()[0].name
        outputs = ort_session.run(None, {input_name: features})
        
        # Get probability (softmax output)
        logits = outputs[0][0]
        proba = float(np.exp(logits[1]) / np.sum(np.exp(logits)))
        
        return PredictionResponse(
            fraud_probability=proba,
            is_fraud=proba >= 0.5,
            risk_level=get_risk_level(proba)
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict_batch", response_model=BatchPredictionResponse)
def predict_batch(data: BatchTransactionData):
    """
    Predict fraud probability for multiple transactions.
    
    Input: list of feature vectors
    Output: list of predictions
    """
    if ort_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to numpy array
        features_list = [list(t.values()) for t in data.transactions]
        features = np.array(features_list, dtype=np.float32)
        
        # Run inference
        input_name = ort_session.get_inputs()[0].name
        outputs = ort_session.run(None, {input_name: features})
        
        # Process outputs
        logits = outputs[0]
        probas = np.exp(logits[:, 1]) / np.sum(np.exp(logits), axis=1)
        
        predictions = [
            PredictionResponse(
                fraud_probability=float(p),
                is_fraud=p >= 0.5,
                risk_level=get_risk_level(p)
            )
            for p in probas
        ]
        
        return BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions)
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/model_info")
def model_info():
    """Get model information."""
    if ort_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    inputs = ort_session.get_inputs()
    outputs = ort_session.get_outputs()
    
    return {
        "inputs": [
            {"name": i.name, "shape": i.shape, "type": i.type}
            for i in inputs
        ],
        "outputs": [
            {"name": o.name, "shape": o.shape, "type": o.type}
            for o in outputs
        ],
        "providers": ort_session.get_providers()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
