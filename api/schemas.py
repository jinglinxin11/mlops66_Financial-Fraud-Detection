from pydantic import BaseModel
from typing import Dict, Any


class PredictRequest(BaseModel):
    data: Dict[str, Any]


class PredictResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
