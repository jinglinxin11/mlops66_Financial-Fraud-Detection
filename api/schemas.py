"""Pydantic schemas for API request and response models."""

from typing import Any, Dict

from pydantic import BaseModel


class PredictRequest(BaseModel):
    """Request schema for prediction endpoint."""

    data: Dict[str, Any]


class PredictResponse(BaseModel):
    """Response schema for prediction endpoint."""

    fraud_probability: float
    is_fraud: bool
