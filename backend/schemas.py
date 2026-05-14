from typing import Literal

from pydantic import BaseModel, Field


class Thresholds(BaseModel):
    ai_threshold: float = Field(..., ge=0.0, le=1.0)
    uncertainty_margin: float = Field(..., ge=0.0, le=1.0)


class ModelInfo(BaseModel):
    provider: str
    backend: str
    model_dir: str | None
    device: str
    loaded: bool


class HealthResponse(BaseModel):
    status: Literal["ok"]
    app_name: str
    app_version: str
    model: ModelInfo


class PredictionResponse(BaseModel):
    filename: str | None
    label: Literal["AI Generated", "Real", "Uncertain"]
    raw_label: Literal["ai", "real", "uncertain"]
    upstream_label: Literal["ai", "real"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    ai_probability: float = Field(..., ge=0.0, le=1.0)
    real_probability: float = Field(..., ge=0.0, le=1.0)
    thresholds: Thresholds
    model: ModelInfo


class MessageResponse(BaseModel):
    message: str
    health_url: str
    predict_url: str
