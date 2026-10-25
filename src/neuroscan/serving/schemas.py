"""Response models of the prediction API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Status = Literal["ok", "uncertain", "rejected_ood", "rejected_input"]

DISCLAIMER = "Research demo, not a medical device. Do not use for diagnosis."


class Prediction(BaseModel):
    model_version: str
    status: Status
    reason: str | None = Field(None, description="why the input was rejected, if it was")
    prediction: str | None = Field(None, description="null when the input was rejected")
    confidence: float | None = None
    probabilities: dict[str, float] | None = None
    calibrated: bool = True
    ood_score: float | None = None
    ood_threshold: float | None = None
    latency_ms: float
    input_png_base64: str | None = Field(None, description="the 224x224 image the model saw")
    heatmap_png_base64: str | None = Field(None, description="Grad-CAM overlay (RGBA)")
    disclaimer: str = DISCLAIMER


class BatchPrediction(BaseModel):
    model_version: str
    results: list[Prediction]


class ModelInfo(BaseModel):
    model_uri: str
    model_version: str
    classes: list[str]
    temperature: float | None
    uncertain_threshold: float | None
    ood_threshold: float | None
    metrics: dict[str, float] = Field(
        default_factory=dict, description="headline test metrics of the served model's run"
    )
    disclaimer: str = DISCLAIMER


class Error(BaseModel):
    error: str
    detail: str | None = None
