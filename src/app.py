"""FastAPI application exposing the credit risk scoring model."""
from __future__ import annotations

import logging

from fastapi import Depends, FastAPI, HTTPException

from src.config import DEFAULT_RISK_THRESHOLD, LOG_LEVEL
from src.model import ModelBundle, load_model_bundle
from src.schemas import LoanApplication, PredictionResponse

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Credit Risk Scoring API",
    description="Scores loan applications for probability of default (Home Credit dataset).",
    version="1.1.0",
)


def get_bundle() -> ModelBundle:
    """Resolve the model bundle, returning a clean 503 if artifacts are missing."""
    try:
        return load_model_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def get_threshold(bundle: ModelBundle) -> float:
    """Serving threshold: calibrated value from training, with a fallback."""
    metrics = bundle.metadata.get("metrics", {})
    if "threshold" in metrics:
        return float(metrics["threshold"])
    return DEFAULT_RISK_THRESHOLD


@app.get("/")
def home() -> dict:
    return {"message": "Credit Risk API is live", "docs": "/docs"}


@app.get("/health")
def health() -> dict:
    """Liveness/readiness probe for Docker and CI. Also warms the model cache."""
    try:
        load_model_bundle()
        return {"status": "ok"}
    except FileNotFoundError:
        return {"status": "model_not_loaded"}


@app.get("/model-info")
def model_info(bundle: ModelBundle = Depends(get_bundle)) -> dict:
    """Serve the metrics recorded at training time, when available."""
    return {
        "model_version": bundle.metadata.get("model_version"),
        "trained_at": bundle.metadata.get("trained_at"),
        "metrics": bundle.metadata.get("metrics", {}),
        "n_features": len(bundle.columns),
        "risk_threshold": get_threshold(bundle),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_risk(
    application: LoanApplication,
    bundle: ModelBundle = Depends(get_bundle),
) -> dict:
    """Score one loan application and return P(default) with a risk label."""
    threshold = get_threshold(bundle)
    try:
        return bundle.score(application.model_dump(), threshold)
    except ZeroDivisionError as exc:
        raise HTTPException(status_code=422, detail="Income and credit amount must be positive") from exc
    except Exception as exc:  # noqa: BLE001 - last-resort guard for the request path
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Internal prediction error") from exc
