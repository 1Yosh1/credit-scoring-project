"""Model loading and inference.

Kept separate from the FastAPI layer so inference logic can be unit-tested
without spinning up the app.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache

import joblib

from src.config import METRICS_PATH, MODEL_COLUMNS_PATH, MODEL_PATH, SCALER_PATH
from src.explain import explain_prediction
from src.features import build_features, decide_risk, scale_features

logger = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    """All artifacts needed to score a loan application."""

    model: object
    columns: list[str]
    scaler: object | None = None
    metadata: dict = field(default_factory=dict)

    def predict_default_probability(self, application: dict) -> float:
        """Score a raw application dict and return P(default)."""
        features = build_features(application)
        if self.scaler is not None:
            features = scale_features(features, self.scaler)
        aligned = features.reindex(columns=self.columns, fill_value=0)
        proba = self.model.predict_proba(aligned)[0, 1]
        return float(proba)

    def score(self, application: dict, threshold: float) -> dict:
        """Full business response: probability, risk level, and explanation."""
        probability = self.predict_default_probability(application)
        result = {
            "default_probability": probability,
            "risk_level": decide_risk(probability, threshold),
            "threshold": threshold,
        }
        try:
            result["explanation"] = self.explain(application)
        except Exception:  # noqa: BLE001 - never fail a prediction over its explanation
            logger.warning("Explanation failed; serving prediction without it", exc_info=True)
        return result

    def explain(self, application: dict) -> dict:
        """SHAP-based top risk factors for one application."""
        features = build_features(application)
        scaled = scale_features(features, self.scaler) if self.scaler is not None else features
        return explain_prediction(self.model, scaled, raw_values=features.iloc[0].to_dict())


@lru_cache(maxsize=1)
def load_model_bundle() -> ModelBundle:
    """Load model artifacts once per process.

    Raises FileNotFoundError with a helpful message if artifacts are missing.
    """
    missing = [
        str(p)
        for p in (MODEL_PATH, MODEL_COLUMNS_PATH)
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Model artifacts not found: "
            + ", ".join(missing)
            + ". Train the model first: `make train` (see README)."
        )

    model = joblib.load(MODEL_PATH)
    columns = list(joblib.load(MODEL_COLUMNS_PATH))
    scaler = None
    if SCALER_PATH.exists():
        scaler = joblib.load(SCALER_PATH)

    metadata: dict = {}
    if METRICS_PATH.exists():
        try:
            metadata = json.loads(METRICS_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning("Could not parse %s; serving without metadata", METRICS_PATH)

    logger.info("Model loaded: %s features", len(columns))
    return ModelBundle(model=model, columns=columns, scaler=scaler, metadata=metadata)
