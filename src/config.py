"""Central configuration for the credit scoring service."""
from __future__ import annotations

import os
from pathlib import Path

# Repository root (the parent of the src/ package)
REPO_ROOT = Path(__file__).resolve().parent.parent

# Model artifacts directory (baked into the Docker image or present locally)
MODEL_DIR = Path(os.getenv("MODEL_DIR", REPO_ROOT / "models"))
MODEL_PATH = MODEL_DIR / "credit_model.pkl"
MODEL_COLUMNS_PATH = MODEL_DIR / "model_columns.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

# Fallback decision threshold used only when artifacts were trained before
# threshold calibration existed. Current training writes the calibrated
# serving threshold into models/metrics.json and the API reads it from there.
DEFAULT_RISK_THRESHOLD = 0.20

# API
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
