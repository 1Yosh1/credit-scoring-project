"""Shared pytest fixtures: a tiny synthetic model bundle for serving tests."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.features import MODEL_FEATURES, SCALED_FEATURES


@pytest.fixture()
def model_dir(tmp_path, monkeypatch):
    """Create synthetic model artifacts and point src.model at them.

    The model learns y = (EXT_SOURCE_MEAN < 0.5) so /predict probabilities vary
    meaningfully with the bureau score. We deliberately use an XGBClassifier
    (not an sklearn tree) so the artifact matches production: binary:logistic
    XGBoost, whose SHAP values live in log-odds space -- the space the
    explanation layer's sigmoid walk assumes.
    """
    rng = np.random.default_rng(42)
    n = 300
    X = pd.DataFrame(
        rng.normal(loc=[50_000, 200_000, 10_000, -11_000, 2.0, 0.05, 0.25, 0.55],
                   scale=[1, 1, 1, 1, 0.5, 0.1, 0.1, 0.2],
                   size=(n, len(MODEL_FEATURES))),
        columns=MODEL_FEATURES,
    )
    X["EXT_SOURCE_MEAN"] = rng.uniform(0.01, 1.0, n)
    y = (X["EXT_SOURCE_MEAN"] < 0.5).astype(int)

    scaler = StandardScaler().fit(X[SCALED_FEATURES])
    model = XGBClassifier(
        n_estimators=20, max_depth=2, learning_rate=0.3,
        eval_metric="logloss", random_state=42, n_jobs=1,
    ).fit(X, y)

    joblib.dump(model, tmp_path / "credit_model.pkl")
    joblib.dump(list(X.columns), tmp_path / "model_columns.pkl")
    joblib.dump(scaler, tmp_path / "scaler.pkl")
    (tmp_path / "metrics.json").write_text(
        json.dumps(
            {
                "model_version": "test-1.0",
                "trained_at": "2025-01-01T00:00:00+00:00",
                "metrics": {"roc_auc": 0.5, "accuracy": 0.5, "recall": 0.5},
            }
        )
    )

    import src.model as model_module

    monkeypatch.setattr(model_module, "MODEL_PATH", tmp_path / "credit_model.pkl")
    monkeypatch.setattr(model_module, "MODEL_COLUMNS_PATH", tmp_path / "model_columns.pkl")
    monkeypatch.setattr(model_module, "SCALER_PATH", tmp_path / "scaler.pkl")
    monkeypatch.setattr(model_module, "METRICS_PATH", tmp_path / "metrics.json")
    model_module.load_model_bundle.cache_clear()

    yield tmp_path

    model_module.load_model_bundle.cache_clear()


@pytest.fixture()
def sample_application() -> dict:
    """A valid loan application payload."""
    return {
        "AMT_INCOME_TOTAL": 202_500.0,
        "AMT_CREDIT": 406_597.5,
        "AMT_ANNUITY": 24_700.5,
        "DAYS_BIRTH": -9_461,
        "CODE_GENDER": "F",
        "EXT_SOURCE_3": 0.14,
    }
