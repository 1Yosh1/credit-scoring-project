"""Tests for the model service (loading + scoring)."""
from __future__ import annotations

import pytest

from src.model import load_model_bundle


def test_bundle_loads_with_synthetic_artifacts(model_dir):
    bundle = load_model_bundle()
    assert bundle.model is not None
    assert len(bundle.columns) == 8
    assert bundle.scaler is not None
    assert bundle.metadata["model_version"] == "test-1.0"


def test_missing_artifacts_raise_helpful_error(tmp_path, monkeypatch):
    import src.model as model_module

    monkeypatch.setattr(model_module, "MODEL_PATH", tmp_path / "nope.pkl")
    monkeypatch.setattr(model_module, "MODEL_COLUMNS_PATH", tmp_path / "nope_cols.pkl")
    model_module.load_model_bundle.cache_clear()
    try:
        with pytest.raises(FileNotFoundError, match="make train"):
            load_model_bundle()
    finally:
        model_module.load_model_bundle.cache_clear()


def test_score_returns_probability_and_risk(model_dir, sample_application):
    bundle = load_model_bundle()
    result = bundle.score(sample_application, threshold=0.20)

    assert set(result) == {"default_probability", "risk_level", "threshold"}
    assert 0.0 <= result["default_probability"] <= 1.0
    assert result["risk_level"] in ("High", "Low")


def test_low_score_maps_to_high_risk(model_dir, sample_application):
    # The synthetic tree flags EXT_SOURCE_3 < 0.5 as default.
    bundle = load_model_bundle()
    result = bundle.score(sample_application, threshold=0.20)
    assert result["risk_level"] == "High"
    assert result["default_probability"] > 0.5


def test_high_score_maps_to_low_risk(model_dir, sample_application):
    bundle = load_model_bundle()
    application = {**sample_application, "EXT_SOURCE_3": 0.9}
    result = bundle.score(application, threshold=0.20)
    assert result["risk_level"] == "Low"
