"""Tests for SHAP-based prediction explanations."""
from __future__ import annotations

import math

from src.explain import explain_prediction, format_feature_value
from src.features import build_features, scale_features
from src.model import load_model_bundle


def _explain(bundle, application):
    features = build_features(application)
    scaled = scale_features(features, bundle.scaler)
    return explain_prediction(bundle.model, scaled, raw_values=features.iloc[0].to_dict())


def test_explain_prediction_structure(model_dir, sample_application):
    bundle = load_model_bundle()
    explanation = _explain(bundle, sample_application)

    assert set(explanation) == {"base_probability", "probability_additive", "top_factors"}
    assert 0 <= explanation["base_probability"] <= 1
    assert 0 <= explanation["probability_additive"] <= 1
    assert 1 <= len(explanation["top_factors"]) <= 3


def test_factor_fields_and_additivity(model_dir, sample_application):
    bundle = load_model_bundle()
    explanation = _explain(bundle, sample_application)

    for factor in explanation["top_factors"]:
        assert factor["label"]
        assert factor["direction"] in ("increases_risk", "decreases_risk")
        assert math.isfinite(factor["impact_pp"])
        assert "value" in factor

    # Contributions must reconstruct the model's probability from the baseline.
    proba = bundle.predict_default_probability(sample_application)
    assert math.isclose(
        explanation["probability_additive"], proba, abs_tol=5e-3
    )


def test_strong_bureau_score_reduces_risk(model_dir, sample_application):
    bundle = load_model_bundle()
    application = {**sample_application, "EXT_SOURCE_3": 0.9}
    explanation = _explain(bundle, application)

    bureau = next(
        f for f in explanation["top_factors"] if f["feature"] == "EXT_SOURCE_MEAN"
    )
    assert bureau["direction"] == "decreases_risk"
    assert bureau["impact_pp"] < 0
    assert bureau["value"] == "0.90"


def test_format_feature_value():
    assert format_feature_value("AMT_CREDIT", 450000.0) == "$450,000"
    assert format_feature_value("DAYS_BIRTH", -9490) == "26 yrs"
    assert format_feature_value("EXT_SOURCE_MEAN", 0.1) == "0.10"
