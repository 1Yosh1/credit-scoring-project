"""Tests for feature engineering parity between training and serving."""
from __future__ import annotations

import pandas as pd
import pytest

from src.features import (
    MODEL_FEATURES,
    build_features,
    decide_risk,
    scale_features,
)


@pytest.fixture()
def raw() -> dict:
    return {
        "AMT_INCOME_TOTAL": 202_500.0,
        "AMT_CREDIT": 406_597.5,
        "AMT_ANNUITY": 24_700.5,
        "DAYS_BIRTH": -9_461,
        "CODE_GENDER": "F",
        "EXT_SOURCE_3": 0.14,
    }


def test_build_features_produces_expected_columns(raw):
    features = build_features(raw)
    assert list(features.columns) == MODEL_FEATURES
    assert len(features) == 1


def test_financial_ratios_match_training_formulas(raw):
    features = build_features(raw).iloc[0]
    assert features["CREDIT_INCOME_PERCENT"] == pytest.approx(raw["AMT_CREDIT"] / raw["AMT_INCOME_TOTAL"])
    assert features["ANNUITY_INCOME_PERCENT"] == pytest.approx(raw["AMT_ANNUITY"] / raw["AMT_INCOME_TOTAL"])
    assert features["CREDIT_TERM"] == pytest.approx(raw["AMT_ANNUITY"] / raw["AMT_CREDIT"])


def test_bureau_score_becomes_the_mean_feature(raw):
    features = build_features(raw).iloc[0]
    assert features["EXT_SOURCE_MEAN"] == raw["EXT_SOURCE_3"]


def test_no_invented_constant_features(raw):
    # Serving must not fabricate values for fields the form does not collect.
    features = build_features(raw)
    assert "DAYS_EMPLOYED" not in features.columns
    assert "EXT_SOURCE_1" not in features.columns


def test_gender_is_not_a_model_feature(raw):
    # The committed model was trained on numeric columns only, so gender is
    # validated at the API boundary (see tests/test_schemas.py) and ignored
    # during feature construction.
    features = build_features({**raw, "CODE_GENDER": "X"})
    assert "CODE_GENDER_F" not in features.columns


def test_scale_features_uses_fitted_columns():
    from sklearn.preprocessing import StandardScaler

    df = pd.DataFrame(
        {
            "AMT_INCOME_TOTAL": [1.0],
            "AMT_CREDIT": [1.0],
            "AMT_ANNUITY": [1.0],
            "DAYS_BIRTH": [1.0],
            "CREDIT_INCOME_PERCENT": [2.0],
            "ANNUITY_INCOME_PERCENT": [0.1],
            "CREDIT_TERM": [0.05],
            "EXT_SOURCE_MEAN": [0.5],
        }
    )
    scaler = StandardScaler().fit(df)
    scaled = scale_features(df, scaler)
    assert scaled["CREDIT_INCOME_PERCENT"].iloc[0] == pytest.approx(0.0, abs=1e-7)


def test_decide_risk_threshold():
    assert decide_risk(0.21, threshold=0.20) == "High"
    assert decide_risk(0.19, threshold=0.20) == "Low"
