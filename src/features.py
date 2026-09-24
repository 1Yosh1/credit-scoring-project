"""Feature engineering and preprocessing shared by training and serving.

The model is trained on exactly the features the application form can collect
(train/serve parity by construction). One design decision to know about:

- The form collects a single bureau score. Training data has three
  (EXT_SOURCE_1/2/3, with many missing values). We aggregate them into
  EXT_SOURCE_MEAN (mean of available scores) at training time; at serving the
  single form value *is* the mean. This replaces the earlier hack of copying
  one score into all three columns.
"""
from __future__ import annotations

import pandas as pd

from src.config import DEFAULT_RISK_THRESHOLD

# Continuous pass-through features collected from the applicant.
PASS_THROUGH_FEATURES = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "DAYS_BIRTH",
]

# Derived financial ratios (identical formulas at train and serve time).
RATIO_FEATURES = [
    "CREDIT_INCOME_PERCENT",
    "ANNUITY_INCOME_PERCENT",
    "CREDIT_TERM",
]

# Bureau score aggregate (see module docstring).
BUREAU_FEATURE = "EXT_SOURCE_MEAN"

MODEL_FEATURES = PASS_THROUGH_FEATURES + RATIO_FEATURES + [BUREAU_FEATURE]

# All model features are standardized; the scaler is fit on exactly this list
# so the fitted column names always match serving.
SCALED_FEATURES = list(MODEL_FEATURES)


def build_features(raw: dict) -> pd.DataFrame:
    """Turn a raw application dict into a one-row (unscaled) feature DataFrame.

    Derives the financial ratios used at training time. Column order is
    deterministic and identical to the training feature matrix.
    """
    income = float(raw["AMT_INCOME_TOTAL"])
    credit = float(raw["AMT_CREDIT"])
    annuity = float(raw["AMT_ANNUITY"])
    bureau_score = float(raw["EXT_SOURCE_3"])

    features = {
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        # Age in days (negative), as in the Home Credit dataset.
        "DAYS_BIRTH": int(raw["DAYS_BIRTH"]),
        "CREDIT_INCOME_PERCENT": credit / income,
        "ANNUITY_INCOME_PERCENT": annuity / income,
        "CREDIT_TERM": annuity / credit,
        # Single form score == bureau aggregate (see module docstring).
        BUREAU_FEATURE: bureau_score,
    }
    return pd.DataFrame([features], columns=MODEL_FEATURES)


def scale_features(df: pd.DataFrame, scaler) -> pd.DataFrame:
    """Standardize the columns the fitted scaler expects.

    Uses the column names recorded on the scaler at fit time so serving stays
    consistent with training even if the feature list changes.
    """
    if hasattr(scaler, "feature_names_in_"):
        cols = [c for c in scaler.feature_names_in_ if c in df.columns]
    else:  # pragma: no cover - only hit by very old sklearn artifacts
        cols = [c for c in SCALED_FEATURES if c in df.columns]
    scaled = df.copy()
    scaled[cols] = scaler.transform(df[cols])
    return scaled


def decide_risk(default_probability: float, threshold: float = DEFAULT_RISK_THRESHOLD) -> str:
    """Map a default probability to a business risk label."""
    return "High" if default_probability > threshold else "Low"
