"""Per-prediction explanations via SHAP.

We use SHAP's TreeExplainer in its native *raw* (log-odds) space and convert
the additive contributions into probability-space percentage points with an
exact telescoping walk: features are transformed one at a time (largest
|phi| first, so the strongest drivers claim the steepest part of the sigmoid)
and each step's probability delta is that feature's contribution.

    p_i = sigmoid(base + phi_1 + ... + phi_i)
    contribution_i = p_i - p_{i-1}

The contributions sum exactly to ``P(default) - P(base)`` (the distance from
the population baseline), and the sign says whether the factor pushed risk up
or down. Because the walk is order-dependent for individual terms, we sort by
impact magnitude so the headline factors are stable and human-readable.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import shap

# Form field -> readable label (used in the API response and dashboard).
FEATURE_LABELS = {
    "EXT_SOURCE_MEAN": "External credit score",
    "CREDIT_INCOME_PERCENT": "Loan amount vs income",
    "ANNUITY_INCOME_PERCENT": "Yearly payment vs income",
    "CREDIT_TERM": "Repayment length",
    "AMT_CREDIT": "Loan amount",
    "AMT_ANNUITY": "Yearly payment",
    "AMT_INCOME_TOTAL": "Total income",
    "DAYS_BIRTH": "Age",
}

# Number of factors to surface in the response.
TOP_K = 3


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def format_feature_value(feature: str, value: float) -> str:
    """Render a raw feature value the way a loan officer would read it."""
    if feature == "DAYS_BIRTH":
        return f"{round(-value / 365)} yrs"
    if feature.startswith("AMT_"):
        return f"${value:,.0f}"
    if feature == "EXT_SOURCE_MEAN":
        return f"{value:.2f}"
    return f"{value:.2f}"


def probability_contributions(
    phi_raw: np.ndarray, base_log_odds: float
) -> tuple[float, float, np.ndarray]:
    """Convert raw SHAP values to exact probability-space contributions.

    Returns ``(base_probability, final_probability, contributions)`` where
    ``contributions.sum() == final - base`` by construction.
    """
    order = np.argsort(-np.abs(phi_raw))  # largest impact first
    p_prev = float(_sigmoid(base_log_odds))
    base_probability = p_prev
    f = base_log_odds
    contributions = np.zeros_like(phi_raw, dtype=float)
    for idx in order:
        f += float(phi_raw[idx])
        p_cur = float(_sigmoid(f))
        contributions[idx] = p_cur - p_prev
        p_prev = p_cur
    return base_probability, p_prev, contributions


@lru_cache(maxsize=1)
def get_tree_explainer(model) -> shap.TreeExplainer:
    """Cache one TreeExplainer per model object (background is tree-path-dependent)."""
    return shap.TreeExplainer(model)


def _extract_shap(explainer: shap.TreeExplainer, features: pd.DataFrame):
    """Return (phi_row, base_log_odds) across SHAP output conventions.

    XGBoost binary classifiers yield an (n, features) array with a scalar
    baseline, while sklearn tree models yield a per-class list (or an
    (n, features, classes) array). We always take the positive-class term.
    """
    raw = explainer.shap_values(features)
    expected = np.asarray(explainer.expected_value).ravel()
    if isinstance(raw, list):  # per-class list (sklearn trees)
        phi = np.asarray(raw[-1])[0]
        base = float(expected[-1])
    else:
        arr = np.asarray(raw)
        if arr.ndim == 3:  # (n, features, classes)
            phi = arr[0][:, -1]
            base = float(expected[-1])
        else:  # (n, features)
            phi = arr[0]
            base = float(expected[0])
    return phi, base


def explain_prediction(
    model,
    scaled_features: pd.DataFrame,
    raw_values: dict | None = None,
    top_k: int = TOP_K,
) -> dict:
    """Explain one scored application in human-readable terms.

    Returns a dict with ``base_probability`` (population baseline),
    ``probability_additive`` (base + contributions, ties to the model output),
    and ``top_factors``: the strongest drivers with label, the applicant's
    value, direction, and impact in percentage points of default probability.
    """
    explainer = get_tree_explainer(model)
    phi, base_log_odds = _extract_shap(explainer, scaled_features)

    base_probability, _, contributions = probability_contributions(phi, base_log_odds)

    impact = pd.Series(contributions, index=scaled_features.columns)
    top = impact.reindex(impact.abs().sort_values(ascending=False).index)[:top_k]

    factors = []
    for name, value in top.items():
        factor = {
            "feature": str(name),
            "label": FEATURE_LABELS.get(str(name), str(name)),
            "direction": "increases_risk" if value > 0 else "decreases_risk",
            "impact_pp": round(float(value) * 100, 2),
        }
        if raw_values is not None and str(name) in raw_values:
            factor["value"] = format_feature_value(str(name), float(raw_values[str(name)]))
        factors.append(factor)

    return {
        "base_probability": round(base_probability, 4),
        "probability_additive": round(
            base_probability + float(contributions.sum()), 4
        ),
        "top_factors": factors,
    }
