"""Tests for the API request schema."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas import LoanApplication


def _valid() -> dict:
    return {
        "AMT_INCOME_TOTAL": 100_000.0,
        "AMT_CREDIT": 200_000.0,
        "AMT_ANNUITY": 10_000.0,
        "DAYS_BIRTH": -10_000,
        "EXT_SOURCE_3": 0.5,
        "CODE_GENDER": "M",
    }


def test_valid_payload_passes():
    app = LoanApplication(**_valid())
    assert app.CODE_GENDER == "M"


def test_negative_income_rejected():
    data = _valid()
    data["AMT_INCOME_TOTAL"] = -1
    with pytest.raises(ValidationError):
        LoanApplication(**data)


def test_ext_source_out_of_range_rejected():
    data = _valid()
    data["EXT_SOURCE_3"] = 1.5
    with pytest.raises(ValidationError):
        LoanApplication(**data)


def test_unknown_gender_rejected():
    data = _valid()
    data["CODE_GENDER"] = "X"
    with pytest.raises(ValidationError):
        LoanApplication(**data)


def test_days_birth_must_be_non_positive():
    data = _valid()
    data["DAYS_BIRTH"] = 5_000
    with pytest.raises(ValidationError):
        LoanApplication(**data)
