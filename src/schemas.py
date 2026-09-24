"""Pydantic schemas describing the API contract."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class LoanApplication(BaseModel):
    """A loan application as submitted by the dashboard or an API client."""

    AMT_INCOME_TOTAL: float = Field(gt=0, description="Total income, USD")
    AMT_CREDIT: float = Field(gt=0, description="Requested loan amount, USD")
    AMT_ANNUITY: float = Field(gt=0, description="Annual annuity payment, USD")
    DAYS_BIRTH: int = Field(le=0, description="Age in days (negative, per Home Credit format)")
    EXT_SOURCE_3: float = Field(ge=0.0, le=1.0, description="Normalized external credit score")
    CODE_GENDER: str = Field(description="'M' or 'F'")

    @field_validator("CODE_GENDER")
    @classmethod
    def gender_must_be_known(cls, v: str) -> str:
        v = v.upper()
        if v not in ("M", "F"):
            raise ValueError("CODE_GENDER must be 'M' or 'F'")
        return v


class PredictionResponse(BaseModel):
    """Scored application returned by /predict."""

    default_probability: float
    risk_level: str
    threshold: float
