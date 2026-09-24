"""Loan officer dashboard for the credit scoring API."""
from __future__ import annotations

import os

import requests
import streamlit as st

st.set_page_config(page_title="Credit Risk Scorer", page_icon="🏦", layout="centered")

API_BASE = os.getenv("API_URL_BASE", "http://127.0.0.1:8000")
PREDICT_URL = os.getenv("API_URL", f"{API_BASE}/predict")
HEALTH_URL = f"{API_BASE}/health"

st.title("🏦 AI Credit Scoring System")
st.caption("Enter an applicant's financial details to assess creditworthiness.")

# --- Sidebar: live API status -------------------------------------------
@st.cache_data(ttl=15)
def check_api(url: str) -> bool:
    try:
        return requests.get(f"{url}/health", timeout=2).ok
    except requests.RequestException:
        return False


with st.sidebar:
    st.subheader("Service Status")
    api_online = check_api(API_BASE)
    if api_online:
        st.success("API online")
    else:
        st.error("API unreachable")
        st.caption(f"Expected at {API_BASE}")

# --- Input form ----------------------------------------------------------
with st.form("risk_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💰 Financials")
        income = st.number_input("Total Income ($)", value=50_000, step=1_000, min_value=1)
        credit_amount = st.number_input("Loan Amount ($)", value=200_000, step=5_000, min_value=1)
        annuity = st.number_input("Annuity Amount ($)", value=10_000, step=500, min_value=1)

    with col2:
        st.subheader("👤 Demographics & Score")
        gender = st.selectbox("Gender", ["M", "F"])
        age = st.number_input("Age (Years)", value=30, min_value=18, max_value=100)
        ext_score = st.number_input(
            "External Credit Score (0.0 - 1.0)",
            min_value=0.01,
            max_value=1.0,
            value=0.5,
            step=0.01,
            format="%.2f",
            help="0.1 = poor history · 0.9 = excellent history",
        )

    submitted = st.form_submit_button("Predict Risk Profile", type="primary")

# --- Prediction ----------------------------------------------------------
if submitted:
    payload = {
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit_amount,
        "AMT_ANNUITY": annuity,
        "DAYS_BIRTH": -int(age) * 365,
        "CODE_GENDER": gender,
        "EXT_SOURCE_3": ext_score,
    }

    with st.spinner("Scoring application ..."):
        try:
            response = requests.post(PREDICT_URL, json=payload, timeout=10)
        except requests.RequestException as exc:
            st.error("Could not reach the scoring API.")
            st.caption(f"Tried {PREDICT_URL} — {exc}")
            st.stop()

    if response.ok:
        result = response.json()
        prob = result["default_probability"]
        risk = result["risk_level"]
        threshold = result.get("threshold", 0.20)

        st.markdown("---")
        st.subheader("Analysis Result")
        if risk == "High":
            st.error("⚠️ **HIGH RISK DETECTED**")
            st.metric(label="Probability of Default", value=f"{prob:.1%}",
                      delta="Above threshold", delta_color="inverse")
            st.write("**Recommendation:** Reject loan or require a guarantor.")
        else:
            st.success("✅ **LOW RISK — APPROVED**")
            st.metric(label="Probability of Default", value=f"{prob:.1%}",
                      delta="Below threshold")
            st.write("**Recommendation:** Approve loan.")

        st.caption(f"Decision threshold: P(default) > {threshold:.0%} ⇒ High risk")

        explanation = result.get("explanation")
        if explanation:
            st.markdown("##### Top risk factors")
            for factor in explanation["top_factors"]:
                arrow = "↑" if factor["direction"] == "increases_risk" else "↓"
                value_part = f" — `{factor['value']}`" if factor.get("value") else ""
                st.markdown(
                    f"{arrow} **{factor['label']}**{value_part}: "
                    f"**{'+' if factor['impact_pp'] >= 0 else ''}{factor['impact_pp']:.1f} pp** "
                    "default risk"
                )
            st.caption(
                f"Population baseline P(default) ≈ {explanation['base_probability']:.0%}. "
                "Factors are SHAP contributions (percentage points) and sum to the gap "
                "between the baseline and this prediction."
            )
    elif response.status_code == 422:
        detail = response.json().get("detail", "Invalid input")
        st.warning("The application was rejected by input validation.")
        st.json(detail if isinstance(detail, list) else {"detail": detail})
    else:
        st.error(f"API error {response.status_code}: {response.text[:300]}")
