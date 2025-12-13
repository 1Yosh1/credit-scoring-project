import streamlit as st
import requests
import os

# Page Config
st.set_page_config(page_title="Credit Risk Scorer", page_icon="🏦", layout="centered")

# Header
st.title("🏦 AI Credit Scoring System")
st.markdown("---")
st.write("Enter the applicant's financial details below to assess creditworthiness.")

# Determine API URL (Docker vs Local)
# If running in Docker Compose, it uses 'http://backend:80/predict'
# If running locally, it defaults to 'http://127.0.0.1:8000/predict'
api_url = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

# Input Form
with st.form("risk_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Financials")
        income = st.number_input("Total Income ($)", value=50000, step=1000)
        credit_amount = st.number_input("Loan Amount ($)", value=200000, step=5000)
        annuity = st.number_input("Annuity Amount ($)", value=10000, step=500)
    
    with col2:
        st.subheader("👤 Demographics & Score")
        gender = st.selectbox("Gender", ["M", "F"])
        age = st.number_input("Age (Years)", value=30, min_value=18, max_value=70)
        
        # NEW: Number Input for Credit Score
        st.markdown("**External Credit Score (0.0 - 1.0)**")
        st.caption("0.1 = Poor History | 0.9 = Excellent History")
        ext_score = st.number_input(
            "Enter Score:", 
            min_value=0.01, 
            max_value=1.0, 
            value=0.5, 
            step=0.01,
            format="%.2f"
        )

    # Submit Button
    submitted = st.form_submit_button("Predict Risk Profile")

# Handling the Prediction
if submitted:
    # Prepare the payload (JSON) to send to the API
    payload = {
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit_amount,
        "AMT_ANNUITY": annuity,
        "DAYS_BIRTH": -1 * age * 365, # Convert Age to Days (Negative)
        "CODE_GENDER": gender,
        "EXT_SOURCE_3": ext_score
    }
    
    try:
        with st.spinner("Analyzing Risk Model..."):
            response = requests.post(api_url, json=payload)
            
        if response.status_code == 200:
            result = response.json()
            prob = result['default_probability']
            risk = result['risk_level']
            
            st.markdown("---")
            st.subheader("Analysis Result")
            
            # Dynamic UI Display
            if risk == "High":
                st.error(f"⚠️ **HIGH RISK DETECTED**")
                st.metric(label="Probability of Default", value=f"{prob:.1%}", delta="Above Threshold", delta_color="inverse")
                st.write("**Recommendation:** Reject Loan or Require Guarantor.")
            else:
                st.success(f"✅ **LOW RISK APPROVED**")
                st.metric(label="Probability of Default", value=f"{prob:.1%}", delta="Safe Range")
                st.write("**Recommendation:** Approve Loan.")
                
        else:
            st.error(f"Server Error: {response.text}")
            
    except Exception as e:
        st.error(f"Connection Error: {e}")
        st.info(f"Trying to connect to: {api_url}")