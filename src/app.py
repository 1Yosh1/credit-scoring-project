import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 1. Initialize App
app = FastAPI(title="Credit Risk Scoring API")

# 2. Load Model & Columns
try:
    model = joblib.load("models/credit_model.pkl")
    model_columns = joblib.load("models/model_columns.pkl")
    print("✅ Model and columns loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# 3. Define Input Schema
class LoanApplication(BaseModel):
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    DAYS_BIRTH: int       # Age in days (negative)
    CODE_GENDER: str      # 'M' or 'F'
    EXT_SOURCE_3: float   # Normalized Credit Score (0.0 to 1.0)

# 4. Prediction Endpoint
@app.post("/predict")
def predict_risk(application: LoanApplication):
    try:
        # Convert input to DataFrame
        data = pd.DataFrame([application.dict()])
        
        # --- FEATURE ENGINEERING ---
        
        # A. Create Financial Ratios
        data['CREDIT_INCOME_PERCENT'] = data['AMT_CREDIT'] / data['AMT_INCOME_TOTAL']
        data['ANNUITY_INCOME_PERCENT'] = data['AMT_ANNUITY'] / data['AMT_INCOME_TOTAL']
        data['CREDIT_TERM'] = data['AMT_ANNUITY'] / data['AMT_CREDIT']
        
        # B. SMART IMPUTATION (The Fix)
        # The model needs 3 different credit scores. We only have 1 input.
        # We copy the input score to the other missing features so the model sees consistency.
        data['EXT_SOURCE_1'] = application.EXT_SOURCE_3
        data['EXT_SOURCE_2'] = application.EXT_SOURCE_3
        
        # Assume the applicant is employed (approx 5.5 years)
        # Without this, the model assumes DAYS_EMPLOYED is 0 (Unemployed) -> High Risk
        data['DAYS_EMPLOYED'] = -2000 
        
        # C. Handle Gender
        if application.CODE_GENDER == 'M':
            data['CODE_GENDER_M'] = 1
            data['CODE_GENDER_F'] = 0
        else:
            data['CODE_GENDER_M'] = 0
            data['CODE_GENDER_F'] = 1
            
        # D. Align Columns
        # This fills any remaining missing columns with 0
        data_final = data.reindex(columns=model_columns, fill_value=0)
        
        # --- PREDICTION ---
        prediction = model.predict_proba(data_final)
        default_prob = prediction[0][1]
        
        # --- BUSINESS LOGIC ---
        # We set the threshold to 20%. 
        # Prob < 0.20 = Green (Safe)
        # Prob > 0.20 = Red (Risky)
        risk_threshold = 0.20
        
        return {
            "default_probability": float(default_prob),
            "risk_level": "High" if default_prob > risk_threshold else "Low"
        }
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"message": "Credit Risk API is Live"}