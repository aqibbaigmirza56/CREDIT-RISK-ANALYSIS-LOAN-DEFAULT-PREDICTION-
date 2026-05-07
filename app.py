# %%
# ===============================
# Loan Default Prediction App
# ===============================

import streamlit as st
import joblib
import numpy as np

# Load trained model (must already exist)
model = joblib.load("logistic_model.pkl")

# App title
st.title("Loan Default Prediction System")

# User inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Annual Income", min_value=0.0, value=500000.0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
loan_amount = st.number_input("Loan Amount", min_value=0.0, value=200000.0)

# Prediction button
if st.button("Predict"):
    data = np.array([[age, income, credit_score, loan_amount]])
    result = model.predict(data)

    if result[0] == 0:
        st.success("Low Risk ✅ Loan Approved")
    else:
        st.error("High Risk ❌ Loan Rejected")

# %%
# %%



