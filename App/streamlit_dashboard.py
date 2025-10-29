
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---- Page Configuration ----
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection")
st.markdown("Enter transaction details to predict whether it's **fraudulent or legitimate**.")

# ---- Load Trained Model ----
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'logistic_model.pkl'))
model = joblib.load(model_path)

# ---- Create Input Columns Layout ----
st.subheader("Enter Feature Values:")

col1, col2, col3 = st.columns(3)

# Time and Amount
with col1:
    time = st.number_input("Time", min_value=0.0, value=10000.0)
with col2:
    amount_scaled = st.number_input("Scaled Amount", min_value=0.0, value=100.0)

# V1 to V28 Inputs
v_features = []
cols = [st.columns(3) for _ in range(10)]  # Create 10 rows of 3 columns each

for i in range(1, 29):
    col_group = cols[(i - 1) // 3]
    with col_group[(i - 1) % 3]:
        v = st.number_input(f"V{i}", value=0.0)
        v_features.append(v)

# ---- Prepare Input for Prediction ----
input_data = np.array([[time] + v_features + [amount_scaled]])
input_df = pd.DataFrame(input_data, columns=['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount_scaled'])

# ---- Prediction Button ----
if st.button("🔍 Predict Transaction"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.error("🚨 The transaction is **FRAUDULENT**!")
    else:
        st.success("✅ The transaction is **LEGITIMATE**.")
