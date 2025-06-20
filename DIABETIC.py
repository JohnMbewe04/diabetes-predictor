import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load model
model = joblib.load("diabetes_model_clean.pkl")

st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")
st.title("🩺 Diabetes Risk Predictor")
st.markdown("Enter the values below to predict diabetes risk:")

# Theme Toggle
theme = st.selectbox("Choose Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("""
        <style>
            body { background-color: #1e1e1e; color: white; }
            .stButton > button { background-color: #555; color: white; }
        </style>
    """, unsafe_allow_html=True)


# Input fields with manual and step control
glucose = st.number_input("Glucose", min_value=0, max_value=200, value=100, step=1)
bp = st.number_input("Blood Pressure", min_value=40, max_value=140, value=80, step=1)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
age = st.number_input("Age", min_value=0, max_value=100, value=30, step=1)

if st.button("Predict"):
    input_data = np.array([[glucose, bp, bmi, age]])
    prediction = model.predict(input_data)[0]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_data)[0]
        confidence = proba[prediction]
    else:
        confidence = 0.5
    result_text = "Diabetic" if prediction == 1 else "Not Diabetic"
    confidence_percent = round(confidence * 100, 2)
    st.success(f"Prediction: **{result_text}**")
    st.info(f"Confidence: **{confidence_percent}%**")
