import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("Diabetesmodel.pkl")

st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")
st.title("🩺 Diabetes Risk Predictor")

st.markdown("Enter the values below to predict diabetes risk:")

# Input sliders
glucose = st.slider("Glucose", 0, 200, 100)
bp = st.slider("Blood Pressure", 40, 140, 80)
bmi = st.slider("BMI", 10.0, 50.0, 25.0)
age = st.slider("Age", 0, 100, 30)

if st.button("Predict"):
    input_data = np.array([[glucose, bp, bmi, age]])
    prediction = model.predict(input_data)[0]

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_data)[0][1]
    else:
        prob = 0.5

    result_text = "Diabetic" if prediction == 1 else "Not Diabetic"
    confidence = round(prob * 100, 2)

    st.success(f"Prediction: **{result_text}**")
    st.info(f"Confidence: **{confidence}%**")
