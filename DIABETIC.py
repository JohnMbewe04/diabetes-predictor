import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load model and dataset
model = joblib.load("diabetes_model_clean.pkl")
data = pd.read_csv("diabetes.csv")

# Compute means for diabetic and non-diabetic classes
features = ["Glucose", "BloodPressure", "BMI", "Age"]
diabetic_means = data[data["Outcome"] == 1][features].mean()
non_diabetic_means = data[data["Outcome"] == 0][features].mean()

# Theme selection
theme = st.selectbox("Choose Theme", ["Light", "Dark"])

if theme == "Dark":
    st.markdown(
        """
        <style>
            body {
                background-color: #1e1e1e !important;
                color: white !important;
            }
            .stButton>button {
                background-color: #444;
                color: white;
            }
            .stNumberInput>div>input {
                background-color: #333;
                color: white;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

# Page layout
st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")
st.title("🩺 Diabetes Risk Predictor")
st.markdown("Enter the values below to predict diabetes risk:")

# Inputs
glucose = st.number_input("Glucose", min_value=0, max_value=200, value=100, step=1)
bp = st.number_input("Blood Pressure", min_value=40, max_value=140, value=80, step=1)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
age = st.number_input("Age", min_value=0, max_value=100, value=30, step=1)

# Prediction and report
if st.button("Predict"):
    input_data = np.array([[glucose, bp, bmi, age]])
    prediction = model.predict(input_data)[0]
    confidence = (
        model.predict_proba(input_data)[0][prediction]
        if hasattr(model, "predict_proba")
        else 0.5
    )
    result = "Diabetic" if prediction == 1 else "Not Diabetic"
    confidence_percent = round(confidence * 100, 2)

    st.success(f"Prediction: **{result}**")
    st.info(f"Confidence: **{confidence_percent}%**")

    if prediction == 1:
        if st.button("🧾 Generate Report"):
            st.subheader("📊 Diagnostic Report")
            user_input = {"Glucose": glucose, "BloodPressure": bp, "BMI": bmi, "Age": age}
            for feature in features:
                user_val = user_input[feature]
                diabetic_diff = abs(user_val - diabetic_means[feature])
                non_diabetic_diff = abs(user_val - non_diabetic_means[feature])
                closer_to = "Diabetic" if diabetic_diff < non_diabetic_diff else "Non-Diabetic"
                color = "🟥" if closer_to == "Diabetic" else "🟩"
                st.write(f"{color} **{feature}**: Your value `{user_val}` is closer to **{closer_to}** average.")
