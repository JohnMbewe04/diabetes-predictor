import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load model and dataset
model = joblib.load("diabetes_model_clean.pkl")
data = pd.read_csv("diabetes.csv")

# Set page config
st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")

# Session state to persist prediction result
if "prediction" not in st.session_state:
    st.session_state.prediction = None
    st.session_state.confidence = None
    st.session_state.inputs = {}

# Theme selection (only affects widget colors manually)
theme = st.radio("Choose Theme", ["Light", "Dark"], horizontal=True)
is_dark = theme == "Dark"

def themed_style(text, color_light, color_dark):
    return f"<span style='color: {color_dark if is_dark else color_light}'>{text}</span>"

# Input values
glucose = st.number_input("Glucose", 0, 200, 100)
bp = st.number_input("Blood Pressure", 40, 140, 80)
bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
age = st.number_input("Age", 0, 100, 30)

if st.button("🔍 Predict"):
    input_data = np.array([[glucose, bp, bmi, age]])
    prediction = model.predict(input_data)[0]
    confidence = model.predict_proba(input_data)[0][prediction] if hasattr(model, "predict_proba") else 0.5

    # Store in session state
    st.session_state.prediction = prediction
    st.session_state.confidence = round(confidence * 100, 2)
    st.session_state.inputs = {
        "Glucose": glucose,
        "BloodPressure": bp,
        "BMI": bmi,
        "Age": age
    }

# Show result if available
if st.session_state.prediction is not None:
    result_text = "Diabetic" if st.session_state.prediction == 1 else "Not Diabetic"
    st.success(f"Prediction: {result_text}")
    st.info(f"Confidence: {st.session_state.confidence}%")

    # Show report only if diabetic
    if st.session_state.prediction == 1:
        if st.button("🧾 Generate Report"):
            st.subheader("📊 Diagnostic Report")

            # Compare user values to diabetic and non-diabetic means
            features = ["Glucose", "BloodPressure", "BMI", "Age"]
            diabetic_mean = data[data["Outcome"] == 1][features].mean()
            non_diabetic_mean = data[data["Outcome"] == 0][features].mean()

            for feature in features:
                user_val = st.session_state.inputs[feature]
                dist_to_diabetic = abs(user_val - diabetic_mean[feature])
                dist_to_non_diabetic = abs(user_val - non_diabetic_mean[feature])
                closer = "Diabetic" if dist_to_diabetic < dist_to_non_diabetic else "Non-Diabetic"
                emoji = "🔴" if closer == "Diabetic" else "🟢"
                st.markdown(f"{emoji} **{feature}** = {user_val} → closer to *{closer}* profile.")

