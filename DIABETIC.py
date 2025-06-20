import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load model and training data summary
model = joblib.load("diabetes_model_clean.pkl")
diabetes_df = pd.read_csv("diabetes.csv")

# Only using 4 features
feature_names = ["Glucose", "BloodPressure", "BMI", "Age"]

# Calculate feature means for diabetic and non-diabetic
diabetic_means = diabetes_df[diabetes_df["Outcome"] == 1][feature_names].mean()
non_diabetic_means = diabetes_df[diabetes_df["Outcome"] == 0][feature_names].mean()

# Theme Toggle
theme = st.selectbox("Choose Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("""
        <style>
            body { background-color: #1e1e1e; color: white; }
            .stButton > button { background-color: #555; color: white; }
        </style>
    """, unsafe_allow_html=True)

# Title
st.title("🩺 Diabetes Risk Predictor")

# Inputs
glucose = st.slider("Glucose", 0, 200, 100)
bp = st.slider("Blood Pressure", 40, 140, 80)
bmi = st.slider("BMI", 10.0, 50.0, 25.0)
age = st.slider("Age", 0, 100, 30)

if st.button("Predict"):
    input_data = np.array([[glucose, bp, bmi, age]])
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else 0.5

    result = "Diabetic" if prediction == 1 else "Not Diabetic"
    confidence = round(prob * 100, 2)

    st.success(f"Prediction: **{result}**")
    st.info(f"Confidence: **{confidence}%**")

    # Diabetic report
    if prediction == 1:
        st.subheader("📝 Why were you predicted as Diabetic?")
        reasons = []
        user_values = {"Glucose": glucose, "BloodPressure": bp, "BMI": bmi, "Age": age}
        for feature in feature_names:
            user_val = user_values[feature]
            diabetic_mean = diabetic_means[feature]
            non_diabetic_mean = non_diabetic_means[feature]

            if abs(user_val - diabetic_mean) < abs(user_val - non_diabetic_mean):
                reasons.append(f"- Your {feature} value of **{user_val}** is closer to diabetic averages.")

        if reasons:
            st.markdown("\n".join(reasons))
        else:
            st.markdown("No strong feature deviation observed.")
