import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and dataset
model = joblib.load("diabetes_model_clean.pkl")
data = pd.read_csv("diabetes.csv")

st.set_page_config(page_title="Diabetes App", layout="centered")

# Initialize session state (page, prediction results)
if "page" not in st.session_state:
    st.session_state.page = "Predict"
if "prediction" not in st.session_state:
    st.session_state.prediction = None
    st.session_state.inputs = {}
    st.session_state.confidence = None

# Initialize input fields only if not already set
defaults = {
    "Glucose": 100,
    "BloodPressure": 80,
    "BMI": 25.0,
    "Age": 30
}
for key, val in defaults.items():
    st.session_state.setdefault(key, val)

# Sidebar navigation
selected_page = st.sidebar.radio("Navigation", ["Predict", "Report"], index=["Predict", "Report"].index(st.session_state.page))
if selected_page != st.session_state.page:
    st.session_state.page = selected_page
    st.rerun()

# ---------------------------
# Page 1: Prediction
# ---------------------------
if st.session_state.page == "Predict":
    st.title("🩺 Diabetes Risk Predictor")
    st.markdown("Enter your health data below:")

    st.number_input("Glucose", 0, 200, key="Glucose")
    st.number_input("Blood Pressure", 40, 140, key="BloodPressure")
    st.number_input("BMI", 10.0, 50.0, key="BMI")
    st.number_input("Age", 0, 100, key="Age")


    if st.button("🔍 Predict"):
        prediction = model.predict(input_data)[0]
        confidence = model.predict_proba(input_data)[0][prediction]

        st.session_state.prediction = prediction
        st.session_state.confidence = round(confidence * 100, 2)
        st.session_state.inputs = {
            "Glucose": st.session_state["Glucose"],
            "BloodPressure": st.session_state["BloodPressure"],
            "BMI": st.session_state["BMI"],
            "Age": st.session_state["Age"]
        }

        result = "Diabetic" if prediction == 1 else "Not Diabetic"
        st.success(f"Prediction: {result}")
        st.info(f"Confidence: {st.session_state.confidence}%")

    if st.session_state.prediction is not None:
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🧾 View Report"):
                st.session_state.page = "Report"
                st.rerun()
        with col2:
            if st.session_state.prediction == 1:
                st.markdown(
                    "[📍 Find Nearby Clinics](https://www.google.com/maps/search/diabetes+clinic+near+me)",
                    unsafe_allow_html=True
                )

# ---------------------------
# Page 2: Report
# ---------------------------
elif st.session_state.page == "Report":
    st.title("🧾 Diabetes Report")
    user_data = st.session_state.inputs
    features = ["Glucose", "BloodPressure", "BMI", "Age"]

    colorblind_mode = st.checkbox("♿ Enable colorblind-friendly palette", value=False)
    if colorblind_mode:
        st.caption("🎨 Using colorblind-safe colors for the graphs.")
        diabetic_color = "#E69F00"
        non_diabetic_color = "#56B4E9"
        user_color = "#009E73"
    else:
        diabetic_color = "red"
        non_diabetic_color = "green"
        user_color = "blue"

    st.subheader("📌 Feature Comparison to Diabetic Averages")
    diabetic_avg = data[data["Outcome"] == 1][features].mean()
    for feature in features:
        user_val = user_data[feature]
        avg_val = diabetic_avg[feature]
        delta = user_val - avg_val
        color = "red" if delta > 0 else "green"
        st.markdown(
            f"**{feature}**: {user_val} _(Avg: {round(avg_val,1)})_ → "
            f"<span style='color:{color}'>{'High' if delta > 0 else 'Low'}</span>",
            unsafe_allow_html=True
        )

    st.subheader("📈 Distribution Comparison")
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()
    for i, feature in enumerate(features):
        ax = axs[i]
        sns.histplot(data[data["Outcome"] == 1][feature], label="Diabetic",
                     color=diabetic_color, ax=ax, kde=True, stat="count", alpha=0.5)
        sns.histplot(data[data["Outcome"] == 0][feature], label="Non-Diabetic",
                     color=non_diabetic_color, ax=ax, kde=True, stat="count", alpha=0.5)
        ax.axvline(user_data[feature], color=user_color, linestyle="--", label="Your Value")
        ax.set_title(f"{feature}", fontsize=14)
        ax.set_xlabel(feature, fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.legend(fontsize=10, loc='upper right')
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("💡 Suggestions to Improve Your Health")
    tips = []
    if user_data["Glucose"] > 125:
        tips.append("⚠️ High glucose — reduce sugar intake and monitor carbohydrate consumption.")
    if user_data["BMI"] > 30:
        tips.append("⚠️ High BMI — consider regular physical activity and healthy eating.")
    if user_data["BloodPressure"] > 120:
        tips.append("⚠️ Elevated blood pressure — reduce salt, avoid stress, and monitor regularly.")
    if user_data["Age"] > 45:
        tips.append("✅ Regular screenings are recommended due to age-related risks.")
    if tips:
        for tip in tips:
            st.markdown(tip)
    else:
        st.success("👍 All your values are within the healthy range!")

    if st.button("🔙 Back to Prediction"):
        st.session_state.page = "Predict"
        st.rerun()
