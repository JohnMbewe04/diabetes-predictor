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

# Session state initialization
if "page" not in st.session_state:
    st.session_state.page = "predict"
if "prediction" not in st.session_state:
    st.session_state.prediction = None
    st.session_state.inputs = {}
    st.session_state.confidence = None

# ---------------------------
# Page 1: Prediction Page
# ---------------------------
if st.session_state.page == "predict":
    st.title("🩺 Diabetes Risk Predictor")
    st.markdown("Enter your health data below:")

    # Inputs
    glucose = st.number_input("Glucose", 0, 200, 100)
    bp = st.number_input("Blood Pressure", 40, 140, 80)
    bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
    age = st.number_input("Age", 0, 100, 30)

    if st.button("🔍 Predict"):
        input_data = np.array([[glucose, bp, bmi, age]])
        prediction = model.predict(input_data)[0]
        confidence = model.predict_proba(input_data)[0][prediction]

        # Save to session
        st.session_state.prediction = prediction
        st.session_state.confidence = round(confidence * 100, 2)
        st.session_state.inputs = {
            "Glucose": glucose,
            "BloodPressure": bp,
            "BMI": bmi,
            "Age": age
        }

        result = "Diabetic" if prediction == 1 else "Not Diabetic"
        st.success(f"Prediction: {result}")
        st.info(f"Confidence: {st.session_state.confidence}%")

        if prediction == 1:
            if st.button("🧾 View Report"):
                st.session_state.page = "report"
                st.experimental_rerun()

# ---------------------------
# Page 2: Report Page
# ---------------------------
elif st.session_state.page == "report":
    st.title("🧾 Diabetes Report")

    user_data = st.session_state.inputs
    features = ["Glucose", "BloodPressure", "BMI", "Age"]

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

    # 📉 Visualizations
    st.subheader("📈 Distribution Comparison")
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    axs = axs.flatten()

    for i, feature in enumerate(features):
        sns.histplot(data[data["Outcome"] == 1][feature], label="Diabetic", color="red", ax=axs[i], kde=True)
        sns.histplot(data[data["Outcome"] == 0][feature], label="Non-Diabetic", color="green", ax=axs[i], kde=True)
        axs[i].axvline(user_data[feature], color="blue", linestyle="--", label="Your Value")
        axs[i].set_title(feature)
        axs[i].legend()

    st.pyplot(fig)

    # 💡 Health Tips
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
        st.session_state.page = "predict"
        st.experimental_rerun()
