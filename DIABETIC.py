# Step-by-step code to implement requested features in Streamlit

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import webbrowser

# Load model and dataset
model = joblib.load("diabetes_model_clean.pkl")
data = pd.read_csv("diabetes.csv")

st.set_page_config(page_title="Diabetes App", layout="wide")

# Session state initialization
if "page" not in st.session_state:
    st.session_state.page = "Predict"
if "prediction" not in st.session_state:
    st.session_state.prediction = None
    st.session_state.inputs = {}
    st.session_state.confidence = None

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Predict", "Report", "What-if Analysis"])
st.session_state.page = page

# Predict Page
if st.session_state.page == "Predict":
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
            if st.button("📍 Find Nearby Clinics"):
                st.write("Redirecting you to Google Maps for nearby clinics...")
                webbrowser.open("https://www.google.com/maps/search/diabetes+clinic+near+me")

# Report Page
elif st.session_state.page == "Report" and st.session_state.prediction is not None:
    st.title("🧾 Diabetes Report")
    user_data = st.session_state.inputs
    features = ["Glucose", "BloodPressure", "BMI", "Age"]

    # Comparison
    st.subheader("📌 Comparison to Diabetic Averages")
    diabetic_avg = data[data["Outcome"] == 1][features].mean()
    for feature in features:
        user_val = user_data[feature]
        avg_val = diabetic_avg[feature]
        delta = user_val - avg_val
        st.markdown(f"**{feature}**: {user_val} _(Avg: {round(avg_val,1)})_ → {'⬆️ High' if delta > 0 else '⬇️ Low'}")

    # Interactive Chart
    st.subheader("📈 Distribution Comparison")
    selected = st.selectbox("Choose a feature to visualize", features)
    fig = px.histogram(data, x=selected, color=data["Outcome"].map({0: "Non-Diabetic", 1: "Diabetic"}),
                       nbins=30, barmode='overlay', color_discrete_map={"Diabetic": "red", "Non-Diabetic": "green"})
    fig.add_vline(x=user_data[selected], line_dash="dash", line_color="blue")
    st.plotly_chart(fig)

    # Generated Summary (simulated langchain logic)
    st.subheader("📝 Personalized Summary")
    msg = f"You are predicted as **Diabetic** with a confidence of **{st.session_state.confidence}%**."
    msg += f"\n\nYour glucose level is {user_data['Glucose']}, BMI is {user_data['BMI']}, and blood pressure is {user_data['BloodPressure']}."
    if user_data['Glucose'] > 125:
        msg += "\n\n⚠️ Consider reducing your sugar intake and increasing physical activity."
    st.markdown(msg)

# What-if Analysis Page
elif st.session_state.page == "What-if Analysis" and st.session_state.prediction is not None:
    st.title("🔮 What-if Lifestyle Simulation")

    st.markdown("Adjust the sliders to simulate lifestyle changes and observe their effect:")
    glucose = st.slider("Simulated Glucose", 70, 200, int(st.session_state.inputs["Glucose"]))
    bmi = st.slider("Simulated BMI", 15, 50, int(st.session_state.inputs["BMI"]))
    bp = st.slider("Simulated Blood Pressure", 60, 140, int(st.session_state.inputs["BloodPressure"]))
    age = st.session_state.inputs["Age"]

    new_input = np.array([[glucose, bp, bmi, age]])
    sim_pred = model.predict(new_input)[0]
    sim_conf = model.predict_proba(new_input)[0][sim_pred]

    new_result = "Diabetic" if sim_pred == 1 else "Not Diabetic"
    st.write(f"**New Prediction:** {new_result}")
    st.write(f"**Confidence:** {round(sim_conf * 100, 2)}%")
