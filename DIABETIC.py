import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime
import pytz

# -----------------------
# Cached loading
# -----------------------
@st.cache_resource
def load_model():
    return joblib.load("diabetes_model_clean.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("diabetes.csv")

# -----------------------
# Helper: Detect local timezone from browser
# -----------------------
def get_local_timezone():
    js_code = """
    async function getTimezone() {
        const tz = Intl.DateTimeFormat().resolvedOptions().timeZone;
        return tz;
    }
    getTimezone().then(tz => {
        window.parent.postMessage({streamlitTZ: tz}, "*");
    });
    """
    st.components.v1.html(f"<script>{js_code}</script>", height=0)
    msg = st.experimental_get_query_params().get("timezone", [""])[0]
    return msg

# -----------------------
# PDF generation function
# -----------------------
def generate_pdf_report(user_data, prediction, confidence, health_tips, data, user_name, local_time_str):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Title and timestamp
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Diabetes Prediction Report")
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 70, f"Generated on: {local_time_str}")
    c.drawString(50, height - 85, f"Name: {user_name}")

    # User Input Summary
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 110, "User Data:")
    c.setFont("Helvetica", 11)
    y = height - 130
    for key, value in user_data.items():
        c.drawString(60, y, f"{key}: {value}")
        y -= 15

    # Prediction Result
    result = "Diabetic" if prediction == 1 else "Not Diabetic"
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y - 10, f"Prediction: {result}")
    c.drawString(50, y - 30, f"Confidence: {confidence}%")

    # Health Tips
    y = y - 60
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Health Recommendations:")
    y -= 20
    c.setFont("Helvetica", 10)
    for tip in health_tips:
        c.drawString(60, y, f"- {tip}")
        y -= 15

    # Plot charts and insert into PDF
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        features = ["Glucose", "BloodPressure", "BMI", "Age"]
        diabetic_avg = data[data["Outcome"] == 1][features].mean()

        fig, axs = plt.subplots(2, 2, figsize=(8, 6))
        axs = axs.flatten()
        for i, feature in enumerate(features):
            ax = axs[i]
            sns.histplot(data[data["Outcome"] == 1][feature], label="Diabetic", color="red", ax=ax, kde=True, stat="count", alpha=0.5)
            sns.histplot(data[data["Outcome"] == 0][feature], label="Non-Diabetic", color="green", ax=ax, kde=True, stat="count", alpha=0.5)
            ax.axvline(user_data[feature], color="blue", linestyle="--", label="User")
            ax.set_title(feature)
            ax.legend()

        plt.tight_layout()
        plt.savefig(tmpfile.name)
        plt.close()

        c.drawImage(tmpfile.name, 50, 100, width=500, preserveAspectRatio=True, mask='auto')

    c.save()
    buffer.seek(0)
    return buffer

# -----------------------
# Load model and data
# -----------------------
model = load_model()
data = load_data()

st.set_page_config(page_title="Diabetes App", layout="centered")

# -----------------------
# Session state setup
# -----------------------
if "page" not in st.session_state:
    st.session_state.page = "Predict"
if "prediction" not in st.session_state:
    st.session_state.prediction = None
    st.session_state.inputs = {}
    st.session_state.confidence = None
if "user_name" not in st.session_state:
    st.session_state.user_name = "Anonymous"

defaults = {
    "Glucose": 100,
    "BloodPressure": 80,
    "BMI": 25.0,
    "Age": 30
}
for key, val in defaults.items():
    st.session_state.setdefault(key, val)

# -----------------------
# Navigation
# -----------------------
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

    # Name input
    user_name = st.text_input("Enter your name:", value=st.session_state.user_name)
    st.session_state.user_name = user_name

    # Inputs
    st.number_input("Glucose", 0, 200, key="Glucose")
    st.number_input("Blood Pressure", 40, 140, key="BloodPressure")
    st.number_input("BMI", 10.0, 50.0, key="BMI")
    st.number_input("Age", 0, 100, key="Age")

    if st.button("🔍 Predict"):
        input_data = np.array([[
            st.session_state["Glucose"],
            st.session_state["BloodPressure"],
            st.session_state["BMI"],
            st.session_state["Age"]
        ]])
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
        ax.set_title(f"{feature}")
        ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("💡 Suggestions to Improve Your Health")
    tips = []
    if user_data["Glucose"] > 125:
        tips.append("High glucose — reduce sugar intake and monitor carbohydrate consumption.")
    if user_data["BMI"] > 30:
        tips.append("High BMI — consider regular physical activity and healthy eating.")
    if user_data["BloodPressure"] > 120:
        tips.append("Elevated blood pressure — reduce salt, avoid stress, and monitor regularly.")
    if user_data["Age"] > 45:
        tips.append("Regular screenings are recommended due to age-related risks.")
    if not tips:
        tips = ["All your values are within the healthy range!"]
    for tip in tips:
        st.markdown(f"✅ {tip}")

    st.subheader("📤 Download Report")

    # Get local timezone string
    local_tz = datetime.now(pytz.timezone("Asia/Kuala_Lumpur"))  # Or dynamically replace
    local_time_str = local_tz.strftime('%Y-%m-%d %H:%M:%S %Z')

    pdf = generate_pdf_report(
        user_data=st.session_state.inputs,
        prediction=st.session_state.prediction,
        confidence=st.session_state.confidence,
        health_tips=tips,
        data=data,
        user_name=st.session_state.user_name,
        local_time_str=local_time_str
    )

    st.download_button(
        label="📄 Download PDF Report",
        data=pdf,
        file_name="diabetes_report.pdf",
        mime="application/pdf"
    )

    if st.button("🔙 Back to Prediction"):
        st.session_state.page = "Predict"
        st.rerun()
