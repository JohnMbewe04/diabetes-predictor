# Diabetes Prediction App with Multilingual Support and PDF Report

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
from deep_translator import GoogleTranslator
from babel.dates import format_datetime

# -----------------------
# Translation helper
# -----------------------
def t(text, lang="en"):
    if lang == "en":
        return text
    try:
        return GoogleTranslator(source='auto', target=lang).translate(text)
    except:
        return text

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
# PDF generation function
# -----------------------
def generate_pdf_report(user_data, prediction, confidence, health_tips, data, user_name, local_time_str, lang_code):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, t("Diabetes Prediction Report", lang_code))
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 70, f"{t('Generated on', lang_code)}: {local_time_str}")
    c.drawString(50, height - 85, f"{t('Name', lang_code)}: {user_name}")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 110, t("User Data", lang_code) + ":")
    c.setFont("Helvetica", 11)
    y = height - 130
    for key, value in user_data.items():
        c.drawString(60, y, f"{t(key, lang_code)}: {value}")
        y -= 15

    result = t("Diabetic", lang_code) if prediction == 1 else t("Not Diabetic", lang_code)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y - 10, f"{t('Prediction', lang_code)}: {result}")
    c.drawString(50, y - 30, f"{t('Confidence', lang_code)}: {confidence}%")

    y -= 60
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, t("Health Recommendations", lang_code) + ":")
    y -= 20
    c.setFont("Helvetica", 10)
    for tip in health_tips:
        c.drawString(60, y, f"- {t(tip, lang_code)}")
        y -= 15

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        features = ["Glucose", "BloodPressure", "BMI", "Age"]
        fig, axs = plt.subplots(2, 2, figsize=(8, 6))
        axs = axs.flatten()
        for i, feature in enumerate(features):
            ax = axs[i]
            sns.histplot(data[data["Outcome"] == 1][feature], label=t("Diabetic", lang_code), color="red", ax=ax, kde=True, stat="count", alpha=0.5)
            sns.histplot(data[data["Outcome"] == 0][feature], label=t("Non-Diabetic", lang_code), color="green", ax=ax, kde=True, stat="count", alpha=0.5)
            ax.axvline(user_data[feature], color="blue", linestyle="--", label=t("User", lang_code))
            ax.set_title(t(feature, lang_code))
            ax.legend()

        plt.tight_layout()
        plt.savefig(tmpfile.name)
        plt.close()

        c.drawImage(tmpfile.name, 50, 100, width=500, preserveAspectRatio=True, mask='auto')

    c.save()
    buffer.seek(0)
    return buffer

# -----------------------
# Main App Logic
# -----------------------
model = load_model()
data = load_data()
st.set_page_config(page_title="Diabetes App", layout="centered")

# Language selection
if "language" not in st.session_state:
    st.session_state.language = "English"
if "lang_code" not in st.session_state:
    st.session_state.lang_code = "en"

language = st.sidebar.selectbox("🌐 Choose language",
    ["English", "Spanish", "French", "German", "Chinese", "Arabic", "Malay", "Japanese"],
    index=["English", "Spanish", "French", "German", "Chinese", "Arabic", "Malay", "Japanese"].index(st.session_state.language))

if language != st.session_state.language:
    st.session_state.language = language
    st.session_state.lang_code = {
        "English": "en", "Spanish": "es", "French": "fr", "German": "de",
        "Chinese": "zh-CN", "Arabic": "ar", "Malay": "ms", "Japanese": "ja"
    }[language]
    st.rerun()

lang_code = st.session_state.lang_code

# Page state
if "page" not in st.session_state:
    st.session_state.page = "Predict"
if "prediction" not in st.session_state:
    st.session_state.prediction = None
    st.session_state.inputs = {}
    st.session_state.confidence = None
if "user_name" not in st.session_state:
    st.session_state.user_name = "Anonymous"

navigation_labels = {"Predict": t("Predict", lang_code), "Report": t("Report", lang_code)}
selected_page = st.sidebar.radio(t("Navigation", lang_code), ["Predict", "Report"], format_func=lambda x: navigation_labels[x])
if selected_page != st.session_state.page:
    st.session_state.page = selected_page
    st.rerun()

# Page: Prediction
if st.session_state.page == "Predict":
    st.title(t("🩺 Diabetes Risk Predictor", lang_code))
    st.markdown(t("Enter your health data below:", lang_code))
    st.session_state.user_name = st.text_input(t("Enter your name:", lang_code), value=st.session_state.user_name)

    st.number_input(t("Glucose", lang_code), 0, 200, key="Glucose")
    st.number_input(t("Blood Pressure", lang_code), 40, 140, key="BloodPressure")
    st.number_input(t("BMI", lang_code), 10.0, 50.0, key="BMI")
    st.number_input(t("Age", lang_code), 0, 100, key="Age")

    if st.button(t("🔍 Predict", lang_code)):
        input_data = np.array([[st.session_state["Glucose"], st.session_state["BloodPressure"], st.session_state["BMI"], st.session_state["Age"]]])
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

        result = t("Diabetic", lang_code) if prediction == 1 else t("Not Diabetic", lang_code)
        st.success(f"{t('Prediction', lang_code)}: {result}")
        st.info(f"{t('Confidence', lang_code)}: {st.session_state.confidence}%")

    if st.session_state.prediction is not None:
        if st.button(t("🧾 View Report", lang_code)):
            st.session_state.page = "Report"
            st.rerun()

# Page: Report
elif st.session_state.page == "Report":
    st.title(t("🧾 Diabetes Report", lang_code))
    user_data = st.session_state.inputs
    features = ["Glucose", "BloodPressure", "BMI", "Age"]

    st.subheader(t("📈 Distribution Comparison", lang_code))
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()
    for i, feature in enumerate(features):
        ax = axs[i]
        sns.histplot(data[data["Outcome"] == 1][feature], label=t("Diabetic", lang_code), color="red", ax=ax, kde=True, stat="count", alpha=0.5)
        sns.histplot(data[data["Outcome"] == 0][feature], label=t("Non-Diabetic", lang_code), color="green", ax=ax, kde=True, stat="count", alpha=0.5)
        ax.axvline(user_data[feature], color="blue", linestyle="--", label=t("User", lang_code))
        ax.set_title(t(feature, lang_code))
        ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

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

    local_tz = datetime.now(pytz.timezone("Asia/Kuala_Lumpur"))
    local_time_str = format_datetime(local_tz, locale=lang_code)

    pdf = generate_pdf_report(
        user_data=user_data,
        prediction=st.session_state.prediction,
        confidence=st.session_state.confidence,
        health_tips=tips,
        data=data,
        user_name=st.session_state.user_name,
        local_time_str=local_time_str,
        lang_code=lang_code
    )

    st.download_button(
        label=t("📄 Download PDF Report", lang_code),
        data=pdf,
        file_name="diabetes_report.pdf",
        mime="application/pdf"
    )

    if st.button(t("🔙 Back to Prediction", lang_code)):
        st.session_state.page = "Predict"
        st.rerun()
