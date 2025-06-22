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

# -----------------------
# Function to translate text
# -----------------------
def t(text, lang="en"):
    if lang == "en":
        return text
    try:
        return GoogleTranslator(source='auto', target=lang).translate(text)
    except:
        return text  # fallback

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
def generate_pdf_report(user_data, prediction, confidence, health_tips, data, user_name, local_time_str):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Diabetes Prediction Report")
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 70, f"Generated on: {local_time_str}")
    c.drawString(50, height - 85, f"Name: {user_name}")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 110, "User Data:")
    c.setFont("Helvetica", 11)
    y = height - 130
    for key, value in user_data.items():
        c.drawString(60, y, f"{key}: {value}")
        y -= 15

    result = "Diabetic" if prediction == 1 else "Not Diabetic"
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y - 10, f"Prediction: {result}")
    c.drawString(50, y - 30, f"Confidence: {confidence}%")

    y = y - 60
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Health Recommendations:")
    y -= 20
    c.setFont("Helvetica", 10)
    for tip in health_tips:
        c.drawString(60, y, f"- {tip}")
        y -= 15

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

# -----------------------
# Sidebar Language Selector
# -----------------------
lang_code = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese": "zh-CN",
    "Arabic": "ar",
    "Malay": "ms",
    "Japanese": "ja"
}[language]

st.sidebar.subheader(t("🌐 Language", lang_code))
language = st.sidebar.selectbox(t("Choose language", lang_code), ["English", "Spanish", "French", "German", "Chinese", "Arabic", "Malay", "Japanese"])


navigation_labels = {
    "Predict": t("Predict", lang_code),
    "Report": t("Report", lang_code)
}

selected_page = st.sidebar.radio(t("Navigation", lang_code), ["Predict", "Report"], format_func=lambda x: navigation_labels[x])

# ---------------------------
# Prediction Page
# ---------------------------
if selected_page == "Predict":
    st.title(t("🩺 Diabetes Risk Predictor", lang_code))
    st.markdown(t("Enter your health data below:", lang_code))

    user_name = st.text_input(t("Enter your name:", lang_code), value=st.session_state.user_name)
    st.session_state.user_name = user_name

    st.number_input(t("Glucose", lang_code), 0, 200, key="Glucose")
    st.number_input(t("Blood Pressure", lang_code), 40, 140, key="BloodPressure")
    st.number_input(t("BMI", lang_code), 10.0, 50.0, key="BMI")
    st.number_input(t("Age", lang_code), 0, 100, key="Age")

    if st.button(t("🔍 Predict", lang_code)):
        input_data = np.array([[st.session_state[k] for k in ["Glucose", "BloodPressure", "BMI", "Age"]]])
        prediction = model.predict(input_data)[0]
        confidence = model.predict_proba(input_data)[0][prediction]

        st.session_state.prediction = prediction
        st.session_state.confidence = round(confidence * 100, 2)
        st.session_state.inputs = {k: st.session_state[k] for k in ["Glucose", "BloodPressure", "BMI", "Age"]}

        result = t("Diabetic", lang_code) if prediction == 1 else t("Not Diabetic", lang_code)
        st.success(f"{t('Prediction', lang_code)}: {result}")
        st.info(f"{t('Confidence', lang_code)}: {st.session_state.confidence}%")

    if st.session_state.prediction is not None:
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button(t("🧾 View Report", lang_code)):
                st.session_state.page = "Report"
                st.rerun()
        with col2:
            if st.session_state.prediction == 1:
                st.markdown("[📍 Find Nearby Clinics](https://www.google.com/maps/search/diabetes+clinic+near+me)", unsafe_allow_html=True)

# ---------------------------
# Report Page
# ---------------------------
elif selected_page == "Report":
    st.title(t("🧾 Diabetes Report", lang_code))
    user_data = st.session_state.inputs
    features = ["Glucose", "BloodPressure", "BMI", "Age"]

    colorblind_mode = st.checkbox(t("♿ Enable colorblind-friendly palette", lang_code), value=False)
    diabetic_color = "#E69F00" if colorblind_mode else "red"
    non_diabetic_color = "#56B4E9" if colorblind_mode else "green"
    user_color = "#009E73" if colorblind_mode else "blue"

    st.subheader(t("📌 Feature Comparison to Diabetic Averages", lang_code))
    diabetic_avg = data[data["Outcome"] == 1][features].mean()
    for feature in features:
        user_val = user_data[feature]
        avg_val = diabetic_avg[feature]
        delta = user_val - avg_val
        color = "red" if delta > 0 else "green"
        st.markdown(f"**{feature}**: {user_val} _(Avg: {round(avg_val,1)})_ → <span style='color:{color}'>{t('High', lang_code) if delta > 0 else t('Low', lang_code)}</span>", unsafe_allow_html=True)

    st.subheader(t("📈 Distribution Comparison", lang_code))
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()
    for i, feature in enumerate(features):
        ax = axs[i]
        sns.histplot(data[data["Outcome"] == 1][feature], label="Diabetic", color=diabetic_color, ax=ax, kde=True, stat="count", alpha=0.5)
        sns.histplot(data[data["Outcome"] == 0][feature], label="Non-Diabetic", color=non_diabetic_color, ax=ax, kde=True, stat="count", alpha=0.5)
        ax.axvline(user_data[feature], color=user_color, linestyle="--", label="Your Value")
        ax.set_title(f"{feature}")
        ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader(t("💡 Suggestions to Improve Your Health", lang_code))
    tips = []
    if user_data["Glucose"] > 125:
        tips.append(t("High glucose — reduce sugar intake and monitor carbohydrate consumption.", lang_code))
    if user_data["BMI"] > 30:
        tips.append(t("High BMI — consider regular physical activity and healthy eating.", lang_code))
    if user_data["BloodPressure"] > 120:
        tips.append(t("Elevated blood pressure — reduce salt, avoid stress, and monitor regularly.", lang_code))
    if user_data["Age"] > 45:
        tips.append(t("Regular screenings are recommended due to age-related risks.", lang_code))
    if not tips:
        tips = [t("All your values are within the healthy range!", lang_code)]
    for tip in tips:
        st.markdown(f"✅ {tip}")

    st.subheader(t("📤 Download Report", lang_code))
    local_tz = datetime.now(pytz.timezone("Asia/Kuala_Lumpur"))
    local_time_str = local_tz.strftime('%A, %B %d, %Y at %I:%M %p (GMT%z)')
    local_time_str = local_time_str[:-2] + ":" + local_time_str[-2:]

    pdf = generate_pdf_report(
        user_data=user_data,
        prediction=st.session_state.prediction,
        confidence=st.session_state.confidence,
        health_tips=tips,
        data=data,
        user_name=st.session_state.user_name,
        local_time_str=local_time_str
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
