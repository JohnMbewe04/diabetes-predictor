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
import geocoder
import webbrowser
from functools import lru_cache
import matplotlib.font_manager as fm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import base64
import requests

if "show_intro" not in st.session_state:
    st.session_state["show_intro"] = True

# --- Footer CSS ---
st.markdown(
    """
    <style>
    .app-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: rgba(255, 255, 255, 0.9);
        color: #333;
        text-align: center;
        padding: 0.5rem 1rem;
        font-size: 0.9rem;
        z-index: 1000;
        box-shadow: 0 -2px 8px rgba(0,0,0,0.1);
    }
    [data-theme="dark"] .app-footer {
        background-color: rgba(0, 0, 0, 0.8);
        color: #ddd;
        box-shadow: 0 -2px 8px rgba(0,0,0,0.5);
    }
    .app-footer a {
        color: inherit;
        text-decoration: underline;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Footer HTML ---
st.markdown(
    """
    <div class="app-footer">
      👋 Welcome to the Diabetes Predictor App!  
      Predictions are not medical advice.  
      <a href="https://www.google.com/maps/search/diabetic+medical+facilities+near+me" target="_blank">
        Find nearby clinics
      </a>
    </div>
    """,
    unsafe_allow_html=True,
)


if "play_music" not in st.session_state:
    st.session_state.play_music = True  # default: don't play

@lru_cache(maxsize=1000)
def cached_translate(text, lang):
    if lang == "en":
        return text
    try:
        return GoogleTranslator(source='auto', target=lang).translate(text)
    except:
        return text

# Use this everywhere instead of `t(...)`
def t(text, lang="en"):
    return cached_translate(text, lang)


# Language mappings
LANGUAGE_SETTINGS = {
    "English": {"translate": "en", "locale": "en_US"},
    "Spanish": {"translate": "es", "locale": "es_ES"},
    "French": {"translate": "fr", "locale": "fr_FR"},
    "German": {"translate": "de", "locale": "de_DE"},
    "Chinese": {"translate": "zh-CN", "locale": "zh_CN"},
    "Arabic": {"translate": "ar", "locale": "ar_EG"},
    "Malay": {"translate": "ms", "locale": "ms_MY"},
    "Japanese": {"translate": "ja", "locale": "ja_JP"}
}

@st.cache_resource
def load_model():
    return joblib.load("diabetes_model_clean.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("diabetes.csv")

def generate_pdf_report(user_data, prediction, confidence, health_tips, data, user_name, local_time_str, lang_code):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    bold_font = "Helvetica"  # You can register bold versions if you have them    
    c.setFont("Helvetica", 16)
    c.drawString(50, height - 50, t("Diabetes Prediction Report", lang_code))
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 70, f"{t('Generated on', lang_code)}: {local_time_str}")
    c.drawString(50, height - 85, f"{t('Name', lang_code)}: {user_name}")

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 110, t("User Data", lang_code) + ":")
    c.setFont("Helvetica", 11)
    y = height - 130
    for key, value in user_data.items():
        c.drawString(60, y, f"{t(key, lang_code)}: {value}")
        y -= 15

    result = t("Diabetic", lang_code) if prediction == 1 else t("Not Diabetic", lang_code)
    c.setFont("Helvetica", 12)
    c.drawString(50, y - 10, f"{t('Prediction', lang_code)}: {result}")
    c.drawString(50, y - 30, f"{t('Confidence', lang_code)}: {confidence}%")

    y -= 60
    c.setFont("Helvetica", 12)
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


# -- Theme selector
theme = st.radio("Choose Theme", ["Light", "Dark"], horizontal=True)

# -- Function to apply background image based on theme
def set_background(theme):
    if theme == "Dark":
        image_url = "https://raw.githubusercontent.com/JohnMbewe04/diabetes-predictor/main/dark_background.jpg"
        overlay_opacity = 0.6
    else:
        image_url = "https://raw.githubusercontent.com/JohnMbewe04/diabetes-predictor/main/light_background.jpeg"
        overlay_opacity = 0.2

    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            encoded = base64.b64encode(response.content).decode()

            st.markdown(
                f"""
                <style>
                .stApp {{
                    animation: fadeIn 1s ease-in-out;
                    background: linear-gradient(rgba(0, 0, 0, {overlay_opacity}), rgba(0, 0, 0, {overlay_opacity})),
                                url("data:image/jpeg;base64,{encoded}");
                    background-size: cover;
                    background-position: center;
                    transition: background 0.8s ease-in-out;
                }}

                @keyframes fadeIn {{
                    0% {{ opacity: 0; }}
                    100% {{ opacity: 1; }}
                }}
                </style>
                """,
                unsafe_allow_html=True
            )
        else:
            st.warning("⚠️ Could not load background image.")
    except Exception as e:
        st.error(f"Error loading background: {e}")

# -- Function to apply custom widget and text styles
def set_theme_styles(theme):
    if theme == "Dark":
        st.markdown(
            """
            <style>
            html, body, [class*="st-"] {{
                color: #f8f9fa;
                background-color: #1e1e1e;
            }}
            .stTextInput > div > input {{
                background-color: #2e2e2e;
                color: white;
            }}
            .stRadio > div {{
                background-color: #2e2e2e;
                color: white;
                padding: 0.5rem;
                border-radius: 8px;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <style>
            html, body, [class*="st-"] {{
                color: #000000;
                background-color: #ffffff;
            }}
            .stTextInput > div > input {{
                background-color: #f0f0f0;
                color: black;
            }}
            .stRadio > div {{
                background-color: #f0f0f0;
                color: black;
                padding: 0.5rem;
                border-radius: 8px;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

# --- Load and encode your audio file ---
def get_audio_base64(audio_file_path):
    with open(audio_file_path, "rb") as f:
        data = f.read()
        return base64.b64encode(data).decode()

# --- Play or stop music using HTML/JS ---
def audio_player_ui(audio_base64, play_audio):
    if play_audio:
        st.markdown(f"""
        <audio id="bg-music" autoplay loop>
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>
        <script>
            var music = document.getElementById("bg-music");
            if (music) {{
                music.play();
            }}
        </script>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <script>
            var music = document.getElementById("bg-music");
            if (music) {
                music.pause();
                music.currentTime = 0;
            }
        </script>
        """, unsafe_allow_html=True)


# ----------------------- MAIN APP -----------------------
model = load_model()
data = load_data()
    
# -- Apply styles and background
set_theme_styles(theme)
set_background(theme)
    
# Load your actual file
audio_b64 = get_audio_base64("background_music.MP3")
audio_player_ui(audio_b64, st.session_state.play_music)
    
# Session init
if "page" not in st.session_state:
    st.session_state.page = "Predict"
if "prediction" not in st.session_state:
    st.session_state.prediction = None
    st.session_state.inputs = {}
    st.session_state.confidence = None
if "user_name" not in st.session_state:
    st.session_state.user_name = "Anonymous"
        
if "language" not in st.session_state:
    st.session_state.language = "English"
    
if "lang_code" not in st.session_state:
    st.session_state.lang_code = LANGUAGE_SETTINGS[st.session_state.language]["translate"]
    
if "locale_code" not in st.session_state:
    st.session_state.locale_code = LANGUAGE_SETTINGS[st.session_state.language]["locale"]
    
    
# Language selector
language = st.sidebar.selectbox("🌐 Choose Language", list(LANGUAGE_SETTINGS.keys()), index=list(LANGUAGE_SETTINGS.keys()).index(st.session_state.language))
if language != st.session_state.language:
    st.session_state.language = language
    st.session_state.lang_code = LANGUAGE_SETTINGS[language]["translate"]
    st.session_state.locale_code = LANGUAGE_SETTINGS[language]["locale"]
    st.rerun()
    
lang_code = st.session_state.lang_code
locale_code = st.session_state.locale_code
    
# Toggle in sidebar
music_play_label = "▶️ " + t("Play Music", lang_code)
music_stop_label = "⏹ " + t("Stop Music", lang_code)
    
play_toggle = st.sidebar.button(music_play_label if not st.session_state.play_music else music_stop_label)
    
if play_toggle:
    st.session_state.play_music = not st.session_state.play_music
    
# Page navigation
selected_page = st.sidebar.radio(t("Navigation", lang_code), ["Predict", "Report"], index=["Predict", "Report"].index(st.session_state.page))
if selected_page != st.session_state.page:
    st.session_state.page = selected_page
    st.rerun()
    
# ------------------- Page: Prediction -------------------
if st.session_state.page == "Predict":
    st.title(t("🩺 Diabetes Risk Predictor", lang_code))
    st.markdown(t("Enter your health data below:", lang_code))
    # Close popup on button press
    if st.session_state.get("show_intro") and st.session_state.get("close_popup"):
        st.session_state["show_intro"] = False

    # End blur container if popup is active
    if st.session_state["show_intro"]:
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.session_state.user_name = st.text_input(t("Enter your name:", lang_code), value=st.session_state.user_name)
    
    glucose = st.number_input(t("Glucose", lang_code), 0, 200, 100)
        
    st.markdown(t("Blood Pressure (Systolic / Diastolic)", lang_code))
    col1, col_mid, col2 = st.columns([2, 0.3, 2])  # narrower middle column
        
    systolic = col1.number_input(
        t("Systolic (mmHg)", lang_code),
        min_value=70,
        max_value=200,
        value=120,
        key="systolic_input"
    )
        
    # Center the slash vertically using HTML
    col_mid.markdown(
        "<div style='text-align: center; padding-top: 38px; font-size: 20px;'>/</div>",
        unsafe_allow_html=True
    )
        
    diastolic = col2.number_input(
        t("Diastolic (mmHg)", lang_code),
        min_value=40,
        max_value=120,
        value=80,
        key="diastolic_input"
    )
        
    # Compute MAP
    bp = round((2 * diastolic + systolic) / 3)
    bmi = st.number_input(t("BMI", lang_code), 10.0, 50.0, 25.0)
    age = st.number_input(t("Age", lang_code), 1, 100, 30)
    
    if st.button(t("🔍 Predict", lang_code)) and bp is not None:
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
        result = t("Diabetic", lang_code) if prediction == 1 else t("Not Diabetic", lang_code)
        st.markdown(f"""
            <div style="
                background-color: rgba(255, 255, 255, 0.85);
                padding: 1rem;
                border-radius: 10px;
                color: black;
                font-weight: bold;
                font-size: 18px;
                margin-top: 10px;
            ">
                ✅ {t('Prediction', lang_code)}: {result}<br>
                📊 {t('Confidence', lang_code)}: {st.session_state.confidence}%
            </div>
        """, unsafe_allow_html=True)
    
    if st.session_state.prediction is not None:
        col1, col2 = st.columns(2)
        with col1:
            if st.button(t("🧾 View Report", lang_code)):
                st.session_state.page = "Report"
                st.rerun()
        with col2:
            if st.session_state.prediction == 1:
                st.markdown(f"[📍 {t('Find Nearby Clinics', lang_code)}](https://www.google.com/maps/search/diabetes+clinic+near+me)", unsafe_allow_html=True)
    
# ------------------- Page: Report -------------------
elif st.session_state.page == "Report":
    st.title(t("🧾 Diabetes Report", lang_code))
    user_data = st.session_state.inputs
    features = ["Glucose", "BloodPressure", "BMI", "Age"]
    
    colorblind_mode = st.checkbox("♿ " + t("Enable colorblind-friendly palette", lang_code), value=False)
    if colorblind_mode:
        st.caption(t("🎨 Using colorblind-safe colors for the graphs.", lang_code))
        diabetic_color = "#E69F00"
        non_diabetic_color = "#56B4E9"
        user_color = "#009E73"
    else:
        diabetic_color = "red"
        non_diabetic_color = "green"
        user_color = "blue"
    
    st.subheader(t("📌 Feature Comparison to Diabetic Averages", lang_code))
    diabetic_avg = data[data["Outcome"] == 1][features].mean()
    for feature in features:
        user_val = user_data[feature]
        avg_val = diabetic_avg[feature]
        delta = user_val - avg_val
        color = "red" if delta > 0 else "green"
        st.markdown(
            f"**{t(feature, lang_code)}**: {user_val} _(Avg: {round(avg_val,1)})_ → "
            f"<span style='color:{color}'>{t('High', lang_code) if delta > 0 else t('Low', lang_code)}</span>",
            unsafe_allow_html=True
        )
    
    st.subheader(t("📈 Distribution Comparison", lang_code))
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()
    for i, feature in enumerate(features):
        ax = axs[i]
        sns.histplot(data[data["Outcome"] == 1][feature], label=t("Diabetic", lang_code),
                        color=diabetic_color, ax=ax, kde=True, stat="count", alpha=0.5)
        sns.histplot(data[data["Outcome"] == 0][feature], label=t("Non-Diabetic", lang_code),
                        color=non_diabetic_color, ax=ax, kde=True, stat="count", alpha=0.5)
        ax.axvline(user_data[feature], color=user_color, linestyle="--", label=t("Your Value", lang_code))
        ax.set_title(f"{t(feature, lang_code)}")
        ax.legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.subheader(t("💡 Suggestions to Improve Your Health", lang_code))
    tips = []
    if user_data["Glucose"] > 125:
        tips.append("High glucose — reduce sugar intake and monitor carbohydrate consumption.")
    if user_data["BMI"] > 30:
        tips.append("High BMI — consider regular physical activity and healthy eating.")
    if user_data["BloodPressure"] > 120:
        tips.append("Elevated blood pressure — reduce salt, avoid stress, and monitor regularly.")
    if user_data["Age"] > 45:
        tips.append("Regular screenings are recommended due to age-related risks.")
    if tips:
        for tip in tips:
            st.markdown(f"✅ {t(tip, lang_code)}")
    else:
        st.success(t("👍 All your values are within the healthy range!", lang_code))
    
    local_time_str = format_datetime(datetime.now(pytz.timezone("Asia/Kuala_Lumpur")), locale=locale_code)
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



