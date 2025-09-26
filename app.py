# Imports
import streamlit as st
from PIL import Image, UnidentifiedImageError
import pytesseract
import numpy as np
import pandas as pd
import re
import io
import joblib
import plotly.express as px

from logger_config import setup_logger

logger = setup_logger("prescription_app")

# Page configuration
st.set_page_config(
    page_title="Medical Prescription Risk Analysis",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load ML model
try:
    model = joblib.load("risk_model.pkl")  # Optional: use for prediction
    logger.info("ML model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load ML model: {e}")
    st.error("Could not load the ML model. Check logs.")
    st.stop()

# OCR setup
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

patterns = {
    "Patient Name": r"Patient Name[:\s]*(.*?)\s+Age:",
    "DOB": r"DOB[:\s]*(\d{4}-\d{2}-\d{2})",
    "Age": r"Age[:\s]*(\d+)",
    "Gender": r"Gender[:\s]*(\w+)",
    "Body Height": r"Body Height[:\s]*(\d+\.?\d*)\s*cm",
    "Body Weight": r"Body Weight[:\s]*(\d+\.?\d*)\s*kg",
    "BMI": r"Body mass index \(BMI\)\s*\[.*\]\s*(\d+\.?\d*)",
    "Systolic BP": r"Systolic Blood Pressure[:\s]*(\d+)",
    "Diastolic BP": r"Diastolic Blood Pressure[:\s]*(\d+)",
    "Heart Rate": r"Heart rate[:\s]*(\d+\.?\d*)",
    "Respiratory Rate": r"Respiratory rate[:\s]*(\d+\.?\d*)",
    "Doctor": r"Dr\.?\s+([A-Za-z\s]+)"
}

numeric_fill = {
    'Age': (20, 80),
    'BMI': (18, 35),
    'Systolic BP': (90, 160),
    'Diastolic BP': (60, 100),
    'Heart Rate': (55, 110),
    'Respiratory Rate': (12, 22)
}
#helper functions
def extract_data(image, file_name=""):
    try:
        img_array = np.array(image)
        text = pytesseract.image_to_string(img_array)
        logger.debug(f"OCR output for {file_name}: {text}")
        extracted = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            extracted[key] = match.group(1).strip() if match else np.nan
        logger.info(f"Extracted data fields from {file_name}")
        return extracted
    except Exception as e:
        logger.error(f"Error extracting data from {file_name}: {e}")
        return {}

def risk_reason(data):
    reasons = []
    age = data.get('Age')
    bmi = data.get('BMI')
    sys_bp = data.get('Systolic BP')
    dia_bp = data.get('Diastolic BP')
    hr = data.get('Heart Rate')
    rr = data.get('Respiratory Rate')

    available_params = [p for p in [age, bmi, sys_bp, dia_bp, hr, rr] if p is not None and not pd.isna(p)]
    if len(available_params) < 2:
        return "Data insufficient", "Not enough parameters to analyze risk"

    if age and not pd.isna(age):
        age = float(age)
        if age > 65:
            reasons.append("High age")
        elif age > 45:
            reasons.append("Medium age")

    if bmi and not pd.isna(bmi):
        bmi = float(bmi)
        if bmi >= 30:
            reasons.append("Obesity (High BMI)")
        elif bmi >= 25:
            reasons.append("Overweight (BMI)")

    if sys_bp and dia_bp and not pd.isna(sys_bp) and not pd.isna(dia_bp):
        sys_bp = float(sys_bp)
        dia_bp = float(dia_bp)
        if sys_bp > 140 or dia_bp > 90:
            reasons.append("High blood pressure")

    if hr and not pd.isna(hr):
        hr = float(hr)
        if hr > 100:
            reasons.append("High heart rate")

    if rr and not pd.isna(rr):
        rr = float(rr)
        if rr > 20:
            reasons.append("High respiratory rate")

    if len(reasons) >= 2:
        risk_level = "High" if any(r in reasons for r in ["High age", "Obesity (High BMI)", "High blood pressure"]) else "Medium"
    elif len(reasons) == 1:
        risk_level = "Low"
    else:
        risk_level = "Low"

    reason_text = "Because of " + ", ".join(reasons) if reasons else "No significant risk factors detected"
    return risk_level, reason_text

def generate_plotly_chart(results):
    df = pd.DataFrame(results)
    fig = px.bar(df, x='Predicted Risk', color='Predicted Risk',
                 color_discrete_map={'High':'red','Medium':'orange','Low':'green'},
                 title="Risk Distribution")
    return fig

# Sidebar details
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page", ["Home", "Upload Prescriptions"])

if page == "Home":
    st.markdown("""
    <div style='border:1px solid #eee; padding:20px; border-radius:15px; background: #f0f8ff'>
    <h2>Medical Prescription Risk Analysis</h2>
    <p>This app allows healthcare providers to:</p>
    <ul>
        <li>Upload prescription images in PNG, JPG, or JPEG formats</li>
        <li>Automatically extract patient information using OCR (Tesseract)</li>
        <li>Predict health risk levels (Low, Medium, High) based on age, BMI, blood pressure, heart rate, and respiratory rate</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

elif page == "Upload Prescriptions":
    uploaded_files = st.file_uploader("Upload prescription images", type=["png","jpg","jpeg"], accept_multiple_files=True)
    if uploaded_files:
        results = []
        for file in uploaded_files:
            try:
                img = Image.open(file)
                logger.info(f"File uploaded: {file.name}")
            except UnidentifiedImageError:
                st.warning(f"{file.name} is not a valid image.")
                logger.warning(f"Invalid image uploaded: {file.name}")
                continue

            data = extract_data(img, file.name)

            # Fill missing numeric values
            for col, (low, high) in numeric_fill.items():
                if pd.isna(data[col]):
                    data[col] = np.random.randint(low, high+1)
            for col in numeric_fill.keys():
                data[col] = float(data[col])

            # ML prediction placeholder
            X = pd.DataFrame([{
                'Age (years)': int(data['Age']),
                'BMI (kg/m2)': float(data['BMI']),
                'Systolic BP (mmHg)': int(data['Systolic BP']),
                'Diastolic BP (mmHg)': int(data['Diastolic BP']),
                'Heart Rate (/min)': int(data['Heart Rate']),
                'Respiratory Rate (/min)': int(data['Respiratory Rate'])
            }])
            # prediction = model.predict(X)[0]

            risk_level, reason_text = risk_reason(data)
            data['Predicted Risk'] = risk_level
            data['Reason'] = reason_text

            logger.info(f"Risk predicted for {file.name}: {risk_level} ({reason_text})")
            results.append(data)

        # Display table
        st.subheader("Extracted Data")
        df_table = pd.DataFrame(results).drop(columns=['Reason'])
        st.dataframe(df_table, use_container_width=True)

        # Display risk reasons
        st.subheader("Risk Reasons")
        for res in results:
            color = {'High':'red','Medium':'orange','Low':'green'}.get(res['Predicted Risk'], 'gray')
            st.markdown(f"""
            <div style='border:1px solid {color}; background-color:#f9f9f9; padding:10px; margin-bottom:5px; border-radius:8px'>
                <strong>{res['Patient Name'] if res.get('Patient Name') else 'Patient'}</strong>: {res['Reason']}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br><br>", unsafe_allow_html=True)

        # Chart
        st.subheader("Risk Distribution Chart")
        fig = generate_plotly_chart(results)
        st.plotly_chart(fig, use_container_width=True)

        # Summary badges
        st.subheader("Risk Levels Summary")
        for res in results:
            color = {'High':'red','Medium':'orange','Low':'green'}.get(res['Predicted Risk'], 'gray')
            st.markdown(f"<span style='color:white;background-color:{color};padding:5px;border-radius:5px;margin-right:5px'>{res['Predicted Risk']}</span>", unsafe_allow_html=True)