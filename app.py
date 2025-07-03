import streamlit as st
import pandas as pd

st.set_page_config(page_title="Readmission Predictor", layout="wide")
menu = st.sidebar.radio("Navigate", ["Home", "Upload CSV", "Readmission Prediction", "Feature Insights", "Patient Clusters"])


if menu == "Home":
    st.title("Diabetes Readmission Prediction")
    st.markdown("Welcome to patient readmission predictor.")
    
    
elif menu == "Upload CSV":
    st.title("📂 Uploaded Dataset: Preview & Info")

    try:
        df = pd.read_csv("data/cleandata.csv") 

        st.success("✅ CSV loaded successfully from data folder!")

        st.write("### Preview of Data (first 5 rows)")
        st.dataframe(df.head())

        st.write("### Dataset Shape")
        st.write(f"{df.shape[0]} rows × {df.shape[1]} columns")

        st.write("### Missing Values (% per column)")
        missing_percent = df.isnull().mean() * 100
        st.dataframe(missing_percent[missing_percent > 0].round(2))

        st.info("Preprocessing will be added soon.")
    except FileNotFoundError:
        st.error("❌ CSV not found in 'data' folder.")

if menu == "Readmission Prediction":
    st.title("🔮 Predict Readmission")

    st.markdown("Enter patient information below:")

    age = st.selectbox("Age Group", ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    time_in_hospital = st.slider("Time in Hospital (days)", 1, 14, 5)
    num_lab_procedures = st.slider("Number of Lab Procedures", 1, 100, 40)
    num_medications = st.slider("Number of Medications", 1, 80, 20)
    number_inpatient = st.slider("Number of Inpatient Visits", 0, 20, 0)
    diagnosis = st.text_input("Primary Diagnosis (e.g. 250 for Diabetes)", "250")

    if st.button("Predict"):
        st.info("🧠 ML model will be integrated here soon.")
        st.success("✨ Placeholder: Patient is likely to be readmitted.")