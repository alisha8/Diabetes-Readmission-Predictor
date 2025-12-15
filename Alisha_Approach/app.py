import streamlit as st
import pandas as pd
import os
import re
import joblib
from Scripts.preprocessing_module import preprocess_single_input
from Scripts.LOS_Regression import run_linear_regression, run_random_forest, run_xgboost

def normalize_string(s):
    # Replace _ and - with space, remove extra spaces, and lowercase everything
    return re.sub(r'\s+', ' ', re.sub(r'[-_]', ' ', s)).strip().lower()

def los_view():
    st.title("🏥 Length of Stay Regression")
    model_choice = st.selectbox("Select Regressor Model", ["Linear Regression", "Random Forest", "XGBoost"])

    if st.button("Run Regression"):
        if model_choice == "Linear Regression":
            output = run_linear_regression()
        elif model_choice == "Random Forest":
            output = run_random_forest()
        else:
            output = run_xgboost()
        st.text(output)

def eda_view():
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.markdown("""
    This section provides a comprehensive look at the dataset's structure and distributions.
    EDA helps uncover important patterns, spot anomalies, and guide feature engineering for better predictions.
    """)
    report_folder = "/Users/alishasarkar/Documents/Python Lab/Diabetes_new/Diabetes-Readmission-Predictor/Alisha_Approach/Reports/EDA_reports"

    image_files = [
        os.path.join(report_folder, f)
        for f in os.listdir(report_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    if image_files:
        for img_path in image_files:
            st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)
    else:
        st.warning(f"⚠️ No image files found in '{report_folder}/'.")

def group_view():
    st.title("👥 Group Analysis")  

    st.markdown("""
    This section examines readmission rates across different patient groups.
    Group analysis helps identify high-risk populations and tailor interventions more effectively.
    """)

    report_folder = "/Users/alishasarkar/Documents/Python Lab/Diabetes_new/Diabetes-Readmission-Predictor/Alisha_Approach/Reports/Group_Analysis_reports"
    image_files = [
        os.path.join(report_folder, f)
        for f in os.listdir(report_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    if image_files:
        for img_path in image_files:
            st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)
    else:
        st.warning(f"⚠️ No image files found in '{report_folder}/'.")
    
def feature_insight_view():
    st.title("🔍 Feature Insight")
    st.markdown("Select a model to view its feature insight reports.")

    model_names = ["Logistic Regression", "Random Forest", "XGBoost"]
    selected_model = st.selectbox("Select Model", model_names)
    normalized_model = normalize_string(selected_model)

    report_folder = "/Users/alishasarkar/Documents/Python Lab/Diabetes_new/Diabetes-Readmission-Predictor/Alisha_Approach/Reports/Model_reports"

    matching_images = []
    for f in os.listdir(report_folder):
        filename_no_ext = os.path.splitext(f)[0]
        normalized_filename = normalize_string(filename_no_ext)

        if normalized_filename.startswith(normalized_model) and f.lower().endswith(('.png', '.jpg', '.jpeg')):
            matching_images.append(os.path.join(report_folder, f))

    if matching_images:
        for img_path in matching_images:
            st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)
    else:
        st.warning(f"⚠️ No report images found for '{selected_model}' in '{report_folder}/'.")

def patient_cluster_view():
    st.title("🧬 Patient Clustering Analysis")

    st.markdown("""
        This section provides insights from unsupervised learning to identify patient subgroups with similar characteristics.
        The clustering helps understand patterns in readmission risks and treatment responses.
    """)

    st.subheader("UMAP 3D Cluster Visualization")
    st.image("/Users/alishasarkar/Documents/Python Lab/Diabetes_new/Diabetes-Readmission-Predictor/Alisha_Approach/Reports/Cluster_reports/Cluster(UMAP_3D).png", use_container_width=True, caption="UMAP 3D Projection of Patient Clusters")

    st.subheader("UMAP 2D Clustering")
    st.image("/Users/alishasarkar/Documents/Python Lab/Diabetes_new/Diabetes-Readmission-Predictor/Alisha_Approach/Reports/Cluster_reports/Cluster(UMAP).png", use_container_width=True, caption="UMAP 2D Visualization of Patient Clusters")


# Load the models
@st.cache_resource
def load_models():
    logistic = joblib.load('/Users/alishasarkar/Documents/Python Lab/Diabetes_new/Diabetes-Readmission-Predictor/Alisha_Approach/Models/trained/Logistic_model.pkl')
    rf = joblib.load('/Users/alishasarkar/Documents/Python Lab/Diabetes_new/Diabetes-Readmission-Predictor/Alisha_Approach/Models/trained/Random_Forest_model.pkl')
    xgb = joblib.load('/Users/alishasarkar/Documents/Python Lab/Diabetes_new/Diabetes-Readmission-Predictor/Alisha_Approach/Models/trained/XGBoost_model.pkl')
    return logistic, rf, xgb

Logistic_model, Random_Forest_model, XGBoost_model = load_models()

# Load feature columns
feature_columns = joblib.load('/Users/alishasarkar/Documents/Python Lab/Diabetes_new/Diabetes-Readmission-Predictor/Alisha_Approach/Data/split_data/feature_columns.pkl')

############# Page layout
st.set_page_config(page_title="Readmission Predictor", layout="wide")
menu = st.sidebar.radio("Navigate", ["Home", "Upload CSV", "EDA", "Readmission Prediction", "Feature Insights", "Group Analysis", "Patient Clusters", "Length of Stay" ])


if menu == "Home":
    st.title("🩺 Diabetes Readmission Prediction")
    st.markdown("""
        Welcome! This app helps predict the **risk of 30-day hospital readmission** for diabetes patients.
        It's designed for clinicians, researchers, and students to explore **predictive modeling** in healthcare.
        """)

    st.subheader("✨ What You Can Do Here")
    #st.image("assets/features.png", width=400, caption="Features at a Glance")
    st.markdown("""
        - **Predict** individual patient readmission risk
        - **Upload** and explore your own datasets
        - **Analyze** feature importance
        - **Discover** patient subgroups with clustering
        """)

    st.subheader("📌 How to Get Started")
    #st.image("assets/steps.png", width=400, caption="Quick Start Guide")
    st.markdown("""
        1. Use the **sidebar** to navigate.
        2. Visit **Upload CSV** to preview your data.
        3. Try **Readmission Prediction** to test new patient profiles.
        4. Explore **Feature Insights** and **Patient Clusters** for deeper analysis.
        """)

    st.info("🔎 Built for learning, research, and clinical exploration!")
    
elif menu == "Upload CSV":
    st.title("📂 Uploaded Dataset: Preview & Info")
    try:
        df = pd.read_csv("Data/clean/Clean_data_for_gui.csv") 
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

elif menu == "Readmission Prediction":
    st.title("🔮 Predict Readmission")

    #Model selection
    model_choice = st.selectbox(
        "Choose Prediction Model",
        ["Logistic Regression", "Random Forest", "XGBoost"]
    )

    st.markdown("### Enter patient information below: \n")

    #Demographics
    st.subheader("Demographics")
    age = st.selectbox("Age Group", ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"])

    #Hospital encounter
    st.subheader("Hospital Encounter")
    admission_type = st.selectbox("Admission Type", ["Elective", "Emergency", "Urgent", "Trauma Center", "Other"])
    discharge_disposition = st.selectbox(
        "Discharge Disposition",["Discharged to home", "Hospice / home", "Another rehab facility", "Another institution type"])

    # Clinical data
    st.subheader("\n Clinical Data")
    time_in_hospital = st.slider("Time in Hospital (days)", 1, 14, 5)
    num_lab_procedures = st.slider("Number of Lab Procedures", 1, 100, 40)
    num_procedures = st.slider("Number of Procedures", 0, 20, 0)
    num_medications = st.slider("Number of Medications", 1, 80, 20)
    number_inpatient = st.slider("Number of Inpatient Visits", 0, 20, 0)
    number_outpatient = st.slider("Number of Outpatient Visits", 0, 20, 0)
    number_emergency = st.slider("Number of Emergency Visits", 0, 20, 0)
    number_diagnoses = st.slider("Number of Diagnoses", 1, 20, 5)
    change = st.selectbox("Change of Medications", ["No", "Ch", "Other"])
    diabetesMed = st.selectbox("Diabetes Medication", ["Yes", "No"])
    insulin = st.selectbox("Insulin Usage", ["No", "Up", "Down", "Steady"])

    # Diagnoses
    st.subheader("Diagnosis")
    diag_1 = st.selectbox("Primary Diagnosis Category", [
        "Circulatory", "Respiratory", "Digestive", "Diabetes", "Injury", "Musculoskeletal", "Genitourinary", "Neoplasms", "Other"])
    diag_2 = st.selectbox("Secondary Diagnosis Category", [
        "Circulatory", "Respiratory", "Digestive", "Diabetes", "Injury", "Musculoskeletal", "Genitourinary", "Neoplasms", "Other"])

    # Lab results
    st.subheader("Lab Results")
    A1Cresult = st.selectbox("A1C Result", ["None", "Norm", ">7", ">8"])
    max_glu_serum = st.selectbox("Max Glucose Serum", ["None", "Norm", ">200", ">300"])

    # Medications
    st.subheader("Medications")
    metformin = st.selectbox("Metformin", ["Yes", "No"])
    glimepiride = st.selectbox("Glimepiride", ["Yes", "No"])
    pioglitazone = st.selectbox("Pioglitazone", ["Yes", "No"])
    glyburide = st.selectbox("Glyburide", ["Yes", "No"])
    rosiglitazone = st.selectbox("Rosiglitazone", ["Yes", "No"])

    ############ Predict block
    if st.button("Predict"):
        st.info(f"🧠 Model selected:  {model_choice}")
    
        input_data = pd.DataFrame([{
            "age": age,
            "gender": gender,
            "race": race,
            "admission_type": admission_type,
            "discharge_disposition": discharge_disposition,
            "time_in_hospital": time_in_hospital,
            "num_lab_procedures": num_lab_procedures,
            "num_procedures": num_procedures,
            "num_medications": num_medications,
            "number_inpatient": number_inpatient,
            "number_outpatient": number_outpatient,
            "number_emergency": number_emergency,
            "number_diagnoses": number_diagnoses,
            "change": change,
            "diabetesMed": diabetesMed,
            "insulin": insulin,
            "diag_1": diag_1,
            "diag_2": diag_2,
            "A1Cresult": A1Cresult,
            "max_glu_serum": max_glu_serum,
            "metformin": metformin,
            "glimepiride": glimepiride,
            "pioglitazone": pioglitazone,
            "glyburide": glyburide,
            "rosiglitazone": rosiglitazone
        }])

        # Apply preprocessing
        input_data_processed = preprocess_single_input(input_data)
        input_data_processed = input_data_processed.reindex(columns=feature_columns, fill_value=0)

        # Pick model
        if model_choice == "Logistic Regression":
            model = Logistic_model
        elif model_choice == "Random Forest":
            model = Random_Forest_model 
        else:
            model = XGBoost_model

        # Predict
        prediction = model.predict(input_data_processed)[0]
        prob = model.predict_proba(input_data_processed)[0][1] if hasattr(model, "predict_proba") else None

        # Show result
        if prob is not None:
            if 0.95 <= prob <= 1.0:
                st.error("🚨 Prediction: Patient is **most likely** to be readmitted.")
            else:
                st.success("✅ Prediction: Patient is **unlikely** to be readmitted.")
            st.write(f"Predicted probability of readmission: **{prob:.2f}**")
        else:
            if prediction == 1:
                st.warning("⚠️ Model does not support probabilities.\nPrediction: Patient is **likely** to be readmitted.")
            else:
                st.success("✅ Model does not support probabilities.\nPrediction: Patient is **unlikely** to be readmitted.")

elif menu == "Feature Insights":
    feature_insight_view()

elif menu == "Patient Clusters":
    patient_cluster_view()

elif menu == "EDA":
    eda_view()

elif menu == "Group Analysis":
    group_view()

elif menu == "Length of Stay":
    los_view()
    
