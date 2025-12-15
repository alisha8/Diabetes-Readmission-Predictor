import pandas as pd
import numpy as np

# Define your mappings *here* (these must match your training)
age_map = {
    "[0-10)": 0, "[10-20)": 1, "[20-30)": 2,
    "[30-40)": 3, "[40-50)": 4, "[50-60)": 5,
    "[60-70)": 6, "[70-80)": 7, "[80-90)": 8, "[90-100)": 9
}
glu_map = {'None': 0, 'Norm': 1, '>200': 2, '>300': 3}
a1c_map = {'None': 0, 'Norm': 1, '>7': 2, '>8': 3}
yes_no_map = {'Yes': 1, 'No': 0}
change_map = {"Ch": 0, "No": 1, "Other": 2}
insulin_map = {'No': 0, 'Down': 1, 'Steady': 2, 'Up': 3}

# You *must* match all the encodings you used
def preprocess_single_input(input_df):
    df = input_df.copy()

    df['age'] = df['age'].map(age_map)
    df['gender'] = df['gender'].map({"Male": 1, "Female": 0})
    df['race'] = df['race'].replace(["?", "Other"], "Unknown")
    race_encoder = {"AfricanAmerican": 0, "Asian": 1, "Caucasian": 2, "Hispanic": 3, "Unknown": 4}
    df['race'] = df['race'].map(race_encoder)

    df['change'] = df['change'].map(change_map)
    df['diabetesMed'] = df['diabetesMed'].map(yes_no_map)
    df['insulin'] = df['insulin'].map(insulin_map)
    df['A1Cresult'] = df['A1Cresult'].map(a1c_map)
    df['max_glu_serum'] = df['max_glu_serum'].map(glu_map)

    # Binary encoding for drugs
    drug_cols = ['metformin', 'glimepiride', 'glyburide','pioglitazone', 'rosiglitazone']
    for col in drug_cols:
        df[col] = df[col].map(yes_no_map)

    # One-hot encode diagnosis
    diagnosis_categories = [
        "Circulatory", "Respiratory", "Digestive", "Diabetes", 
        "Injury", "Musculoskeletal", "Genitourinary", "Neoplasms", "Other"
    ]
    for col in ['diag_1', 'diag_2']:
        for cat in diagnosis_categories:
            df[f"{col}_{cat}"] = (df[col] == cat).astype(int)
        df.drop(columns=[col], inplace=True)

    # One-hot encode admission_type
    admission_types = ["Elective", "Emergency", "Urgent", "Trauma Center", "Other"]
    for cat in admission_types:
        df[f"admission_type_{cat}"] = (df["admission_type"] == cat).astype(int)
    df.drop(columns=["admission_type"], inplace=True)

    # One-hot encode discharge_disposition
    discharge_types = [
        "Discharged to home", "Hospice / home", 
        "Another rehab facility", "Another institution type"
    ]
    for cat in discharge_types:
        df[f"discharge_disposition_{cat}"] = (df["discharge_disposition"] == cat).astype(int)
    df.drop(columns=["discharge_disposition"], inplace=True)

    return df
