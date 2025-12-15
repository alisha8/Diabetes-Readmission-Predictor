import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("/Users/alishasarkar/Documents/Python Lab/Diabetes_new/Diabetes-Readmission-Predictor/Alisha_Approach/Data/raw/diabetic_data.csv")
df.replace("?", np.nan, inplace=True)

# Fill NaN and then map for Max_glu_serum and A1C_result
df['max_glu_serum'] = df['max_glu_serum'].fillna('None')
df['A1Cresult'] = df['A1Cresult'].fillna('None')


# Drop columns with too much missing or irrelevant
df.drop(['weight', 'payer_code', 'medical_specialty', 'encounter_id', 'patient_nbr'], axis=1, inplace=True)

# Create binary target variable
df['readmitted_30days'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
df.drop('readmitted', axis=1, inplace=True)

# Drop rows with unknown gender
df = df[df['gender'] != 'Unknown/Invalid']

# Fill missing values
cat_cols = df.select_dtypes(include='object').columns
num_cols = df.select_dtypes(include=np.number).columns

for col in cat_cols:
    df[col] = df[col].fillna("Unknown")
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Read ID mappings
mapping_df = pd.read_csv("/Data/IDS_mapping.csv", header=None)

def extract_mapping(df, section_name):
    start_idx = df[df[0] == section_name].index[0]
    try:
        end_idx = df.iloc[start_idx+1:].index[df.iloc[start_idx+1:][0].isna()].tolist()[0]
    except:
        end_idx = len(df)
    section = df.iloc[start_idx+1:end_idx].dropna()
    section.columns = [section_name, 'description']
    section[section_name] = section[section_name].astype(int)
    return section.set_index(section_name)['description'].to_dict()

# Map and replace ID columns
adm_map = extract_mapping(mapping_df, 'admission_type_id')
dis_map = extract_mapping(mapping_df, 'discharge_disposition_id')
src_map = extract_mapping(mapping_df, 'admission_source_id')

df['admission_type'] = df['admission_type_id'].map(adm_map).fillna("Unknown")
df['discharge_disposition'] = df['discharge_disposition_id'].map(dis_map).fillna("Unknown")
df['admission_source'] = df['admission_source_id'].map(src_map).fillna("Unknown")

df.drop(['admission_type_id', 'discharge_disposition_id', 'admission_source_id'], axis=1, inplace=True)

# Drop patients who died (discharge_disposition = Expired)
df = df[df['discharge_disposition'] != 'Expired']

# One-hot encode the mapped label columns
df = pd.get_dummies(df, columns=['admission_type', 'discharge_disposition', 'admission_source'])


# Replace '?' and 'Other' with 'Unknown' in race
df['race'] = df['race'].replace(["?","Other"], "Unknown")

# Label encode simple categorical columns
le = LabelEncoder()
label_cols = ['race', 'gender', 'age', 'change', 'diabetesMed']
for col in label_cols:
    df[col] = le.fit_transform(df[col])

#Define ICD-to-category mapping for diagnosis columns
def map_icd_to_category(icd_code):
    if pd.isnull(icd_code):
        return 'Unknown'
    code = str(icd_code)
    if code.startswith('250'):
        return 'Diabetes'
    if code.startswith('V'):
        return 'Supplementary'
    if code.startswith('E'):
        return 'External Causes'
    try:
        num = int(code[:3])
    except ValueError:
        return 'Other'
    
    if 390 <= int(code[:3]) <= 459 or code.startswith('785'):
        return 'Circulatory'
    if 460 <= int(code[:3]) <= 519 or code.startswith('786'):
        return 'Respiratory'
    if 520 <= int(code[:3]) <= 579:
        return 'Digestive'
    if 580 <= int(code[:3]) <= 629:
        return 'Genitourinary'
    if 800 <= int(code[:3]) <= 999:
        return 'Injury'
    if 710 <= int(code[:3]) <= 739:
        return 'Musculoskeletal'
    if 140 <= int(code[:3]) <= 239:
        return 'Neoplasms'
    return 'Other'

# Apply mapping to columns
for col in ['diag_1', 'diag_2', 'diag_3']:
    new_col = col + '_Category'
    df.insert(df.columns.get_loc(col) + 1, new_col, df[col].apply(map_icd_to_category))
    insert_idx = df.columns.get_loc(col)
    dummies = pd.get_dummies(df[new_col], prefix=col)
    
    # 3. Drop original & category column
    df.drop(columns=[col, new_col], inplace=True)

    # 4. Insert dummies at original column position
    for i, dummy_col in enumerate(dummies.columns):
        df.insert(insert_idx + i, dummy_col, dummies[dummy_col])

# Glu_map and A1C_result column to numeric
glu_map = {'None': 0, 'Norm': 1, '>200': 2, '>300': 3}
a1c_map = {'None': 0, 'Norm': 1, '>7': 2, '>8': 3}

df['max_glu_serum'] = df['max_glu_serum'].map(glu_map)
df['A1Cresult'] = df['A1Cresult'].map(a1c_map)

# Encode drug features
drug_cols = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
    'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
    'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton',
    'insulin', 'glyburide-metformin', 'glipizide-metformin',
    'glimepiride-pioglitazone', 'metformin-rosiglitazone',
    'metformin-pioglitazone'
]
drug_map = {'No': 0, 'Down': 1, 'Steady': 2, 'Up': 3}
for col in drug_cols:
    if col in df.columns:
        df[col] = df[col].map(drug_map)

# Boolean to int
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

# Save cleaned data
df.to_csv("/Users/alishasarkar/Documents/Python Lab/Diabetes_new/Diabetes-Readmission-Predictor/Alisha_Approach/Data/clean/Clean_data_for_train(1).csv", index=False)
print("Cleaned training data saved as 'Data/clean/Clean_data_for_train(1).csv'")