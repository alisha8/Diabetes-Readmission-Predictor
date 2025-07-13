import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load raw data
df = pd.read_csv("Data/diabetic_data.csv")
# Replace "?" with NaN
df.replace("?", np.nan, inplace=True)

# Drop high-missing columns & unhelpful IDs
df.drop(['weight', 'payer_code', 'medical_specialty', 'encounter_id', 'patient_nbr'], axis=1, inplace=True)

# # Create readmitted binary label
# df['readmitted_30days'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
# df.drop('readmitted', axis=1, inplace=True)

# Handle missing values
categorical_cols = df.select_dtypes(include='object').columns
numeric_cols = df.select_dtypes(include=np.number).columns

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Handle gender outlier
df = df[df['gender'] != 'Unknown/Invalid']

# Drop patients that died during admission (discharge_disposition_id == 11)
#df = df[df['discharge_disposition_id'] != 11]

# Load IDS_mapping.csv and split it into 3 chunks
mapping_df = pd.read_csv("Data/IDS_mapping.csv", header=None)
df = df[df['discharge_disposition_id'] != 11]

def extract_mapping(df, start_col):
    start_idx = df[df[0] == start_col].index[0]
    try:
        end_idx = df.iloc[start_idx+1:].index[df.iloc[start_idx+1:][0].isna()].tolist()[0]
    except:
        end_idx = len(df)
    section = df.iloc[start_idx+1:end_idx].dropna()
    section.columns = [start_col, 'description']
    section[start_col] = section[start_col].astype(int)
    return section

# Replace ID columns with their descriptions
def map_and_replace(df, mapping_df, id_col, label):
    idx = df.columns.get_loc(id_col)
    map_df = extract_mapping(mapping_df, id_col)
    df = df.merge(map_df.rename(columns={'description': label}), on=id_col, how='left')
    df[label] = df[label].fillna("Unknown")
    df.drop(id_col, axis=1, inplace=True)
    cols = df.columns.tolist()
    label_col = cols.pop()
    cols.insert(idx, label_col)
    df = df[cols]
    return df

df = map_and_replace(df, mapping_df, 'admission_type_id', 'admission_type')
df = map_and_replace(df, mapping_df, 'discharge_disposition_id', 'discharge_disposition')
df = map_and_replace(df, mapping_df, 'admission_source_id', 'admission_source')


# Encode drug columns
drug_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
    'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
    'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton',
    'insulin', 'glyburide-metformin', 'glipizide-metformin',
    'glimepiride-pioglitazone', 'metformin-rosiglitazone',
    'metformin-pioglitazone']

med_map = {"No": 0, "Down": 1, "Steady": 2, "Up": 3}
for col in drug_cols:
    if col in df.columns:
        df[col] = df[col].map(med_map)

# Label encode some categorical fields
label_cols = ['gender', 'race', 'age', 'change', 'diabetesMed']
le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col])

# Save cleaned file
df.to_csv("Clean_data_for_gui.csv", index=False)
print("Cleaned data saved as 'Clean_data_for_gui.csv'")