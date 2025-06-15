
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

import seaborn as sns
import matplotlib.pyplot as plt

#............................Importing Data Set..............................#

df = pd.read_csv("diabetic_data.csv", header=0)
print(df.columns.tolist()) 
print(df.shape)
print(df.head())
print(df.info())
print(df.describe())
print(df['readmitted'].value_counts())
df['readmitted_30days'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)



# ...............................Data Cleaning...............................#

# Replace '?' with NaN
df.replace("?", np.nan, inplace=True)

# Drop columns with >50% missing values
missing_ratio = df.isnull().mean()
columns_to_drop = missing_ratio[missing_ratio > 0.5].index.tolist() # Convert to list

# Include additional irrelevant/sparse columns
additional_drops = [
    'encounter_id', 'patient_nbr', 'admission_type_id', 'payer_code',
    'discharge_disposition_id', 'admission_source_id', 'medical_specialty',
    'num_lab_procedures', 'num_procedures', 'number_outpatient',
    'number_emergency', 'number_inpatient', 'num_medications',
    'diag_1', 'diag_2', 'diag_3',
    
    # Removed combo meds (before filtering)
    'repaglinide', 'nateglinide', 'chlorpropamide', 'acetohexamide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
]

# Combine both sets
all_to_drop = list(set(columns_to_drop + additional_drops))

# Drop columns (safe drop with errors='ignore')
df.drop(columns=all_to_drop, axis=1, inplace=True, errors='ignore')

# filling in missing categorical data #

categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    df[col]= df[col].fillna(df[col].mode()[0])

# filling in missing numerical data #

numerical_cols = df.select_dtypes(include=np.number).columns
for col in numerical_cols:
    df[col]= df[col].fillna(df[col].median())

   

# ................................ Preprocessing................................#

# Label Encoding for basic categorical columns
label_cols = ['gender', 'race', 'age', 'change', 'diabetesMed']
le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col])

df['age_original'] = df['age'] # saving Age for visualization later
df['race_original'] = df['race'] # saving Race for visualization later
# One-hot encode the 'age' and 'race' columns
df = pd.get_dummies(df, columns=['age', 'race'], prefix=['age', 'race'])

# Ordinal Encoding for 'max_glu_serum' and 'A1Cresult'
glu_a1c_map = {
    'None': 0,
    'Norm': 1,
    '>200': 2,
    '>300': 3
}


# Encoding MAx_Glu_Serum and A1Cresults, if still retained
if 'max_glu_serum' in df.columns:
    df['max_glu_serum'] = df['max_glu_serum'].map(glu_a1c_map)

if 'A1Cresult' in df.columns:
    df['A1Cresult'] = df['A1Cresult'].map(glu_a1c_map)


# Encoding Meds -- Mrtformin, Glimepiride, Glipizide
retained_meds = [ 'metformin', 'glimepiride', 'glipizide']

med_map = {"No": 0, "Down": 1, "Steady": 2, "Up": 3}

for col in retained_meds:
    df[col] = df[col].fillna("No").map(med_map)
    print(df[retained_meds].head())
    print("Encoded medication columns retained:", retained_meds)

if 'insulin' in df.columns:
    df['insulin'] = df['insulin'].map(med_map)

df.to_csv("cleaned_data.csv", index=False)


# Possible code to remove all medication colums that have a certain percent of "No" entries
# Doesent make much sense ... most entries are No, in most colums, aprox 99.9%
"""
medication_cols = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
    'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
    'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton',
]

df[medication_cols] = df[medication_cols].replace("No", np.nan)
missing_ratio = df[medication_cols].isnull().mean()

# Drop medication columns with > 1% missing values
retained_meds = missing_ratio[missing_ratio < 99 ].index.tolist()
cols_to_drop = [col for col in medication_cols if col not in retained_meds]
df.drop(columns=cols_to_drop, inplace=True)

med_map = {"No": 0, "Down": 1, "Steady": 2, "Up": 3}

for col in retained_meds:
    df[col] = df[col].fillna("No").map(med_map)
    print(df[retained_meds].head())
    print("Encoded medication columns retained:", retained_meds)


if 'insulin' in df.columns:
    df['insulin'] = df['insulin'].map(med_map)

df.to_csv("cleaned_data.csv", index=False)
"""  




#..............................Plotting Count Plots............................#

sns.set(style="whitegrid")
sns.countplot(x='readmitted_30days', data=df)
plt.title("Readmitted within 30 Days")
plt.xlabel("Readmitted (1 = Yes, 0 = No)")
plt.ylabel("Number of Patients")
plt.show()

print(df.columns.tolist())
sns.countplot(x='age_original', hue='readmitted', data=df)
plt.title("Age Group vs. Readmission")
plt.xlabel("Age Group")
plt.ylabel("Number of Patients")
plt.xticks(rotation=45)
plt.legend(title='Readmitted (<30 days)')
plt.show()

sns.countplot(x='gender', hue='readmitted', data=df)
plt.title("Gender vs. Readmission")
plt.legend(title="Readmitted")
plt.show()

sns.countplot(x='race_original', hue='readmitted', data=df)
plt.xticks(rotation=45)  
plt.tight_layout()       
plt.show()

sns.countplot(x='insulin', hue='readmitted', data=df)
plt.xlabel("Insulin Category")
plt.show()


for med in retained_meds:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=med, hue='readmitted_30days', data=df)
    plt.title(f"{med.capitalize()} vs. Readmission")
    plt.xlabel(f"{med.capitalize()} Category")
    plt.ylabel("Count")
    plt.legend(title="Readmitted in 30 Days")
    plt.tight_layout()
plt.show()

df['time_in_hospital']
sns.boxplot(x='readmitted_30days', y='time_in_hospital', data=df)
plt.title("Time in Hospital vs. Readmission")
plt.xlabel("Readmitted")
plt.ylabel("Days in Hospital")
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()




#........................principal component analysis..........................#


# Select only numeric features
numeric_df = df.select_dtypes(include=['int64', 'float64'])

# Fill missing values (mean strategy is common)
imputer = SimpleImputer(strategy='mean')
imputed_data = imputer.fit_transform(numeric_df)

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(imputed_data)

# Apply PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

# Create a PCA result dataframe
df_pca = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
df_pca['readmitted_30days'] = df['readmitted_30days'].values  # Assuming this column still exists

# Plot
plt.figure(figsize=(10,6))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='readmitted_30days', alpha=0.5)
plt.title("PCA Projection of Patients")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Readmitted (<30)")
plt.show()


#df.to_excel("C:\Users\grhdu\OneDrive\Documents\Visual Studio\Diabetes-Readmission-Predictor/cleaned_diabetes_data.xlsx", index=False)
df.to_csv("C:\\Users\\grhdu\\Downloads\\cleaned_diabetes_data.csv", index=False)
print("Cleaned data saved to cleaned_diabetes_data.csv")
