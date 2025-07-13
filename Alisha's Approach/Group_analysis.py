import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Read data
df = pd.read_csv("Data/Clean_data_for_train(1).csv")

print(df.groupby('race')['readmitted_30days'].mean())
print(df.groupby('gender')['readmitted_30days'].mean())
print(df.groupby('age')['readmitted_30days'].mean())

sns.barplot(x='race', y='readmitted_30days', data=df)
plt.title('Readmission Rate by Race')
plt.show()

sns.barplot(x='gender', y='readmitted_30days', data=df)
plt.title('Readmission Rate by Gender')
plt.show()

sns.barplot(x='age', y='readmitted_30days', data=df)
plt.title('Readmission Rate by Age')
plt.show()

med_cols = ['max_glu_serum', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide',
            'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 
            'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']

# Collect means
results = {}
for med in med_cols:
    means = df.groupby(med)['readmitted_30days'].mean()
    # Usually medication columns are binary (0/1)
    if 1 in means.index:
        results[med] = means[1]
    else:
        results[med] = None  # or 0 if you want

# Create DataFrame
med_effect = pd.DataFrame({
    'medication': results.keys(),
    'readmission_rate': results.values()
}).dropna()

print(med_effect)

plt.figure(figsize=(12, 8))
sns.barplot(x='medication', y='readmission_rate', data=med_effect, errorbar=None)
plt.xticks(rotation=90)
plt.title('Readmission Rate by Medication')
plt.tight_layout()
plt.show()

# Create a column combining medication usage as one (1 if any med taken, else 0)
df['any_med'] = df[med_cols].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)

# Group by age, race, gender, and medication usage
grouped = df.groupby(['age', 'race', 'gender', 'any_med'])['readmitted_30days'].mean().reset_index()

print(grouped.head())

# Barplot: Readmission rate by age and medication usage
plt.figure(figsize=(12, 6))
sns.barplot(data=grouped, x='age', y='readmitted_30days', hue='any_med')
plt.title('Readmission Rate by Age and Medication Usage')
plt.ylabel('Average Readmission Rate (30 days)')
plt.xlabel('Age Group')
plt.legend(title='Any Medication')
plt.show()

# Barplot: Readmission rate by gender and medication usage
plt.figure(figsize=(12, 6))
sns.barplot(data=grouped, x='gender', y='readmitted_30days', hue='any_med')
plt.title('Readmission Rate by Gender and Medication Usage')
plt.ylabel('Average Readmission Rate (30 days)')
plt.xlabel('Gender')
plt.legend(title='Any Medication')
plt.show()

# Barplot: Readmission rate by race and medication usage
plt.figure(figsize=(12, 6))
sns.barplot(data=grouped, x='race', y='readmitted_30days', hue='any_med')
plt.title('Readmission Rate by Race and Medication Usage')
plt.ylabel('Average Readmission Rate (30 days)')
plt.xlabel('Race')
plt.legend(title='Any Medication')
plt.show()