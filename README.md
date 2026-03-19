# Diabetes-Readmission-Predictor

A machine learning-powered tool to predict whether a diabetic patient is likely to be readmitted to a hospital within 30 days, based on their clinical, demographic, and treatment information.


## Project Structure

```text
diabetes-readmission-predictor/
├── data/               # Raw and cleaned data
├── notebooks/          # Notebooks for EDA & modeling
├── models/             # Trained models and scalers
├── app/                # Streamlit GUI application
├── reports/            # Final report, plots, insights
├── requirements.txt    # Project dependencies
└── README.md           # Project overview
```
---

## Features

- 📊 Data upload & preview
- 🔍 Predict 30-day readmission
- 🧠 Model interpretation using SHAP
- 👥 Patient clustering visualization
- 📈 Feature importance charts
- 📋 Demographic & subgroup analysis

---

## 📦 Setup Instructions

```bash
# Clone the repo
git clone https://github.com/your-username/diabetes-readmission-predictor.git
cd diabetes-readmission-predictor

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app/main.py
