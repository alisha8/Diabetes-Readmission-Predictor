import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
import io
import sys

def load_data():
    df = pd.read_csv('/Users/alishasarkar/Documents/Python Lab/Diabetes_new/Diabetes-Readmission-Predictor/Alisha_Approach/Data/clean/Clean_data_for_train(1).csv')
    target = 'time_in_hospital'
    features = df.drop(['readmitted_30days', target], axis=1).select_dtypes(include=[np.number])
    X = features
    y = df[target]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train.values.ravel(), y_test.values.ravel()

def run_linear_regression():
    x_train, x_test, y_train, y_test = load_data()
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    md_output = f"""
                    Linear Regression

                    Mean Absolute Error (MAE): {mae:.4f}
                    Root Mean Squared Error (RMSE): {rmse:.4f}

                    - Assumes a linear relationship between features and hospital stay duration.

"""
    return md_output

def run_random_forest():
    x_train, x_test, y_train, y_test = load_data()
    rf = RandomForestRegressor(random_state=42)
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    md_output = f"""
                    Random Forest Regression

                    Mean Absolute Error (MAE): {mae:.4f}
                    Root Mean Squared Error (RMSE): {rmse:.4f}

                    - Uses an ensemble of decision trees for prediction.

"""
    return md_output

def run_xgboost():
    x_train, x_test, y_train, y_test = load_data()
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }
    xgb = XGBRegressor(random_state=42)
    grid = GridSearchCV(xgb, param_grid, scoring='neg_mean_absolute_error', cv=3, n_jobs=-1, verbose=0)
    grid.fit(x_train, y_train)
    best_xgb = grid.best_estimator_
    y_pred = best_xgb.predict(x_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    md_output = f"""
                    XGBoost Regression

                    Best Hyperparameters: {grid.best_params_}
                    Mean Absolute Error (MAE): {mae:.4f}
                    Root Mean Squared Error (RMSE): {rmse:.4f}

                    - Uses gradient-boosted trees optimized via GridSearchCV.

"""
    return md_output