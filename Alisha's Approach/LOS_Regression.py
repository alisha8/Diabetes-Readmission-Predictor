import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# Load your data
df = pd.read_csv('Data/Clean_data_for_train(1).csv')

# Assume your LOS column is called 'length_of_stay' (adjust if needed)
target = 'time_in_hospital'

# Drop target and readmission from features
features = df.drop(['readmitted_30days', target], axis=1).select_dtypes(include=[np.number])

X = features
y = df[target]

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()


####################################
# Linear Regression
lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred_lr = lr.predict(x_test)

print("\nLinear Regression:")
print("MAE:", mean_absolute_error(y_test, y_pred_lr))
print("RMSE:", root_mean_squared_error(y_test, y_pred_lr))

####################################
# Random Forest Regressor
rf = RandomForestRegressor(random_state=42)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)

print("\nRandom Forest Regressor:")
print("MAE:", mean_absolute_error(y_test, y_pred_rf))
print("RMSE:", root_mean_squared_error(y_test, y_pred_rf))

####################################
# XGBoost Regressor
xgb = XGBRegressor(random_state=42)
xgb.fit(x_train, y_train)
y_pred_xgb = xgb.predict(x_test)

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

# Initialize model
xgb = XGBRegressor(random_state=42)

# Grid search with 3-fold CV
grid = GridSearchCV(xgb, param_grid, scoring='neg_mean_absolute_error', cv=3, n_jobs=-1, verbose=1)
grid.fit(x_train, y_train)

# Best model
best_xgb = grid.best_estimator_
y_pred_best = best_xgb.predict(x_test)

# Evaluation
print("\nBest Parameters:", grid.best_params_)
print("MAE:", mean_absolute_error(y_test, y_pred_best))
print("RMSE:", root_mean_squared_error(y_test, y_pred_best))
