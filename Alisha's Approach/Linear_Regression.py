import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Read data
x_train = pd.read_csv('Data/X_train.csv')
x_test = pd.read_csv('Data/X_test.csv')
y_train = pd.read_csv('Data/Y_train.csv')
y_test = pd.read_csv('Data/Y_test.csv')

# 1. Define features and target
features = ['age', 'num_lab_procedures', 'time_in_hospital']  # adjust as needed
X = x_train[features]
y = df['LOS']

# Train Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train_los)

y_pred_lr = lin_reg.predict(x_test)
print("MAE LR:", mean_absolute_error(y_test_los, y_pred_lr))
print("RMSE LR:", np.sqrt(mean_squared_error(y_test_los, y_pred_lr)))

# Train RF Regressor
rf_reg = RandomForestRegressor(random_state=42)
rf_reg.fit(x_train, y_train_los)

y_pred_rf = rf_reg.predict(x_test)
print("MAE RF:", mean_absolute_error(y_test_los, y_pred_rf))
print("RMSE RF:", np.sqrt(mean_squared_error(y_test_los, y_pred_rf)))