from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)


print(tf.__version__)


diamonds = fetch_openml('diamonds', version=1, as_frame=False)


x_train, x_test, y_train, y_test = train_test_split(
    diamonds["data"], diamonds["target"], shuffle=True)

# Define Grid
grid = {
    'n_estimators': [200, 300, 400, 500],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [3, 4, 5, 6, 7],
    'random_state': [18]
}  # show start time
print(datetime.now())  # Grid Search function
CV_rfr = GridSearchCV(estimator=RandomForestRegressor(),
                      param_grid=grid, cv=5, verbose=10, n_jobs=-1)
CV_rfr.fit(x_train, y_train)  # show end time
print(datetime.now())
print("\n The best parameters across ALL searched params:\n", CV_rfr.best_params_)
#  {'max_depth': 7, 'max_features': 'sqrt', 'n_estimators': 300, 'random_state': 18}

# [CV 4/5; 40/40] END max_depth=7, max_features=log2, n_estimators=500, random_state=18;, score=0.954 total time=   8.5s

# rf = RandomForestRegressor(
#     n_estimators=300, max_features='sqrt', max_depth=5, random_state=18)

# rf.fit(x_train, y_train)

prediction = CV_rfr.predict(x_test)
print(prediction)
print(y_test)
mse = mean_squared_error(y_test, prediction)
rmse = mse**.5
print(mse)
print(rmse)
