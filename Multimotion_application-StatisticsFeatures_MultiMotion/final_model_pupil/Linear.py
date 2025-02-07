# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 10:51:40 2024

@author: zp20945
"""

import pandas as pd
import matplotlib.pyplot as plt
#import grandavgfun
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot
from numpy import arange
from sklearn.metrics import r2_score
from final_model_pupil import Find_lux as FL
from sklearn.model_selection import LeaveOneOut

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from itertools import product
from sklearn.model_selection import cross_val_predict


# Regression model

""" Normal regression """

def normal_regression(selected_features, measured_ps):
    """
    Perform linear regression on the selected features and measured_ps.

    Parameters:
    selected_features (list of np.ndarray): List of arrays representing the feature columns.
    measured_ps (np.ndarray): Array of measured target values.

    Returns:
    tuple: A tuple containing the predicted values, average percentage error, coefficients, and intercept.
    """
    measured_ps = measured_ps.reset_index(drop=True)
    
    # Combine the selected features into a single 2D array
    X = np.column_stack(selected_features)
    
    # Initialize the linear regression model
    model = LinearRegression()
    
    # Fit the model
    model.fit(X, measured_ps)
    
    # Get the coefficients and intercept
    coefficients = model.coef_
    intercept = model.intercept_
    
    # Make predictions
    #y_pred = model.predict(X)
    
    # Calculate mean squared error
    #mse = mean_squared_error(measured_ps, y_pred)
    
    # Calculate the average percentage error
    #diff = np.abs((y_pred - measured_ps) / measured_ps) * 100
    #avg_percentage_error = diff.mean()
    
    # Return the results
    return coefficients, intercept, model

""" leave one out linear regression """

def normal_regression_LOO(selected_features, measured_ps):
    """
    Perform linear regression on the selected features and measured_ps.

    Parameters:
    selected_features (list of np.ndarray): List of arrays representing the feature columns.
    measured_ps (np.ndarray): Array of measured target values.

    Returns:
    tuple: A tuple containing the predicted values, average percentage error, coefficients, and intercept.
    """
    #y = measured_ps.reset_index(drop=True)
    y = measured_ps
    #errors = []
    
    # Combine the selected features into a single 2D array
    X = np.column_stack(selected_features)
    
    #avg_percentage_error, error, y_pred, model = perform_loocv(X, measured_ps)
    
    loo = LeaveOneOut()
    y_true = []
    y_pred = []

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = LinearRegression()
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)

        y_true.append(y_test.item())
        y_pred.append(prediction.item())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    avg_percentage_error = calculate_average_percentage_error(y_true, y_pred)
    
    #avg_percentage_error, error = calculate_average_percentage_error(y_true, y_pred)
    
    
    
    # Return the results
    return y_pred, avg_percentage_error

""" Regression of multiple combinations """

def calculate_average_percentage_error(y_true, y_pred):
    """Calculate the average percentage error between true and predicted values."""
    diff = np.abs((y_pred - y_true) / y_true) * 100
    
    #diff_er = ((y_true -y_pred ))
    return diff.mean()

def perform_loocv(X, y):
    """Perform Leave-One-Out Cross-Validation (LOOCV) and return average percentage error."""
    loo = LeaveOneOut()
    y_true = []
    y_pred = []

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = LinearRegression()
        pred_ps_lnr = cross_val_predict(model, X, y, cv=10)  
        #model.fit(X_train, y_train)
        prediction = model.predict(X_test)

        y_true.append(y_test.item())
        y_pred.append(prediction.item())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    AVG_error, error = calculate_average_percentage_error(y_true, y_pred)

    return AVG_error, error, y_pred, model

def evaluate_feature_combinations(fixed_features, variable_features, measured_ps):
    """Evaluate different combinations of features and return average percentage errors."""
    variable_combinations = list(product([0, 1], repeat=len(variable_features)))
    errors = []

    for combination in variable_combinations:
        selected_features = fixed_features + [feature for feature, include in zip(variable_features, combination) if include]

        if not selected_features:
            continue

        X = np.column_stack(selected_features)
        avg_percentage_error = perform_loocv(X, measured_ps)
        errors.append(avg_percentage_error)

    return errors

def evaluate_feature_combinations_with_fixed_zero(fixed_features, variable_features, measured_ps):
    """Evaluate different combinations of features with fixed features set to zero."""
    variable_combinations = list(product([0, 1], repeat=len(variable_features)))
    errors = []

    for combination in variable_combinations:
        selected_features = [feature for feature, include in zip(variable_features, combination) if include]

        if not selected_features:
            continue

        X = np.zeros((len(measured_ps), len(fixed_features) + len(selected_features)))
        if selected_features:
            X[:, len(fixed_features):] = np.column_stack(selected_features)

        avg_percentage_error = perform_loocv(X, measured_ps)
        errors.append(avg_percentage_error)

    return errors

# Example usage
# Define your features and measured values
# r_ps, g_ps, b_ps, x1, x2, x3 = ... # Your actual feature data arrays
# measured_ps = ... # Your target values

# diff_1 = evaluate_feature_combinations([r_ps, g_ps, b_ps], [x1, x2, x3], measured_ps)
# diff_2 = evaluate_feature_combinations_with_fixed_zero([r_ps, g_ps, b_ps], [x1, x2, x3], measured_ps)

    
     
"""        
# Combine x1, x2, x3 into a single 2D array
X = np.column_stack((r_ps, g_ps, b_ps, x1, x2, x3))  # Note: x2 is not included
#X = np.column_stack((x1, x2, x3))  # Note: x2 is not included
# Initialize the LOOCV method
loo = LeaveOneOut()

# Initialize lists to store the results
y_true = []
y_pred = []
coefficients = []
intercepts = []

# Perform LOOCV
for train_index, test_index in loo.split(X):
    # Split the data
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = measured_ps[train_index], measured_ps[test_index]
    
    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make prediction
    prediction = model.predict(X_test)
    
    # Store the results
    y_true.append(y_test)
    y_pred.append(prediction)
    
    # Store the coefficients and intercept
    coefficients.append(model.coef_)
    intercepts.append(model.intercept_)

# Convert lists to arrays for easier handling
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Calculate mean squared error
mse = mean_squared_error(y_true, y_pred)

# Calculate average percentage error
diff = abs((y_pred - y_true) / y_true) * 100
avg_percentage_error = diff.mean()
print(avg_percentage_error)
# Print the results
#print(f"Mean Squared Error: {mse}")
#print(f"Average Percentage Error: {avg_percentage_error}")

# Optional: Create a DataFrame to compare actual vs predicted values
#results_df = pd.DataFrame({
#    'Actual': y_true,
#    'Predicted': y_pred
#})

#print(results_df)
"""