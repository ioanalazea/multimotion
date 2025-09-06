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
from scipy.optimize import minimize


def constrained_linear_regression(X, y):
    """
    Perform linear regression with non-negative coefficients using constrained optimization.
    
    Parameters:
    X (np.ndarray): Feature matrix.
    y (np.ndarray): Target values.
    
    Returns:
    np.ndarray: Coefficients of the regression model (all positive).
    float: Intercept of the regression model.
    """
    def objective(coef_intercept):
        # The objective is to minimize the residual sum of squares
        coef = coef_intercept[:-1]
        intercept = coef_intercept[-1]
        predictions = np.dot(X, coef) + intercept
        residuals = y - predictions
        return np.sum(residuals**2)

    def constraint(coef_intercept):
        # This ensures that the coefficients are non-negative
        return coef_intercept[:-1]

    # Initial guess: starting with zero coefficients and intercept
    initial_guess = np.zeros(X.shape[1] + 1)
    
    # Define bounds: coefficients must be >= 0, intercept is unconstrained
    bounds = [(0, None)] * X.shape[1] + [(-np.inf, np.inf)]
    
    # Perform constrained optimization to find the coefficients and intercept
    result = minimize(objective, initial_guess, constraints={'type': 'ineq', 'fun': constraint}, bounds=bounds)
    
    # Extract the coefficients and intercept from the result
    coef = result.x[:-1]
    intercept = result.x[-1]
    
    return coef, intercept

#only use the positive coefficients
def linear_regression_with_initial_coef_no_C(X, y, initial_coef, initial_K=1.0):
    """
    Perform constrained linear regression with a scaling factor K and sum of coefficients equal to 1.
    
    Parameters:
    X (np.ndarray): Feature matrix.
    y (np.ndarray): Target variable.
    initial_coef (np.ndarray): Initial values for coefficients (a1, a2, a3, a4).
    initial_K (float): Initial value for K.

    Returns:
    tuple: Optimized K, coefficients (a1, a2, a3, a4), and intercept.
    """
    # Add intercept column (bias term)
    X_with_intercept = np.column_stack((np.ones(X.shape[0]), X))

    # Define the objective function (least squares error)
    def objective_function(beta):
        K = beta[0]  # Scaling factor
        coefficients = beta[1:]  # a1, a2, a3, a4
        predictions = K * (X_with_intercept[:, 1:] @ coefficients)  # Apply equation
        return np.sum((y - predictions) ** 2)  # Minimize squared error

    # Constraint: a1 + a2 + a3 + a4 = 1
    def coef_constraint(beta):
        return np.sum(beta[1:]) - 1  # Must be zero

    # Initial values: K + coefficients
    initial_beta = np.insert(initial_coef, 0, initial_K)  # Adding K as the first element

    # Define constraints
    constraints = {'type': 'eq', 'fun': coef_constraint}

    # Optimize using BFGS with constraints
    result = minimize(objective_function, initial_beta, method='SLSQP', constraints=constraints)

    # Extract optimized values
    optimized_K = result.x[0]
    optimized_coef = result.x[1:]

    return optimized_K, optimized_coef


def normal_regression_LOO_2(selected_features, measured_ps):
    """
    Perform linear regression with non-negative coefficients on the selected features and measured_ps using leave-one-out cross-validation.

    Parameters:
    selected_features (list of np.ndarray): List of arrays representing the feature columns.
    measured_ps (np.ndarray): Array of measured target values.

    Returns:
    tuple: A tuple containing the predicted values, average percentage error, coefficients, and intercept.
    """
    y = measured_ps

    # Combine the selected features into a single 2D array
    X = np.column_stack(selected_features)

    loo = LeaveOneOut()
    y_true = []
    y_pred = []

    coefficients_list = []  # To store the coefficients for each fold
    intercepts_list = []  # To store the intercept for each fold

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Apply the constrained linear regression
        coef, intercept = constrained_linear_regression(X_train, y_train)
        
        # Make prediction for the test data
        prediction = np.dot(X_test, coef) + intercept

        y_true.append(y_test.item())
        y_pred.append(prediction.item())

        # Store coefficients and intercept for this fold
        coefficients_list.append(coef)
        intercepts_list.append(intercept)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    avg_percentage_error = calculate_average_percentage_error(y_true, y_pred)

    avg_coefficients = np.mean(coefficients_list, axis=0)
    avg_intercept = np.mean(intercepts_list)

    return y_pred, avg_percentage_error, avg_coefficients, avg_intercept
# Regression model



def linear_regression_with_initial_coef(X, y, initial_coef, initial_K=1.0, initial_C=0.0):
    """
    Perform constrained linear regression with a scaling factor K, an intercept C,
    and a1 + a2 + a3 + a4 = 1.
    
    Parameters:
    X (np.ndarray): Feature matrix.
    y (np.ndarray): Target variable.
    initial_coef (np.ndarray): Initial values for coefficients (a1, a2, a3, a4).
    initial_K (float): Initial value for K.
    initial_C (float): Initial value for intercept C.

    Returns:
    tuple: Optimized K, coefficients (a1, a2, a3, a4), and intercept C.
    """
    
    # Define the objective function (least squares error)
    def objective_function(beta):
        K = beta[0]  # Scaling factor
        coefficients = beta[1:5]  # a1, a2, a3, a4
        C = beta[5]  # Intercept
        predictions = K * (X @ coefficients) + C  # Apply equation
        #predictions =  (X @ coefficients) + C  # Apply equation
        
        return np.sum((y - predictions) ** 2)  # Minimize squared error

    # Constraint: a1 + a2 + a3 + a4 = 1
    def coef_constraint(beta):
        return np.sum(beta[1:5]) - 1  # Must be zero

    # Bounds for coefficients: C is constrained between -1 and 1
    #bounds = [(None, None)] * 5 + [(-1, 1)]
    
    # Initial values: [K, a1, a2, a3, a4, C]
    initial_beta = np.concatenate(([initial_K], initial_coef, [initial_C]))

    # Define constraints
    constraints = {'type': 'eq', 'fun': coef_constraint}

    # Optimize using SLSQP (supports equality constraints and bounds)
    result = minimize(objective_function, initial_beta, method='SLSQP', constraints=constraints) #bounds=bounds)

    # Extract optimized values
    optimized_K = result.x[0]
    optimized_coef = result.x[1:5]
    optimized_C = result.x[5]

    return optimized_K, optimized_coef, optimized_C


def normal_regression_initial_coef(selected_features, measured_ps):
    """
    Perform linear regression with initial coefficients and a scaling factor K 
    on the selected features and measured_ps.

    Parameters:
    selected_features (list of np.ndarray): List of arrays representing the feature columns.
    measured_ps (np.ndarray): Array of measured target values.

    Returns:
    tuple: A tuple containing the predicted values, average percentage error, 
           coefficients, intercept, and scaling factor K.
    """
    y = measured_ps
    X = np.column_stack(selected_features)  # Combine features into a 2D array

    # Given initial coefficients
    initial_coef = np.array([0.5, 0.165, 0.165, 0.165])  # a1, a2, a3, a4
    initial_K = 1.0  # Initial scaling factor

    # Apply linear regression with initial coefficients and scaling factor
    optimized_K, optimized_coef, coeff_c = linear_regression_with_initial_coef(X, y, initial_coef, initial_K, 0)
    #optimized_K, optimized_coef = linear_regression_with_initial_coef(X, y, initial_coef, initial_K, 0)

    # Make predictions
    y_pred = optimized_K * ((X @ optimized_coef) ) + coeff_c

    # Calculate the average percentage error
    avg_percentage_error = calculate_average_percentage_error(y, y_pred)

    return y_pred, avg_percentage_error, optimized_coef, optimized_K, coeff_c

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

def normal_regression_LOO_1(selected_features, measured_ps):
    """
    Perform linear regression on the selected features and measured_ps.

    Parameters:
    selected_features (list of np.ndarray): List of arrays representing the feature columns.
    measured_ps (np.ndarray): Array of measured target values.

    Returns:
    tuple: A tuple containing the predicted values, average percentage error, coefficients, and intercept.
    """
    y = measured_ps

    # Combine the selected features into a single 2D array
    X = np.column_stack(selected_features)

    loo = LeaveOneOut()
    y_true = []
    y_pred = []

    coefficients_list = []  # To store the coefficients for each fold
    intercepts_list = []  # To store the intercept for each fold

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = LinearRegression()
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)

        y_true.append(y_test.item())
        y_pred.append(prediction.item())

        # Store coefficients and intercept for this fold
        coefficients_list.append(model.coef_)
        intercepts_list.append(model.intercept_)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    avg_percentage_error = calculate_average_percentage_error(y_true, y_pred)

    avg_coefficients = np.mean(coefficients_list, axis=0)
    avg_intercept = np.mean(intercepts_list)

    return y_pred, avg_percentage_error, avg_coefficients, avg_intercept

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
    return y_pred, avg_percentage_error, model

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

