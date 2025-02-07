# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 11:01:26 2024

@author: zp20945
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot
from numpy import arange
from sklearn.metrics import r2_score

# import Find_lux as FL #This file is used to find luminosity for the given intensity of RGB between 0 to 100:
from final_model_pupil import Find_lux as FL


""" define function """
def objective(x, a, b, c, e): 
    '''
        curve fit formula to fit prediction of pupil size model.
        x: luminosity
        a,b,c,d : coefficients of the model
    '''
                                
    return a * np.exp(-b * x) + c * x + e

def calculate_ps_gen(lux_values, coefficients_file):
    '''
        predict the general pupil size based on the original pupil size model.
        lux_values:
        coefficients_file:
    '''
    
    # Load coefficients from CSV
    red_coeff = pd.read_csv(coefficients_file)

    # Extract coefficients
    a1 = red_coeff.at[0, 'a']  # Assuming 'a' is the column name
    b1 = red_coeff.at[0, 'b']  # Assuming 'b' is the column name
    c1 = red_coeff.at[0, 'c']  # Assuming 'c' is the column name
    e1 = red_coeff.at[0, 'e']  # Assuming 'e' is the column name

    # Calculate ps_gen values
    ps_gen = round(objective(lux_values, a1, b1, c1, e1), 2)

    return ps_gen, a1, b1, c1, e1


def get_filtered_values(df, R, G, B, column):
    filtered_df = df.loc[(df["R"] == R) & (df["G"] == G) & (df["B"] == B)]
    filtered_values = filtered_df[column].iloc[0]
    return filtered_values


def recalibration(lux_intensity, coefficients_file, file_path, part_file, color, points, column):
    '''
        Calibrate the original pupil size model based on participant's calibration data.
        lux_intensity:
        coefficients_file:
        file_path:
        part_file:
        color:
        points:
        column:
    '''
    
    
    
    if points == 0:
        cal_intensity = [0, 50, 100]
    else:
        cal_intensity = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    PS = []
    processed_lux = set()  # To track processed lux intensity values
    
    for i in range(len(cal_intensity) - 1):
        filtered_lux_intensity_between_cal = [value for value in lux_intensity if cal_intensity[i] <= value <= cal_intensity[i+1]]
        #print(filtered_lux_intensity_between_cal)

        if not filtered_lux_intensity_between_cal:
            continue
        
        if color == 'red':
            get_lux_value = lambda lux: FL.find_lux_from_file(lux, 0, 0, file_path)
            get_cal_value = lambda lux: get_filtered_values(part_file, lux, 0, 0, column)
        elif color == 'green':
            get_lux_value = lambda lux: FL.find_lux_from_file(0, lux, 0, file_path)
            get_cal_value = lambda lux: get_filtered_values(part_file, 0, lux, 0, column)
        elif color == 'blue':
            get_lux_value = lambda lux: FL.find_lux_from_file(0, 0, lux, file_path)
            get_cal_value = lambda lux: get_filtered_values(part_file, 0, 0, lux, column)
        elif color == 'white':
            
            get_lux_value = lambda lux: FL.find_lux_from_file(lux, lux, lux, file_path)
            get_cal_value = lambda lux: get_filtered_values(part_file, lux, lux, lux, column)
        else:
            raise ValueError("Invalid color. Choose either 'red' or 'green'.")
        
        ps_model_min, _, _, _, _ = calculate_ps_gen(get_lux_value(filtered_lux_intensity_between_cal[0]), coefficients_file)
        ps_model_max, _, _, _, _ = calculate_ps_gen(get_lux_value(filtered_lux_intensity_between_cal[-2]), coefficients_file)
        
        ps_cal_min = get_cal_value(filtered_lux_intensity_between_cal[0])  # Assuming single value
        #print(filtered_lux_intensity_between_cal[-1])
        ps_cal_max = get_cal_value(filtered_lux_intensity_between_cal[-1])  # Assuming single value
        
        Total_distance_model = ps_model_min - ps_model_max
        Total_distance_part = ps_cal_min - ps_cal_max
        
        for lux in filtered_lux_intensity_between_cal:
            if lux in processed_lux:
                continue  # Skip if lux intensity has already been processed
            processed_lux.add(lux)
            
            ps_current, _, _, _, _ = calculate_ps_gen(get_lux_value(lux), coefficients_file)
            distance = ps_model_min - ps_current
            ratio = distance / Total_distance_model if Total_distance_model != 0 else 1
            Rescaled_point = Total_distance_part * ratio
            PS.append(round((ps_cal_min - Rescaled_point), 4))
      
    return PS




def fit_model(lux , pupil_size):
    '''
        fit the model based on rescaled pupil size and luminosity.
        lux: list of luminosity values
        pupil_size: list of pupil size for the given list of luminosity
    
    '''
    
    popt1, _ = curve_fit(objective, lux, pupil_size, p0 = [1,0,0,1])  #p0 = [3.4,0.1,0.01,2.4]
    
    y_pred = objective(lux, *popt1)
    r_squared = r2_score(pupil_size, y_pred)

    a, b, c, e = popt1
    
    x_line = arange(0, max(lux), 1)
    plt.scatter(lux,pupil_size, color='blue')
    y_line = objective(x_line, a, b ,c, e)
    plt.plot(x_line, y_line, '--', color='black', label=f'Curve Fit (R^2={r_squared:.2f})')
    pyplot.title('blue model for participant 3')
    pyplot.xlabel('lux grid')
    pyplot.ylabel('Pupil size(mm)')
    plt.legend()
    pyplot.show()
    return popt1





def lux_function(intensity, file_path, color):
    ''' 
        find the value of luminosity for the given color:
        red, green, bleu, and grey.
        intensity: list of intensity values from 0 to 100 at which you want to predict the luminosity
        file_path: path of the file of the measured luminosity for possible intensities
        color: the name of color from red, green, blue, and white, which lux value you want to find        
        
    '''
    lux_temp = []
    for i in intensity:
        if color == 'red':
            lux = FL.find_lux_from_file(i, 0, 0, file_path)
        elif color == 'green':
            lux = FL.find_lux_from_file(0, i, 0, file_path)
        elif color == 'blue':
            lux = FL.find_lux_from_file(0, 0, i, file_path)
        elif color == 'white':
            lux = FL.find_lux_from_file(i, i, i, file_path)
        lux_temp.append(lux)
    return lux_temp







