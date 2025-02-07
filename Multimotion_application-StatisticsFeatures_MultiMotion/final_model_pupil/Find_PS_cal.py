# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 16:02:10 2024

@author: zp20945
"""

import pandas as pd


def find_avg(data):
    
    """ Find the average pupil size for each stimuli after doing NAN imputation using median """
    # Create a copy of the DataFrame to avoid modifying the original data
    data_copy = data.copy()
    # Step 1: Convert the 'ET_PupilLeft' column to numeric, coercing errors to NaN
    data_copy['ET_PupilLeft'] = pd.to_numeric(data_copy['ET_PupilLeft'], errors='coerce')
    data_copy['ET_PupilRight'] = pd.to_numeric(data_copy['ET_PupilRight'], errors='coerce')
    # Convert the imputed data back to a DataFrame
    data_copy['ET_PupilLeft'] = data_copy['ET_PupilLeft'].fillna(data_copy['ET_PupilLeft'].median())
    data_copy['ET_PupilRight'] = data_copy['ET_PupilRight'].fillna(data_copy['ET_PupilRight'].median())
    
    # Calculate the average across rows
    average = data_copy[['ET_PupilLeft', 'ET_PupilRight']].mean(axis=1)
    return average



def find_ps_seconds(df, interval, sample_count):
    """ Find the 27 points at every 4 seconds from the calibration video """
    
    x = []  # Store the average values
    correlations = []  # Store the Pearson correlation values
    start_time = df['Timestamp'].min()
    
    for i in range(sample_count):
        start_ms = start_time + i * interval * 1000  # Convert interval to milliseconds
        end_ms = start_ms + interval * 1000  # Interval end in milliseconds
        
        # Create mask to select data within the interval
        mask = (df['Timestamp'] >= start_ms) & (df['Timestamp'] <= end_ms)
        interval_data = df.loc[mask]
        
        # Calculate the mean of the 'Average' column for the current interval
        x_data = interval_data['Average'].mean()
        x.append(x_data)
        
            
    return x, correlations


