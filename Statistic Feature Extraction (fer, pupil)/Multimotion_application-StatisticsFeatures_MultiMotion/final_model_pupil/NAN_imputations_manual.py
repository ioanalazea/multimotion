# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 13:27:53 2024

@author: zeelp
"""

import numpy as np
import pandas as pd
def process_none_data(df, column_left, column_right, average_column_name='Average'):
    """
    Process data in specified columns by handling NaN and specific values, 
    calculating differences, and computing an average column based on conditions.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the columns to process.
    - column_left (str): The name of the "left" column to process.
    - column_right (str): The name of the "right" column to process.
    - average_column_name (str): Name of the column to store computed averages. Defaults to 'Average'.

    Returns:
    - pd.DataFrame: Processed DataFrame with additional columns:
      'Prev_<column_left>', 'Left_Diff_Next_Prev', 'Prev_<column_right>', 'Right_Diff_Next_Prev',
      'difference', and the specified 'average_column_name' column.
    """
    """
    
    # Drop rows where both specified columns are NaN
    df = df.dropna(subset=[column_left, column_right]).copy()  # Ensure a copy to avoid warnings
    
    # Replace -1 values with NaN in the specified columns
    df[[column_left, column_right]] = df[[column_left, column_right]].replace(-1, np.nan)
    
    # Set both columns to NaN if either one is NaN
    df.loc[df[column_left].isna(), column_right] = np.nan
    df.loc[df[column_right].isna(), column_left] = np.nan
    """


    # Shift indices and calculate previous values
    df[f'Prev_{column_left}'] = df[column_left].shift(1)
    df[f'Prev_{column_right}'] = df[column_right].shift(1)
    
    # Calculate differences
    df['Left_Diff_Next_Prev'] = abs(df[column_left] - df[f'Prev_{column_left}'])
    df['Right_Diff_Next_Prev'] = abs(df[column_right] - df[f'Prev_{column_right}'])
    
    # Calculate the ratio of differences
    df['difference'] = df['Left_Diff_Next_Prev'] / df['Right_Diff_Next_Prev']

    # Create and populate the average column based on conditions for 'difference'
    conditions = (df['difference'] < 0.1) | (df['difference'] > 10)
    
    # Use np.where to compute the average column efficiently
    df[average_column_name] = np.where(
        conditions,
        np.where(df['Left_Diff_Next_Prev'] < df['Right_Diff_Next_Prev'], df[column_left], df[column_right]),
        df[[column_left, column_right]].mean(axis=1)
    )
    
    return df

def process_column_data(df, column_left, column_right):
    # Find indices where either 'ET_PupilLeft' or 'ET_PupilRight' is NaN
    nan_indices = df[df[column_left].isna() | df[column_right].isna()].index

    # Extend rows_to_remove to include two indices before and two indices after each NaN index
    rows_to_remove = np.concatenate([
        nan_indices - 2,  # Two rows before
        nan_indices - 1,  # One row before
        nan_indices,      # The NaN rows themselves
        nan_indices + 1,  # One row after
        nan_indices + 2   # Two rows after
    ])
    
    # Filter out any indices that are out of bounds
    rows_to_remove = rows_to_remove[(rows_to_remove >= 0) & (rows_to_remove < len(df))]

    # Remove duplicates to avoid redundant operations on the same rows
    rows_to_remove = np.unique(rows_to_remove)

    # Set the rows to NaN for all columns (or specific columns) at those indices
    df.loc[rows_to_remove] = np.nan  # Sets all columns in those rows to NaN

    # Example continuation of processing as needed (this part of your function remains as it is)
    # Calculating previous row values and differences for each specified column...
    # Further processing steps can be added as required

    return df

#def process_column_data(df, column_left, column_right, average_column_name='Average'):
    """
    Process data in specified columns by handling NaN and specific values, 
    calculating differences, and computing an average column based on conditions.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the columns to process.
    - column_left (str): The name of the "left" column to process.
    - column_right (str): The name of the "right" column to process.
    - average_column_name (str): Name of the column to store computed averages. Defaults to 'Average'.

    Returns:
    - pd.DataFrame: Processed DataFrame with additional columns:
      'Prev_<column_left>', 'Left_Diff_Next_Prev', 'Prev_<column_right>', 'Right_Diff_Next_Prev',
      'difference', and the specified 'average_column_name' column.
    """
    """
    # Drop rows where both specified columns are NaN
    df = df.dropna(subset=[column_left, column_right])
    
    # Replace -1 values with NaN in the specified columns
    df[[column_left, column_right]] = df[[column_left, column_right]].replace(-1, np.nan)
    
    # Set both columns to NaN if either one is NaN
    df.loc[df[column_left].isna(), column_right] = np.nan
    df.loc[df[column_right].isna(), column_left] = np.nan
    
    # Calculate previous row values and differences for each specified column
    df[f'Prev_{column_left}'] = df[column_left].shift(1)
    df['Left_Diff_Next_Prev'] = abs(df[column_left] - df[f'Prev_{column_left}'])
    
    df[f'Prev_{column_right}'] = df[column_right].shift(1)
    df['Right_Diff_Next_Prev'] = abs(df[column_right] - df[f'Prev_{column_right}'])
    
    # Calculate the ratio of differences between the columns
    df['difference'] = df['Left_Diff_Next_Prev'] / df['Right_Diff_Next_Prev']
    
    # Create and populate the average column based on conditions for 'difference'
    df[average_column_name] = None  # Initialize the average column
    
    for idx, row in df.iterrows():
        if row['difference'] < 0.1 or row['difference'] > 10:
            # Select the value with the smaller difference
            if row['Left_Diff_Next_Prev'] < row['Right_Diff_Next_Prev']:
                df.at[idx, average_column_name] = row[column_left]
            else:
                df.at[idx, average_column_name] = row[column_right]
        else:
            # If within threshold, calculate the mean of the left and right columns
            df.at[idx, average_column_name] = row[[column_left, column_right]].mean()
    
    return df
    """
# Function to perform gradual interpolation based on values before and after NaNs
def interpolate_with_constant_fill(df, column_name):
    """
    Interpolates NaN values in a column by filling continuous NaN segments with 
    the first interpolated value calculated between the values immediately before and after each NaN segment.

    Parameters:
    - df: DataFrame containing the column.
    - column_name: Name of the column to be interpolated.

    Returns:
    - DataFrame with interpolated values for NaNs in specified column, using a constant fill.
    """
    # Ensure integer-based indexing by resetting the index
    df = df.reset_index(drop=True)
    
    # Iterate through the column to identify NaN segments and interpolate them
    for idx in range(1, len(df) - 1):
        # If the current value is NaN
        if pd.isna(df.loc[idx, column_name]):
            # Find the previous non-NaN value
            prev_idx = idx - 1
            while prev_idx >= 0 and pd.isna(df.loc[prev_idx, column_name]):
                prev_idx -= 1
            
            # Find the next non-NaN value
            next_idx = idx + 1
            while next_idx < len(df) and pd.isna(df.loc[next_idx, column_name]):
                next_idx += 1
            
            # If we have both previous and next values for interpolation
            if prev_idx >= 0 and next_idx < len(df):
                prev_value = df.loc[prev_idx, column_name]
                next_value = df.loc[next_idx, column_name]
                
                # Calculate the constant fill value (first interpolated value)
                constant_fill_value = (prev_value + next_value) / 2
                
                # Fill each NaN in the segment with the constant interpolated value
                for fill_idx in range(prev_idx + 1, next_idx):
                    df.loc[fill_idx, column_name] = constant_fill_value

    return df