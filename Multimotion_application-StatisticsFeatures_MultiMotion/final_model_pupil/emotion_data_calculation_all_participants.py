# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 16:54:04 2024

@author: zp20945
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


from sklearn.preprocessing import MinMaxScaler


from final_model_pupil import Normal_RGB_model as RGB_N
from final_model_pupil import Linear as LN
from final_model_pupil import Find_PS_cal as ps_cal
from final_model_pupil import set_ps_at_100_lux as STL
from final_model_pupil import Single_video_file_processor_extract_RGB as VEX
from final_model_pupil import predict_ps_videoframe as PPS
from final_model_pupil import NAN_imputations_manual as NI

import os
import gc
import psutil

# Function to clear memory
def clear_memory():
    """
    Clears memory by triggering garbage collection and prints memory usage before and after.
    """
    
    # Print current memory usage before clearing
    memory_before = psutil.virtual_memory().percent
    print(f"Memory usage before clearing: {memory_before}%")
    
    # Run garbage collection
    gc.collect()
    
    # Print current memory usage after clearing
    memory_after = psutil.virtual_memory().percent
    print(f"Memory usage after clearing: {memory_after}%")
    
    # Provide feedback on garbage collection
    if memory_before > memory_after:
        print("Garbage collection reduced memory usage.")
    else:
        print("No significant change in memory usage.")
    
    # Collect garbage
    gc.collect()


def find_and_set_header(df, start_index=31, expected_header_indicator='Expected Header Indicator'):
    """
    Locates and sets the header row in a DataFrame, and extracts respondent name.
    """

    header = None
    for i in range(start_index, len(df)):
        row = df.iloc[i]
        #take the respondent name from the file
        respondent_name = df.iloc[1,1]
        
        if row[0] == expected_header_indicator:  # Check the first cell
            header = row
            df.columns = header  # Set this row as the header
            df = df[i+1:].reset_index(drop=True)  # Slice the DataFrame from the next row
            break
    return df, header, respondent_name

def process_csv_files(relative_path, output_path):
    """
    Main processing function that:
    1. Processes eye tracking data for each participant
    2. Calculates pupil size calibration using RGB color model
    3. Analyzes video stimuli and predicts pupil sizes
    4. Generates error metrics and scaled values
    """

    # Clear memory at the start of your script
    clear_memory()

    # Get the user's home directory ---- this to be changes rep.txt
    home_dir=os.getcwd() + "/final_model_pupil"

    # files path
    files_path =  os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'Files')  

    all_pssobilities_lux_v4 = os.path.join(files_path, 'required_files/all_pssobilities_lux_v4.csv')
    coefficients_file_r = os.path.join(files_path, 'required_files/red_coefficients.csv')
    coefficients_file_b = os.path.join(files_path, 'required_files/blue_coefficients.csv')
    coefficients_file_g = os.path.join(files_path, 'required_files/green_coefficients.csv')
    coefficients_file_w = os.path.join(files_path, 'required_files/grey_coefficients.csv')
    file_with_rgb_file_w = os.path.join(files_path, 'required_files/file_with_RGB_values.csv')

    print(coefficients_file_r)

    """ column name for participants """
    part_name = "Part_2"

    # Step 2: Get the list of video file names from the foldersurvey_stimuli
    video_folder = os.path.join(files_path, 'survey_stimuli') 
    print(os.listdir(video_folder))
    video_files = [os.path.splitext(f)[0] for f in os.listdir(video_folder) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    for filename in os.listdir(relative_path):
        if filename.endswith('.csv'):  # Check if the file is a CSV file
            filepath = os.path.join(relative_path, filename)

            # Read the CSV file
            df = pd.read_csv(filepath, encoding='ISO-8859-1')
            # Adjust header and respondent name
            df, header, respondent_name = find_and_set_header(df, 0, "Row")
            if header is None:
                header = df.columns.tolist()
                respondent_name = df['respondent_name'][0]

            # Add header back
            df.columns = header
            
            # Get unique stimuli and filter range from index 24 to 124
            stimuli = (df['SourceStimuliName'].unique()).tolist()
            
            #indices_to_drop = [0,1,2,3,5,6,23,72,73,74,77,126,128,129,131]
            indices_to_drop = [0,1,2,3,5,6]
            
            # Filter out elements at the specified indices
            stimuli_filtered = [item for idx, item in enumerate(stimuli) if idx not in indices_to_drop]
            
            # Optionally, reset the index of the filtered list
            stimuli_filtered = list(stimuli_filtered)  # Converts to a list if necessary
            
            # Filter baseline_stimuli by excluding any items that start with 'survey'
            stimuli_filtered = [item for item in stimuli_filtered if not item.startswith("survey")]
            
            filtered_df = df[df['SourceStimuliName'].isin(stimuli_filtered)]
            
            # Convert multiple columns to numeric, setting non-numeric values to NaN
            columns_to_convert = ['ET_PupilLeft', 'ET_PupilRight', 'ET_GazeLeftx', 'ET_GazeLefty', 'Timestamp']
            df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')
            
            df = df.dropna(subset=['ET_PupilLeft', 'ET_PupilRight']).reset_index(drop=True)
            
            # Replace -1 with NaN and drop rows with NaN values in 'ET_PupilLeft' and 'ET_PupilRight'
            df[['ET_PupilLeft', 'ET_PupilRight', 'ET_GazeLeftx', 'ET_GazeLefty']] = df[['ET_PupilLeft', 'ET_PupilRight', 'ET_GazeLeftx', 'ET_GazeLefty']].replace(-1, np.nan)
            
            # Set both 'ET_PupilLeft' and 'ET_PupilRight' to NaN if either is NaN
            df.loc[df['ET_PupilLeft'].isna(), 'ET_PupilRight'] = np.nan
            df.loc[df['ET_PupilRight'].isna(), 'ET_PupilLeft'] = np.nan
            
            
            df = NI.process_column_data(df, 'ET_PupilLeft', 'ET_PupilRight')
            
                    
            df = NI.process_none_data(df, 'ET_PupilLeft', 'ET_PupilRight', average_column_name='Average')
            
            # Apply the function to both 'ET_PupilLeft' and 'ET_PupilRight' columns
            df = NI.interpolate_with_constant_fill(df, 'Average')
            
            #df[['ET_GazeLeftx', 'ET_GazeLefty']] = df[['ET_GazeLeftx', 'ET_GazeLefty']].replace(-1, np.nan)
            
            # Set both 'ET_PupilLeft' and 'ET_PupilRight' to NaN if either is NaN
            df.loc[df['ET_GazeLeftx'].isna(), 'ET_GazeLefty'] = np.nan
            df.loc[df['ET_GazeLefty'].isna(), 'ET_GazeLeftx'] = np.nan
            
            # Find indices where either 'ET_GazeLeftx' or 'ET_GazeLefty' is NaN
            nan_indices = df[df['ET_GazeLeftx'].isna() | df['ET_GazeLefty'].isna()].index
            
            # Shift indices by one to get the rows after NaNs and remove duplicates
            rows_to_remove = nan_indices + 1
            rows_to_remove = rows_to_remove[rows_to_remove < len(df)]  # Ensure indices are within bounds
            
            # Drop the identified rows
            df = df.drop(rows_to_remove).reset_index(drop=True)
            
            df = NI.interpolate_with_constant_fill(df, 'ET_GazeLeftx')
            df = NI.interpolate_with_constant_fill(df, 'ET_GazeLefty')
            
            
            cal_27 = df[df['SourceStimuliName'] == '27_cal_points_s'].reset_index(drop=True) # for laptop 27_cal_points-1
            
            ps_27, _ = ps_cal.find_ps_seconds(cal_27, 4, 27)
            
            stim_name = df['SourceStimuliName'].unique()
            
            
            test_image_ps = []
            
            for i in stim_name:
                # Filter the dataframe for the specific stimulus
                df_1 = df[df['SourceStimuliName'] == i].reset_index(drop=True)
                
                # Get the rows starting from index 50
                df_1 = df_1[50:]
                
                test_image_ps.append(df_1['Average'].mean())
            test_image_ps = test_image_ps[:12]
            #measured_ps = ps_27 + test_image_ps
            
            
            """ save these data into csv """
            
            test_images = pd.read_csv(file_with_rgb_file_w) #original_data_23082024_all_laptop
            
            test_images[part_name] = list(ps_27) + test_image_ps # imp: these are the value of calibration and single color test images which is required to recalibration function
            
            
            """ Define the list """
            
            lux_intensity = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99, 100]
            
            
            
            PS_red = RGB_N.recalibration(lux_intensity, coefficients_file_r, all_pssobilities_lux_v4, test_images, 'red', 0, part_name) # 0 = 27 points calibration
            PS_green = RGB_N.recalibration(lux_intensity, coefficients_file_g, all_pssobilities_lux_v4, test_images, 'green', 0, part_name)
            PS_blue = RGB_N.recalibration(lux_intensity, coefficients_file_b, all_pssobilities_lux_v4, test_images, 'blue', 0, part_name)
            PS_white = RGB_N.recalibration(lux_intensity, coefficients_file_w, all_pssobilities_lux_v4, test_images, 'white', 0, part_name)
            
            
            """ pupil size at maximum luminosity"""
            #ps_color_max = RGB_N.get_filtered_values(test_images, 100, 100, 100, part_name) 
            
            lux_r = RGB_N.lux_function(lux_intensity, all_pssobilities_lux_v4, 'red')
            lux_r.append(100)
            lux_g = RGB_N.lux_function(lux_intensity, all_pssobilities_lux_v4, 'green')
            lux_g.append(100)
            lux_b = RGB_N.lux_function(lux_intensity, all_pssobilities_lux_v4, 'blue')
            lux_b.append(100)
            lux_w = RGB_N.lux_function(lux_intensity, all_pssobilities_lux_v4, 'white')
            lux_w.append(100)
            
            STL.append_based_on_last_values(PS_red)
            STL.append_based_on_last_values(PS_green)
            STL.append_based_on_last_values(PS_blue)
            STL.append_based_on_last_values(PS_white)
            try: 
                """ get the coefficients for all colors for particular participants """
                popt_r = RGB_N.fit_model(np.array(lux_r), PS_red)
                popt_g = RGB_N.fit_model(np.array(lux_g), PS_green)
                popt_b = RGB_N.fit_model(np.array(lux_b), PS_blue)
                popt_w = RGB_N.fit_model(np.array(lux_w), PS_white)
                
                
                
                frames_folder = 'frames'  # Folder to save frames from the video
                
        
                video_stimuli = [item for item in stimuli if item.startswith(("HN", "LN", "HP", "LP"))]
                
                
                data_all = []
                dataframe_part = []
                
                for video_file in video_files:
                        
                        pred_ps_lnr_cleaned = []
                        pred_ps_lnr_cleaned_0 = []        
                        print(video_file)
                        #results = VEX.process_video_file(video_file, df, home_dir, video_folder, frames_folder,  popt_r, popt_g, popt_b, popt_w, all_pssobilities_lux_v4)
                        results = VEX.process_video_file_with_gaze_plot(video_file, df, home_dir, video_folder, frames_folder)
                        
                        
                        if results is not None:
                            
                            results, ps_black = PPS.predict_ps(results, popt_r, popt_g, popt_b, popt_w, all_pssobilities_lux_v4)
                        
                            
                            #linear regression 
                            pred_ps_lnr, avg_percentage_error_lnr = LN.normal_regression_LOO([results['r_ps'], results['g_ps'], results['b_ps'], results['grey_based']], results['Average Pupil Size'])
                            
                            # Calculate the average of RGB tuples
                            results['Average'] = results['RGB'].apply(lambda x: np.mean(x))
                            
                            for i, ele in enumerate(results['Lux']):
                                if ele == 0:
                                    pred_ps_lnr_cleaned_0.append(ps_black)
                                else:
                                    pred_ps_lnr_cleaned_0.append(pred_ps_lnr[i])
                            
                            
                            for i, ele in enumerate(pred_ps_lnr_cleaned_0):
                                if ele > max(ps_27):
                                    pred_ps_lnr_cleaned.append(max(ps_27) - ((max(ps_27) - min(ps_27)) * (results['Average'][i]/100)) )
                                elif ele < min(ps_27):
                                    pred_ps_lnr_cleaned.append(min(ps_27))
                                else:
                                    pred_ps_lnr_cleaned.append(ele)
                            
                
                            
                            results['predicted_ps'] = pred_ps_lnr_cleaned
                            results['shifted_predicted_ps'] = results['predicted_ps'].shift(1)
                            
                            error = results['Average Pupil Size'] - results['shifted_predicted_ps']
                            results['error'] = error
                            
                            # Min-Max scaling
                            scaler = MinMaxScaler()
                            results['Scaled Average'] = scaler.fit_transform(results[['Average']])
                            results['Scaled Average'] = 2 - results['Scaled Average']
                            results['Scaled Average'] = results['Scaled Average'].shift(1)
                            results['stimuli'] = pd.Series([video_file] * len(results))
                            results['respondent_name'] = pd.Series([respondent_name] * len(results))
                            data_all.append(results)
                            

                # Concatenate all DataFrames in the list
                merged_dataframe_part = pd.concat(data_all, ignore_index=True)

                merged_dataframe_part.to_csv(os.path.join(output_path + str(respondent_name) + ".csv"))
                    
                dataframe_part.append(merged_dataframe_part)
            except Exception as e:
                print(f'There is a problem to fit the model for participant {respondent_name}: {e}')