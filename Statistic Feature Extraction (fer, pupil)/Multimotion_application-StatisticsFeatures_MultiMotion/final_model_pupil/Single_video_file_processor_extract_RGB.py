# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 12:19:50 2024

@author: zp20945
"""
import pandas as pd
import numpy as np

from final_model_pupil import get_radius


import os

import cv2 
import matplotlib.pyplot as plt


def process_video_file_with_gaze_plot(video_file, df, home_dir, video_folder, frames_folder):    
    """
    Processes a single video file to extract frame RGB values, calculate pupil sizes, and append results.
    Also generates a plot of the gaze position with a 500-radius circle.
    
    Parameters:
    video_file (str): Name of the video file.
    df (DataFrame): DataFrame containing pupil size, gaze data, and timestamp information.
    home_dir (str): The home directory path.
    video_folder (str): The folder where the video files are stored.
    frames_folder (str): The folder where frames will be saved.
    
    Returns:
    DataFrame: Results including frame filename, average pupil size, gaze data, and calculated RGB lux values.
    """
    # Filter DataFrame for the current video file
    df_filtered = df[df['SourceStimuliName'] == video_file].reset_index(drop=True)
    
    if df_filtered.empty:
        return None
    
    # Get 10 equidistant timestamps from the filtered dataframe
    equidistant_indices = np.linspace(0, len(df_filtered) - 1, 10, dtype=int)
    equidistant_rows = df_filtered.iloc[equidistant_indices]

    video_RGB = []
    gaze_values = []  # Store gaze values for each frame
    video_path = os.path.join(home_dir, video_folder, f"{video_file}.mp4")
    
    df_filtered['Timestamp_filtered'] = df_filtered['Timestamp'] - df_filtered['Timestamp'].iloc[0]
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video file: {video_path}")
        return None
    
    

    frame_count = 0
    old_timestamp = 0
    frame_timestamps = []
    fps = cap.get(cv2.CAP_PROP_FPS)

    target_width, target_height = 1920, 1080  # Target resolution

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame to target resolution 1920x1080
        frame = cv2.resize(frame, (target_width, target_height))

        frame_height, frame_width = frame.shape[:2]
        timestamp = (frame_count / fps) * 1000
        
        
        avg_color_per_row = np.mean(frame, axis=0)  # Average per row
        avg_color = np.mean(avg_color_per_row, axis=0)  # Average across all rows
        b_avg_frame, g_avg_frame, r_avg_frame = avg_color
        
        matching_data = df_filtered[(df_filtered['Timestamp_filtered'] <= timestamp) & (df_filtered['Timestamp_filtered'] > old_timestamp)].reset_index(drop=True)
        old_timestamp = timestamp
        
        gaze_x_list, gaze_y_list = [], []
        gaze_x_avg, gaze_y_avg = None, None  # Default values if no gaze found for the frame
        
        # Use gaze data from the matching_data dataframe instead of equidistant_rows
        for _, row in matching_data.iterrows():  # Iterate through the rows in matching_data
            gaze_x, gaze_y = row['ET_GazeLeftx'], row['ET_GazeLefty']
            
            # Scale the gaze coordinates to match the frame's resolution (assuming the gaze data is in 1920x1080 resolution)
            scaled_x = int((frame_width / 1920) * gaze_x)
            scaled_y = int((frame_height / 1080) * gaze_y)
        
            if 0 <= scaled_x < frame_width and 0 <= scaled_y < frame_height:
                gaze_x_list.append(scaled_x)
                gaze_y_list.append(scaled_y)
        
        if gaze_x_list and gaze_y_list:
            
            gaze_x_avg = int(np.mean(gaze_x_list))
            gaze_y_avg = int(np.mean(gaze_y_list))
            gaze_values.append((gaze_x_avg, gaze_y_avg))

            # Get average RGB within a 500-radius circle
            r_avg, g_avg, b_avg = get_radius.get_points_average_rgb_within_circle(frame, gaze_x_avg, gaze_y_avg, radius=300)
            r_avg_frame, g_avg_frame, b_avg_frame = get_radius.get_average_rgb(frame)
            
            # Check if the region is bright (close to 255)
            if r_avg > r_avg_frame or g_avg > g_avg_frame or b_avg > b_avg_frame:
                
                r_10p_avg, g_10p_avg, b_10p_avg = get_radius.get_10_points_average_rgb_within_circle(frame, gaze_x_avg, gaze_y_avg, radius=400)
            else:
                r_10p_avg, g_10p_avg, b_10p_avg = r_avg_frame, g_avg_frame, b_avg_frame
            
            # Convert to percentage
            R = int((r_10p_avg * 100) / 255)
            G = int((g_10p_avg * 100) / 255)
            B = int((b_10p_avg * 100) / 255)

            video_RGB.append((R, G, B))

            # Plot gaze position on the frame
            #plot_gaze_with_circle(frame, gaze_x_avg, gaze_y_avg, frame_count, frames_folder)
        else:
            gaze_values.append((None, None))
            video_RGB.append((r_avg_frame, g_avg_frame, b_avg_frame))

        frame_timestamps.append(timestamp)
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    # Align frame timestamps with pupil size data
    average_pupil_sizes = []
    df_timestamp_filtered = []
    df_timestamp = []

    for timestamp in frame_timestamps:
        matching_data = df_filtered[(df_filtered['Timestamp_filtered'] <= timestamp) & (df_filtered['Timestamp_filtered'] > old_timestamp)].reset_index(drop=True)
        old_timestamp = timestamp

        if not matching_data.empty:
            average_size = matching_data['Average'].mean()
            average_pupil_sizes.append(average_size)
            df_timestamp_filtered.append(timestamp / 1000)
            df_timestamp.append(matching_data['Timestamp'].iloc[0])
        else:
            average_pupil_sizes.append(None)
            df_timestamp.append(None)

    print(len(average_pupil_sizes))
    print(len(video_RGB))
    print(len(df_timestamp))
    
    # Create a DataFrame with matching data
    results_df = pd.DataFrame({
        'Average Pupil Size': average_pupil_sizes,
        'RGB': video_RGB,
        #'Gaze (x, y)': gaze_values,  # Add gaze values
        'df_Timestamp': df_timestamp
    }).dropna(subset=['Average Pupil Size']).reset_index(drop=True)
    
    results_df['df_timestamp_filtered'] = df_timestamp_filtered
    
    return results_df

def plot_gaze_with_circle(frame, gaze_x, gaze_y, frame_count, frames_folder):
    """
    Plots the gaze position on the current frame and draws a 500-radius circle.
    
    Parameters:
    frame (numpy array): The video frame.
    gaze_x (int): The x-coordinate of the gaze.
    gaze_y (int): The y-coordinate of the gaze.
    frame_count (int): The index of the current frame.
    frames_folder (str): Directory to save the plot images.
    """
    fig, ax = plt.subplots(1)
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display in matplotlib

    # Plot the gaze point
    ax.plot(gaze_x, gaze_y, 'ro')  # Red dot for gaze point

    # Draw a circle around the gaze point
    circle = plt.Circle((gaze_x, gaze_y), 400, color='green', fill=False, linewidth=2)
    ax.add_patch(circle)

    # Set plot title and save the figure
    ax.set_title(f'Frame {frame_count} - Gaze Point with 500-Radius Circle')
    plt.savefig(os.path.join(frames_folder, f'gaze_frame_{frame_count}.png'))
    plt.show()
    plt.close()
    
"""    

def process_video_file(video_file, df, home_dir, video_folder, frames_folder):    
    
    # Filter DataFrame for the current video file
    df_filtered = df[df['SourceStimuliName'] == video_file].reset_index(drop=True)
    
    if df_filtered.empty:
        return None
    
    # Get 10 equidistant timestamps from the filtered dataframe
    equidistant_indices = np.linspace(0, len(df_filtered) - 1, 10, dtype=int)
    equidistant_rows = df_filtered.iloc[equidistant_indices]

    measured_ps = [df_filtered['Average'].mean()]  # Pre-calculate pupil size mean
    video_RGB = []
    video_path = os.path.join(home_dir, video_folder, f"{video_file}.mp4")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video file: {video_path}")
        return None

    frame_count = 0
    frame_timestamps = []
    fps = cap.get(cv2.CAP_PROP_FPS)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate the average RGB values for the current frame
        avg_color_per_row = np.mean(frame, axis=0)  # Average per row
        
        #avg_color = np.mean(avg_color_per_row, axis=0)  # Average across all rows
        
        # Append the average RGB values to the video_RGB list
        #video_RGB.append(tuple(avg_color))  # Store as (R, G, B) tup
        
        

        frame_height, frame_width = frame.shape[:2]

        gaze_x_list, gaze_y_list = [], []

        for _, row in equidistant_rows.iterrows():
            gaze_x, gaze_y = row['ET_GazeLeftx'], row['ET_GazeLefty']
            scaled_x = int((frame_width / 1920) * gaze_x)
            scaled_y = int((frame_height / 1080) * gaze_y)
            
            if 0 <= scaled_x < frame_width and 0 <= scaled_y < frame_height:
                gaze_x_list.append(scaled_x)
                gaze_y_list.append(scaled_y)

        if gaze_x_list and gaze_y_list:
            avg_gaze_x = int(np.mean(gaze_x_list))
            avg_gaze_y = int(np.mean(gaze_y_list))

            # Get average RGB within a 500-radius circle
            r_avg, g_avg, b_avg, frame_with_circle = get_radius.get_points_average_rgb_within_circle(frame, avg_gaze_x, avg_gaze_y, radius=500)
            r_avg_frame, g_avg_frame, b_avg_frame = get_radius.get_average_rgb(frame)
            
            # Draw the circle on the frame
            cv2.circle(frame_with_circle, (avg_gaze_x, avg_gaze_y), 500, (0, 255, 0), 3)  # Green circle with thickness of 3

            if r_avg > r_avg_frame or g_avg > g_avg_frame or b_avg > b_avg_frame:
                r_10p_avg, g_10p_avg, b_10p_avg = get_radius.get_10_points_average_rgb_within_circle(frame, avg_gaze_x, avg_gaze_y, radius=1100)
            else:
                r_10p_avg, g_10p_avg, b_10p_avg = r_avg_frame, g_avg_frame, b_avg_frame

            R = int((r_10p_avg * 100) / 255)
            G = int((g_10p_avg * 100) / 255)
            B = int((b_10p_avg * 100) / 255)
            video_RGB.append((R, G, B))
            
        timestamp = (frame_count / fps) * 1000
        frame_timestamps.append(timestamp)
        
        # Display the frame with the circle
        cv2.imshow('Frame with Circle', frame_with_circle)
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break
       
        frame_count += 1

    cap.release()
    
   


    # Align frame timestamps with pupil size data
    old_timestamp = 0
    average_pupil_sizes = []
    df_timestamp_filtered = []
    df_timestamp = []

    df_filtered['Timestamp_filtered'] = df_filtered['Timestamp'] - df_filtered['Timestamp'].iloc[0]
    
    for timestamp in frame_timestamps:
        matching_data = df_filtered[(df_filtered['Timestamp_filtered'] <= timestamp) & (df_filtered['Timestamp_filtered'] > old_timestamp)].reset_index(drop=True)
        old_timestamp = timestamp

        if not matching_data.empty:
            average_size = matching_data['Average'].mean()
            average_pupil_sizes.append(average_size)
            df_timestamp_filtered.append(timestamp/1000)
            df_timestamp.append(matching_data['Timestamp'].iloc[0])
        else:
            average_pupil_sizes.append(None)
            #df_timestamp_filtered.append(None)
            df_timestamp.append(None)

    # Create a DataFrame with matching data
    results_df = pd.DataFrame({
        'Average Pupil Size': average_pupil_sizes,
        'RGB': video_RGB,
        'df_Timestamp': df_timestamp
    }).dropna(subset=['Average Pupil Size']).reset_index(drop=True)
    
     
    results_df['df_timestamp_filtered'] = df_timestamp_filtered
    
    return results_df
"""
    
"""        
    # Adjust timestamp column for the current video
    
    df_filtered['Timestamp_filtered'] = df_filtered['Timestamp'] - df_filtered['Timestamp'][0]


    for timestamp in frame_timestamps:
        matching_data = df_filtered[(df_filtered['Timestamp_filtered'] <= timestamp) & (df_filtered['Timestamp_filtered'] > old_timestamp)]
        #print(matching_data)
        matching_data = matching_data.reset_index(drop=True)

        old_timestamp = timestamp

        if not matching_data.empty:
            average_size = matching_data['Average'].mean()
            average_pupil_sizes.append(average_size)
            df_timestamp_filtered.append(matching_data['Timestamp_filtered'].iloc[0])
            df_timestamp.append(matching_data['Timestamp'].iloc[0])
            
        else:
            average_pupil_sizes.append(None)
            df_timestamp_filtered.append(None)
            df_timestamp.append(None)
            

    # Filter out frames with no matching pupil size data
    results_df = pd.DataFrame({
        'Average Pupil Size': average_pupil_sizes,
        'RGB': video_RGB,
        'df_Timestamp': df_timestamp
    }).dropna(subset=['Average Pupil Size']).reset_index(drop=True)

    #results_df['df_Timestamp_filtered'] = df_timestamp_filtered
    
    # Add one second to the last frame's timestamp and copy the last frame
    if not results_df.empty:
        last_rgb = results_df['RGB'].iloc[-1]
        last_timestamp = results_df['df_Timestamp'].iloc[-1] + 1000  # Add 1000 milliseconds

        # Append the last frame's RGB and new timestamp
        results_df = results_df.append({
            'Average Pupil Size': average_pupil_sizes[-1],
            'RGB': last_rgb,
            'df_Timestamp': last_timestamp
        }, ignore_index=True)
    """
    

"""
# Example usage
for video_file in video_files:
    results = process_video_file(video_file, df, home_dir, video_folder, frames_folder, file_path, popt_r, popt_g, popt_b, popt_w)
    if results is not None:
        results.to_csv(f'{video_file}_results.csv', index=False)
"""