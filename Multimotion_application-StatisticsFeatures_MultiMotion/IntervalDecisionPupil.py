# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 11:05:34 2024

@author: zeelp
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from matplotlib.lines import Line2D
import os

# Global configurations
sns.set(style="whitegrid")

# Define Paths
def define_paths():
    home_dir = "D:/MASTER/Uni of Essex/Disseration/Hassan/Files/Myfiles"
    relative_path = "D:/MASTER/Uni of Essex/Disseration/Hassan/Files/Myfiles/all_healthy_part"
    interval_path = "D:/MASTER/Uni of Essex/Disseration/Hassan/Files/Myfiles/filtered_participant/interval.csv"
    gt_path = "D:/MASTER/Uni of Essex/Disseration/Hassan/Files/Myfiles/filtered_participant/individual_ground_truth.csv"
    return home_dir, relative_path, interval_path, gt_path

# Load Data
# def load_data(home_dir, relative_path, interval_path, filename):
#     interval_data = pd.read_csv(interval_path)
#     participant_files = [
#         os.path.join(home_dir, relative_path, f)
#         for f in os.listdir(os.path.join(home_dir, relative_path)) if f.endswith('.csv')
#     ]
#     return interval_data, participant_files

def load_data(home_dir, relative_path, interval_path, filename):
    # Extract the 5-character participant code from the filename
    participant_code = filename.split('_')[1][:5]  # Assumes the format "XXX_YYYYY.csv"
    
    # Load the interval data
    interval_data = pd.read_csv(interval_path)
    
    # Filter participant files that match the 5-character code
    participant_files = [
        os.path.join(home_dir, relative_path, f)
        for f in os.listdir(os.path.join(home_dir, relative_path))
        if f.endswith('.csv') and participant_code in f
    ]
    
    return interval_data, participant_files
# Process Individual Participant
def process_participant(file, interval_data):
    data_participant = pd.read_csv(file, encoding='ISO-8859-1')
    part_name = os.path.splitext(os.path.basename(file))[0]
    stimuli = data_participant['stimuli'].unique()

    stimuli_data = []
    for stimulus in stimuli:
        filtered_df = filter_data_by_stimulus(data_participant, interval_data, stimulus)
        if not filtered_df.empty:
            avg_pupil_size = filtered_df['Average Pupil Size'].mean()
            corrected_pupil_size = filtered_df['shifted_predicted_ps'].mean()
            stimuli_data.append({
                "Stimulus_Name": stimulus,
                "average_pupil_size": avg_pupil_size,
                "corrected_pupil_size": corrected_pupil_size,
                "Participant": part_name,
                "arousal_data": avg_pupil_size - corrected_pupil_size
            })
    return pd.DataFrame(stimuli_data)

# Filter Data by Stimulus
def filter_data_by_stimulus(data_participant, interval_data, stimulus):
    stimulus_data = data_participant[data_participant['stimuli'] == stimulus]
    start_times, end_times = [], []
    for _, row in interval_data[interval_data['stimuli_names'] == stimulus].iterrows():
        start_times.extend(map(float, row['interval_start'].split(';')))
        end_times.extend(map(float, row['interval_end'].split(';')))

    filtered_parts = [
        stimulus_data[(stimulus_data['df_timestamp_filtered'] >= start) &
                      (stimulus_data['df_timestamp_filtered'] <= end)].reset_index(drop=True)
        for start, end in zip(start_times, end_times)
    ]
    return pd.concat(filtered_parts, ignore_index=True) if filtered_parts else pd.DataFrame()

# Merge Data and Ground Truth
def merge_with_ground_truth(all_data, gt_path, mapping_data):
    data_gt = pd.read_csv(gt_path)
    all_data['Stimulus_Name'] = all_data['Stimulus_Name'].replace(mapping_data)
    data_gt.rename(columns={'Stimulus_Name': 'stimuli_name_2'}, inplace=True)
    all_data.rename(columns={'Stimulus_Name': 'stimuli_name_1'}, inplace=True)
    return pd.merge(all_data, data_gt, left_on=['stimuli_name_1', 'Participant'], right_on=['stimuli_name_2', 'Participant'], how='inner').dropna(subset=['arousal_data'])

       
def get_arousal_data(filename):
    home_dir, relative_path, interval_path, gt_path = define_paths()
    interval_data, participant_files = load_data(home_dir, relative_path, interval_path, filename)

    all_data = pd.concat([process_participant(file, interval_data) for file in participant_files], ignore_index=True)

    mapping_data = {
        'HN_1-1' : 'HN_1',         'HN_2-1' : 'HN_2',        'HN_4-1' : 'HN_4',        'HN_5-1' : 'HN_5',        'HN_6-1' : 'HN_6',        'LP_3-1' : 'LP_3',
        'LP_4-1' : 'LP_4',       'LP_6-1' : 'LP_6',      'LN_1-1' : 'LN_1',      'LN_2-1' : 'LN_2',     'LN_3-1' : 'LN_3',     'LN_4-1' : 'LN_4',    'LN_5-1' : 'LN_5',
        'LN_6-1' : 'LN_6',  'LN_7-1' : 'LN_7',     'LN_8-1' : 'LN_8',   'HP_1-1' : 'HP_1',  'HP_2-1' : 'HP_2', 'HP_3-1' : 'HP_3', 'HP_4-1' : 'HP_4', 'HP_6-1' : 'HP_6',
        'HP_7-1' : 'HP_7', 'HN_3-1' : 'HN_3', 'HN_8-1' : 'HN_8', 'HN_7-1' : 'HN_7', 'LP_1-1' : 'LP_1',  'LP_2-1' : 'LP_2', 'LP_5-1' : 'LP_5', 'LP_7-1' : 'LP_7',
        'LP_8-1' : 'LP_8',  'HP_5-1' : 'HP_5',  'HP_8-1' : 'HP_8'
        }


    merged_df = merge_with_ground_truth(all_data, os.path.join(home_dir, gt_path), mapping_data)
    return merged_df
