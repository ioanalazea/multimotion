# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 11:05:34 2024

@author: zeelp
"""
import numpy as np
import pandas as pd
import seaborn as sns
import os

# Global configurations
sns.set(style="whitegrid")


# Define Paths
def define_paths():
    # files path
    files_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "Files",
    )
    home_dir = os.path.join(files_path, "Myfiles")
                                                                         
    return home_dir


def load_data(home_dir, pupil_data_path, interval_path, filename):
    # Extract the 5-character participant code from the filename
    participant_code = filename.split("_")[1][:5]  # Assumes the format "XXX_YYYYY.csv"

    # Load the interval data
    interval_data = pd.read_csv(interval_path)

    # Filter participant files that match the 5-character code
    participant_files = [
        os.path.join(home_dir, pupil_data_path, f)
        for f in os.listdir(os.path.join(home_dir, pupil_data_path))
        if f.endswith(".csv") and participant_code in f
    ]

    return interval_data, participant_files


# Process Individual Participant with intervals
def process_participant(file, interval_data):
    data_participant = pd.read_csv(file, encoding="ISO-8859-1")
    part_name = os.path.splitext(os.path.basename(file))[0]
    stimuli = data_participant["stimuli"].unique()

    splited_videos = ["HN_2-1", "HN_3-1", "HP_1-1", "HP_3-1", "HP_7-1", "LN_7-1"]

    stimuli_data = []
    for stimulus in stimuli:
        if stimulus in splited_videos:

            stimulis = [stimulus, stimulus[:-2] + "-2"]
            for i in stimulis:
                filtered_df = filter_data_by_stimulus(
                    data_participant, interval_data, stimulus, i
                )
                if not filtered_df.empty:
                    avg_pupil_size = filtered_df["Average Pupil Size"].mean()
                    corrected_pupil_size = filtered_df["shifted_predicted_ps"].mean()
                    stimuli_data.append(
                        {
                            "Stimulus_Name": i,
                            "average_pupil_size": avg_pupil_size,
                            "corrected_pupil_size": corrected_pupil_size,
                            "Participant": part_name,
                            "arousal_data": avg_pupil_size - corrected_pupil_size,
                        }
                    )

        else:

            filtered_df = filter_data_by_stimulus(
                data_participant, interval_data, stimulus, stimulus
            )
            if not filtered_df.empty:
                avg_pupil_size = filtered_df["Average Pupil Size"].mean()
                corrected_pupil_size = filtered_df["shifted_predicted_ps"].mean()
                stimuli_data.append(
                    {
                        "Stimulus_Name": stimulus,
                        "average_pupil_size": avg_pupil_size,
                        "corrected_pupil_size": corrected_pupil_size,
                        "Participant": part_name,
                        "arousal_data": avg_pupil_size - corrected_pupil_size,
                    }
                )

    return pd.DataFrame(stimuli_data)


# Process Individual Participant for the whole video
def process_participant_without_interval(file):
    data_participant = pd.read_csv(file, encoding="ISO-8859-1")
    part_name = os.path.splitext(os.path.basename(file))[0]
    stimuli = data_participant["stimuli"].unique()

    # change here
    stimuli_data = []
    for stimulus in stimuli:
        filtered_df = data_participant.loc[data_participant["stimuli"] == stimulus]
        avg_pupil_size = filtered_df["Average Pupil Size"].mean()
        corrected_pupil_size = filtered_df["shifted_predicted_ps"].mean()
        stimuli_data.append(
            {
                "Stimulus_Name": stimulus,
                "average_pupil_size": avg_pupil_size,
                "corrected_pupil_size": corrected_pupil_size,
                "Participant": part_name,
                "arousal_data": avg_pupil_size - corrected_pupil_size,
            }
        )

    return pd.DataFrame(stimuli_data)


# Filter Data by Stimulus

def filter_data_by_stimulus(data_participant, interval_data, stimulus, interval_stimulus):
    stimulus_data = data_participant[data_participant["stimuli"] == stimulus]
    start_times, end_times = [], []

    for _, row in interval_data[interval_data["stimuli_names"] == interval_stimulus].iterrows():
        # Convert to string before splitting
        start = str(row["interval_start"]).split(";")
        end = str(row["interval_end"]).split(";")

        # Convert to floats
        start_times.extend(map(float, start))
        end_times.extend(map(float, end))

    filtered_parts = [
        stimulus_data[
            (stimulus_data["df_timestamp_filtered"] >= start)
            & (stimulus_data["df_timestamp_filtered"] <= end)
        ].reset_index(drop=True)
        for start, end in zip(start_times, end_times)
    ]

    return pd.concat(filtered_parts, ignore_index=True) if filtered_parts else pd.DataFrame()


# Main
def get_arousal_data(filename, interval_path, pupil_data_path):
    home_dir = define_paths()
    interval_data, participant_files = load_data(
        home_dir, pupil_data_path, interval_path, filename
    )

    # Define the full list of stimuli
    stimuli_list = [
        'HN_1', 'HN_2_H', 'HN_4', 'HN_5', 'HN_6', 'LP_3', 'LP_4', 'LP_6', 'LN_1', 'LN_2', 'LN_3', 'LN_4', 'LN_5', 
        'LN_6', 'LN_7_N', 'LN_9', 'HP_1_L', 'HP_2', 'HP_3_L', 'HP_4', 'HP_6', 'HP_7_H', 'HN_3_H', 'HN_9', 'HN_7', 
        'LP_1', 'LP_2', 'LP_5', 'LP_7', 'LP_9', 'HP_5', 'HP_9', 'HN_2_L', 'HN_3_L', 'HP_1_H', 'HP_3_H', 'HP_7_L', 'LN_7_P'
    ]

    mapping_data = {
        'HN_1-1' : 'HN_1', 'HN_2-1' : 'HN_2_H', 'HN_4-1' : 'HN_4', 'HN_5-1' : 'HN_5', 'HN_6-1' : 'HN_6', 'LP_3-1' : 'LP_3',
        'LP_4-1' : 'LP_4', 'LP_6-1' : 'LP_6', 'LN_1-1' : 'LN_1', 'LN_2-1' : 'LN_2', 'LN_3-1' : 'LN_3', 'LN_4-1' : 'LN_4', 'LN_5-1' : 'LN_5',
        'LN_6-1' : 'LN_6', 'LN_7-1' : 'LN_7_N', 'LN_8-1' : 'LN_9', 'HP_1-1' : 'HP_1_L', 'HP_2-1' : 'HP_2', 'HP_3-1' : 'HP_3_L', 'HP_4-1' : 'HP_4', 'HP_6-1' : 'HP_6',
        'HP_7-1' : 'HP_7_H', 'HN_3-1' : 'HN_3_H', 'HN_8-1' : 'HN_9', 'HN_7-1' : 'HN_7', 'LP_1-1' : 'LP_1', 'LP_2-1' : 'LP_2', 'LP_5-1' : 'LP_5', 'LP_7-1' : 'LP_7',
        'LP_8-1' : 'LP_9', 'HP_5-1' : 'HP_5', 'HP_8-1' : 'HP_9',
        'HN_2-2' : 'HN_2_L', 'HN_3-2' : 'HN_3_L', 'HP_1-2' : 'HP_1_H','HP_3-2' : 'HP_3_H', 'HP_7-2' : 'HP_7_L', 'LN_7-2' : 'LN_7_P',
    }

    all_data = []

    # Check if participant files exist
    if not participant_files:
        # If no participant files are found, return NaN values for each video
        all_data = pd.DataFrame(
            [[stimulus, np.nan, np.nan, filename.split('.')[0], np.nan] for stimulus in stimuli_list],
            columns=["Stimulus_Name", "average_pupil_size", "corrected_pupil_size", "Participant", "arousal_data"]
        )
    else:
        # Process the participant files
        all_data = pd.concat([process_participant(file, interval_data) for file in participant_files], ignore_index=True)

    all_data["Stimulus_Name"] = all_data["Stimulus_Name"].replace(mapping_data)
    return all_data