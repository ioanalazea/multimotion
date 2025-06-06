from emotion_vector_processor import output_vector_processor

import pandas as pd
import os
import statistics as stats
from scipy.stats import kurtosis
from final_model_pupil import interval_decesion_for_all_participants_split_videos


files_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Files"
)
interval_path = os.path.join(files_path, "required_files/interval-splitted-videos.csv")   #HERE IF I CAN TRY TO PUT IN IN JSON FILE

mapping_data = {
    "HN_1-1": "HN_1",
    "HN_2-1": "HN_2_H",
    "HN_4-1": "HN_4",
    "HN_5-1": "HN_5",
    "HN_6-1": "HN_6",
    "LP_3-1": "LP_3",
    "LP_4-1": "LP_4",
    "LP_6-1": "LP_6",
    "LN_1-1": "LN_1",
    "LN_2-1": "LN_2",
    "LN_3-1": "LN_3",
    "LN_4-1": "LN_4",
    "LN_5-1": "LN_5",
    "LN_6-1": "LN_6",
    "LN_7-1": "LN_7_N",
    "LN_8-1": "LN_9",
    "HP_1-1": "HP_1_L",
    "HP_2-1": "HP_2",
    "HP_3-1": "HP_3_L",
    "HP_4-1": "HP_4",
    "HP_6-1": "HP_6",
    "HP_7-1": "HP_7_H",
    "HN_3-1": "HN_3_H",
    "HN_8-1": "HN_9",
    "HN_7-1": "HN_7",
    "LP_1-1": "LP_1",
    "LP_2-1": "LP_2",
    "LP_5-1": "LP_5",
    "LP_7-1": "LP_7",
    "LP_8-1": "LP_9",
    "HP_5-1": "HP_5",
    "HP_8-1": "HP_9",
    "HN_2-2": "HN_2_L",
    "HN_3-2": "HN_3_L",
    "HP_1-2": "HP_1_H",
    "HP_3-2": "HP_3_H",
    "HP_7-2": "HP_7_L",
    "LN_7-2": "LN_7_P",
}


# Function to handle FER feature extraction
def extract_fer_features(directory_path, interval_path, processed_data_path, mapping_data):
    all_fer_data = []

    # Load the interval data
    interval_data = pd.read_csv(interval_path)
    print("Now extracting for ", interval_path,"Change if you want full/intervals.\n\n")

    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path)
            print("\nExtracting FER features from file:", filename, "\n")
            stimuli = df["SourceStimuliName"].unique()

            # Compute FER features
            fer_features = compute_features_interval(
                df,
                stimuli=stimuli,
                interval_data=interval_data,
                filename=filename,
                file_names=[],
            )

            df_fer = pd.DataFrame(fer_features)
            participant_name = os.path.splitext(filename)[0].split("_")[1]  # Extracts participant code

            df_fer["participant"] = participant_name  # Add participant column

            # Renaming the videos
            df_fer["video"] = df_fer["video"].replace(mapping_data)

            all_fer_data.append(df_fer)

    # Combine all data into a single DataFrame
    final_fer_df = pd.concat(all_fer_data, ignore_index=True)

    # Define file path for FER features and save as CSV
    fer_path = os.path.join(processed_data_path, "FER_features.csv")
    final_fer_df.to_csv(fer_path, index=False)
    print(f"Saved FER features to {fer_path}")

# Function to handle pupil size data extraction
def extract_pupil_data(directory_path, interval_path, pupil_data_path, processed_data_path):
    all_pupil_data = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            print("\nExtracting Pupil Size from file:", filename, "\n")

            # Get pupil size data
            pupil_arousal_data = interval_decesion_for_all_participants_split_videos.get_arousal_data(filename, interval_path, pupil_data_path)

            participant_name = os.path.splitext(filename)[0].split("_")[1]  # Extracts participant code

            df_pupil = pd.DataFrame(
                {
                    "participant": participant_name,
                    "video": pupil_arousal_data["Stimulus_Name"],
                    "Pupil_size": pupil_arousal_data["arousal_data"],
                }
            )
            all_pupil_data.append(df_pupil)

    # Combine all pupil data into a single DataFrame
    final_pupil_df = pd.concat(all_pupil_data, ignore_index=True)

    # Define file path for pupil size data and save as CSV
    pupil_path = os.path.join(processed_data_path, "Pupil_size.csv")
    final_pupil_df.to_csv(pupil_path, index=False)
    print(f"Saved Pupil size data to {pupil_path}")

# Main function to prompt the user for their choice
def statistics_features(data_path, processed_data_path, pupil_data_path):
    print("\nWhat do you want to process?")
    print("1. Extract FER features only")
    print("2. Extract Pupil size data only")
    print("3. Extract both FER features and Pupil size data")

    choice = input("Enter your choice (1/2/3): ")

    if choice == "1":
        extract_fer_features(data_path, interval_path, processed_data_path, mapping_data)
    elif choice == "2":
        extract_pupil_data(data_path, interval_path, pupil_data_path, processed_data_path)
    elif choice == "3":
        extract_fer_features(data_path, interval_path, processed_data_path, mapping_data)
        extract_pupil_data(data_path, interval_path, pupil_data_path, processed_data_path)
    else:
        print("Invalid choice. Please select a valid option.")


# Filter Data by Stimulus
def filter_data_by_stimulus(
    data_participant, interval_data, stimulus, interval_stimulus
):
    stimulus_data = data_participant[data_participant["SourceStimuliName"] == stimulus]
    start_times, end_times = [], []

    # Normalize Timestamp
    stimulus_data = stimulus_data.copy()  # This makes a copy of the slice
    stimulus_data.loc[:, "Timestamp"] = (
        stimulus_data["Timestamp"] - stimulus_data["Timestamp"].min()
    ) / 1000

    for _, row in interval_data[
        interval_data["stimuli_names"] == interval_stimulus
    ].iterrows():
        # Ensure values are strings before splitting
        start_str = str(row["interval_start"])
        end_str = str(row["interval_end"])

        # Convert to float after splitting
        start_times.extend(
            map(float, start_str.split(";") if ";" in start_str else [start_str])
        )
        end_times.extend(
            map(float, end_str.split(";") if ";" in end_str else [end_str])
        )

    filtered_parts = [
        stimulus_data[
            (stimulus_data["Timestamp"] >= start) & (stimulus_data["Timestamp"] <= end)
        ].reset_index(drop=True)
        for start, end in zip(start_times, end_times)
    ]

    return (
        pd.concat(filtered_parts, ignore_index=True)
        if filtered_parts
        else pd.DataFrame()
    )


def compute_features_interval(df, stimuli, interval_data, filename, file_names):
    splited_videos = ["HN_2-1", "HN_3-1", "HP_1-1", "HP_3-1", "HP_7-1", "LN_7-1"]  #
    features = []
    for stimulus in stimuli:
        if stimulus in splited_videos:
            stimulis = [stimulus, stimulus[:-2] + "-2"]
            for i in stimulis:
                filtered_df = filter_data_by_stimulus(df, interval_data, stimulus, i)
                arousal_valence_pairs = output_vector_processor(filtered_df, i)

                all_values = arousal_valence_pairs
                max_valence_values = 0
                max_arousal_values = 0
                min_valence_values = 0
                min_arousal_values = 0
                final_mean_arousal = 0
                final_mean_valence = 0
                std_deviation_arousal = 0
                std_deviation_valence = 0
                kurtosis_FER_arousal = 0
                kurtosis_FER_valence = 0

                sum_valence = 0
                sum_arousal = 0
                total_number_valence = 0
                total_number_arosual = 0
                file_names.append(filename.split(".")[0])
                current_emotion_data = all_values

                # Valence
                if len(current_emotion_data["valence"]) > 0:
                    max_min, max_max = min(current_emotion_data["valence"]), max(
                        current_emotion_data["valence"]
                    )

                    if max_max > abs(max_min):
                        max_valence_values = max_max
                    else:
                        max_valence_values = max_min

                    if all(value > 0 for value in current_emotion_data["valence"]):
                        min_valence_values = min(current_emotion_data["valence"])
                    elif all(value < 0 for value in current_emotion_data["valence"]):
                        min_valence_values = max(current_emotion_data["valence"])
                    else:
                        min_valence_values = 0

                    sum_valence += sum(current_emotion_data["valence"])
                    total_number_valence += len(current_emotion_data["valence"])

                else:
                    max_valence_values = 0
                    min_valence_values = 0

                if len(current_emotion_data["valence"]) > 1:
                    std_deviation_valence = stats.stdev(current_emotion_data["valence"])
                    kurtosis_FER_valence = kurtosis(current_emotion_data["valence"])
                else:
                    std_deviation_valence = 0
                    kurtosis_FER_valence = 0

                # Arousal
                if len(current_emotion_data["arousal"]) > 0:
                    mini, maxi = min(current_emotion_data["arousal"]), max(
                        current_emotion_data["arousal"]
                    )

                    max_arousal_values = maxi
                    min_arousal_values = mini

                    sum_arousal += sum(current_emotion_data["arousal"])
                    total_number_arosual += len(current_emotion_data["arousal"])
                else:
                    max_arousal_values = 0
                    min_arousal_values = 0

                if len(current_emotion_data["arousal"]) > 1:
                    std_deviation_arousal = stats.stdev(current_emotion_data["arousal"])
                    kurtosis_FER_arousal = kurtosis(current_emotion_data["arousal"])
                else:
                    std_deviation_arousal = 0
                    kurtosis_FER_arousal = 0

                final_mean_arousal = (
                    sum_arousal / total_number_arosual
                    if total_number_arosual > 0
                    else 0
                )
                final_mean_valence = (
                    sum_valence / total_number_valence
                    if total_number_valence > 0
                    else 0
                )

                features.append(
                    {
                        "participant": file_names[0],
                        "video": i,
                        "FER_Mean_Valence": ""
                        if final_mean_valence == 0
                        else final_mean_valence,
                        "FER_Mean_Arousal": ""
                        if final_mean_arousal == 0
                        else final_mean_arousal,
                        "FER_Std_Valence": ""
                        if std_deviation_valence == 0
                        else std_deviation_valence,
                        "FER_Std_Arousal": ""
                        if std_deviation_arousal == 0
                        else std_deviation_arousal,
                        "FER_Kurtosis_Valence": ""
                        if kurtosis_FER_valence == 0
                        else kurtosis_FER_valence,
                        "FER_Kurtosis_Arousal": ""
                        if kurtosis_FER_arousal == 0
                        else kurtosis_FER_arousal,
                        "FER_Max_Valence": ""
                        if max_valence_values == 0
                        else max_valence_values,
                        "FER_Max_Arousal": ""
                        if max_arousal_values == 0
                        else max_arousal_values,
                        "FER_Min_Valence": ""
                        if min_valence_values == 0
                        else min_valence_values,
                        "FER_Min_Arousal": ""
                        if min_arousal_values == 0
                        else min_arousal_values,
                    }
                )
        else:
            filtered_df = filter_data_by_stimulus(df, interval_data, stimulus, stimulus)

            if not filtered_df.empty:
                arousal_valence_pairs = output_vector_processor(filtered_df, stimulus)
                all_values = arousal_valence_pairs
                max_valence_values = 0
                max_arousal_values = 0
                min_valence_values = 0
                min_arousal_values = 0
                final_mean_arousal = 0
                final_mean_valence = 0
                std_deviation_arousal = 0
                std_deviation_valence = 0
                kurtosis_FER_arousal = 0
                kurtosis_FER_valence = 0

                sum_valence = 0
                sum_arousal = 0
                total_number_valence = 0
                total_number_arosual = 0
                file_names.append(filename.split(".")[0])
                current_emotion_data = all_values

                # Valence
                if len(current_emotion_data["valence"]) > 0:
                    max_min, max_max = min(current_emotion_data["valence"]), max(
                        current_emotion_data["valence"]
                    )

                    if max_max > abs(max_min):
                        max_valence_values = max_max
                    else:
                        max_valence_values = max_min

                    if all(value > 0 for value in current_emotion_data["valence"]):
                        min_valence_values = min(current_emotion_data["valence"])
                    elif all(value < 0 for value in current_emotion_data["valence"]):
                        min_valence_values = max(current_emotion_data["valence"])
                    else:
                        min_valence_values = 0

                    sum_valence += sum(current_emotion_data["valence"])
                    total_number_valence += len(current_emotion_data["valence"])

                else:
                    max_valence_values = 0
                    min_valence_values = 0

                if len(current_emotion_data["valence"]) > 1:
                    std_deviation_valence = stats.stdev(current_emotion_data["valence"])
                    kurtosis_FER_valence = kurtosis(current_emotion_data["valence"])
                else:
                    std_deviation_valence = 0
                    kurtosis_FER_valence = 0

                # Arousal
                if len(current_emotion_data["arousal"]) > 0:
                    mini, maxi = min(current_emotion_data["arousal"]), max(
                        current_emotion_data["arousal"]
                    )

                    max_arousal_values = maxi
                    min_arousal_values = mini

                    sum_arousal += sum(current_emotion_data["arousal"])
                    total_number_arosual += len(current_emotion_data["arousal"])
                else:
                    max_arousal_values = 0
                    min_arousal_values = 0

                if len(current_emotion_data["arousal"]) > 1:
                    std_deviation_arousal = stats.stdev(current_emotion_data["arousal"])
                    kurtosis_FER_arousal = kurtosis(current_emotion_data["arousal"])
                else:
                    std_deviation_arousal = 0
                    kurtosis_FER_arousal = 0

                final_mean_arousal = (
                    sum_arousal / total_number_arosual
                    if total_number_arosual > 0
                    else 0
                )
                final_mean_valence = (
                    sum_valence / total_number_valence
                    if total_number_valence > 0
                    else 0
                )

                features.append(
                    {
                        "participant": file_names[0],
                        "video": stimulus,
                        "FER_Mean_Valence": ""
                        if final_mean_valence == 0
                        else final_mean_valence,
                        "FER_Mean_Arousal": ""
                        if final_mean_arousal == 0
                        else final_mean_arousal,
                        "FER_Std_Valence": ""
                        if std_deviation_valence == 0
                        else std_deviation_valence,
                        "FER_Std_Arousal": ""
                        if std_deviation_arousal == 0
                        else std_deviation_arousal,
                        "FER_Kurtosis_Valence": ""
                        if kurtosis_FER_valence == 0
                        else kurtosis_FER_valence,
                        "FER_Kurtosis_Arousal": ""
                        if kurtosis_FER_arousal == 0
                        else kurtosis_FER_arousal,
                        "FER_Max_Valence": ""
                        if max_valence_values == 0
                        else max_valence_values,
                        "FER_Max_Arousal": ""
                        if max_arousal_values == 0
                        else max_arousal_values,
                        "FER_Min_Valence": ""
                        if min_valence_values == 0
                        else min_valence_values,
                        "FER_Min_Arousal": ""
                        if min_arousal_values == 0
                        else min_arousal_values,
                    }
                )
    return features
