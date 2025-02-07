from emotion_vector_processor import (
    output_vector_processor,
    output_vector_processor_path,
    video_emotion,
)
from Final_Code_Pupil_Dilation import pupil_statistics_data
from gsr_feature_extraction import GSR_statisticsFeatures
from Heart_Rate_code import heartRate_statistics_data
from signalQualityGSR import GSRSignalQuality
import pandas as pd
import os
import statistics as stats
from scipy.stats import kurtosis
from Quality_Signals import (
    quality_signals_FER,
    quality_signals_PupilSize,
    quality_signals_HeartRate,
    quality_signals_GSR,
)
from final_model_pupil import interval_decesion_for_all_participants_split_videos

interval_path = "D:/MASTER/Uni of Essex/Disseration/Hassan/multimotion-emotion-recognition/Multimotion_application-StatisticsFeatures_MultiMotion/final_model_pupil/required_files/interval.csv"
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


def statistics_features(data_path, processed_data_path):
    directory_path = data_path
    # Load the interval data
    interval_data = pd.read_csv(interval_path)

    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            file_names = []
            df = pd.read_csv(file_path)

            df.dropna(axis=1, how="all", inplace=True)
            stimuli = df["SourceStimuliName"].unique()
            
            # New pupil features
            pupil_arousal_data = (
                interval_decesion_for_all_participants_split_videos.get_arousal_data(
                    filename
                )
            )
            
            
            # Features for whole video
            features = compute_features(
                df, filename=filename, file_path=file_path, file_names=file_names
            )
            
            # Features for interval
            # features = compute_features_interval(
            #     df,
            #     stimuli=stimuli,
            #     interval_data=interval_data,
            #     filename=filename,
            #     file_names=file_names,
            # )


        df = pd.DataFrame(features)
        df["video"] = df["video"].replace(mapping_data)

        df_merged = df.merge(
            pupil_arousal_data[["stimuli_name_1", "arousal_data"]],
            left_on="video",
            right_on="stimuli_name_1",
            how="left",
        )

        # Drop the helper column 'stimuli_name_1' from the merge
        df_merged = df_merged.drop(columns=["stimuli_name_1"])

        path = processed_data_path + rf"\{filename}"
        df_merged.to_csv(path, index=False)


# Filter Data by Stimulus
def filter_data_by_stimulus(
    data_participant, interval_data, stimulus, interval_stimulus
):
    stimulus_data = data_participant[data_participant["SourceStimuliName"] == stimulus]
    start_times, end_times = [], []

    stimulus_data["Timestamp"] = (
        stimulus_data["Timestamp"] - stimulus_data["Timestamp"].min()
    ) / 1000

    for _, row in interval_data[
        interval_data["stimuli_names"] == interval_stimulus
    ].iterrows():
        start_times.extend(map(float, row["interval_start"].split(";")))
        end_times.extend(map(float, row["interval_end"].split(";")))

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
                        "FER_Mean_Valence": final_mean_valence,
                        "FER_Mean_Arousal": final_mean_arousal,
                        "FER_Std_Valence": std_deviation_valence,
                        "FER_Std_Arousal": std_deviation_arousal,
                        "FER_Kurtosis_Valence": kurtosis_FER_valence,
                        "FER_Kurtosis_Arousal": kurtosis_FER_arousal,
                        "FER_Max_Valence": max_valence_values,
                        "FER_Max_Arousal": max_arousal_values,
                        "FER_Min_Valence": min_valence_values,
                        "FER_Min_Arousal": min_arousal_values,
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
                        "FER_Mean_Valence": final_mean_valence,
                        "FER_Mean_Arousal": final_mean_arousal,
                        "FER_Std_Valence": std_deviation_valence,
                        "FER_Std_Arousal": std_deviation_arousal,
                        "FER_Kurtosis_Valence": kurtosis_FER_valence,
                        "FER_Kurtosis_Arousal": kurtosis_FER_arousal,
                        "FER_Max_Valence": max_valence_values,
                        "FER_Max_Arousal": max_arousal_values,
                        "FER_Min_Valence": min_valence_values,
                        "FER_Min_Arousal": min_arousal_values,
                    }
                )
    return features


def compute_features(df, filename, file_path, file_names):
    print('HERE')
    data_for_qualitySignals_FER = df[["SourceStimuliName", "Anger", "Timestamp"]]
    # data_for_qualitySignals_PupilSize = df[['SourceStimuliName', 'ET_PupilLeft', 'Timestamp']]
    # data_for_qualitySignals_HeartRate = df[['SourceStimuliName', 'IBI PPG ALG', 'Timestamp']]
    # data_for_qualitySignals_gsr = df[['SourceStimuliName', 'Tonic Signal', 'Timestamp']]

    # pupilSize_quality_signals = quality_signals_PupilSize(data_for_qualitySignals_PupilSize, filename)
    fer_quality_signals = quality_signals_FER(data_for_qualitySignals_FER, filename)
    # heartRate_quality_signals = quality_signals_HeartRate(data_for_qualitySignals_HeartRate, filename)
    # gsr_quality_signals = quality_signals_GSR(data_for_qualitySignals_gsr, filename)

    arousal_valence_pairs = output_vector_processor_path(file_path)
    all_values = arousal_valence_pairs
    emotions = list(video_emotion.keys())
    values = list(video_emotion.values())
    max_valence_values = []
    max_arousal_values = []
    min_valence_values = []
    min_arousal_values = []
    final_mean_arousal = []
    final_mean_valence = []
    std_deviation_arousal = []
    std_deviation_valence = []
    kurtosis_FER_arousal = []
    kurtosis_FER_valence = []

    for emotion in emotions:
        print(emotion)
        sum_valence = 0
        sum_arousal = 0
        total_number_valence = 0
        total_number_arosual = 0
        file_names.append(filename.split(".")[0])
        current_emotion_data = all_values[emotion]

        # Valence
        if len(current_emotion_data["valence"]) > 0:
            max_min, max_max = min(current_emotion_data["valence"]), max(
                current_emotion_data["valence"]
            )

            if max_max > abs(max_min):
                max_valence_values.append(max_max)
            else:
                max_valence_values.append(max_min)

            if all(value > 0 for value in current_emotion_data["valence"]):
                min_valence_values.append(min(current_emotion_data["valence"]))
            elif all(value < 0 for value in current_emotion_data["valence"]):
                min_valence_values.append(max(current_emotion_data["valence"]))
            else:
                min_valence_values.append(0)

            sum_valence += sum(current_emotion_data["valence"])
            total_number_valence += len(current_emotion_data["valence"])

        else:
            max_valence_values.append(0)
            min_valence_values.append(0)

        if len(current_emotion_data["valence"]) > 1:
            std_deviation_valence.append(stats.stdev(current_emotion_data["valence"]))
            kurtosis_FER_valence.append(kurtosis(current_emotion_data["valence"]))
        else:
            std_deviation_valence.append(0)
            kurtosis_FER_valence.append(0)

        # Arousal
        if len(current_emotion_data["arousal"]) > 0:
            mini, maxi = min(current_emotion_data["arousal"]), max(
                current_emotion_data["arousal"]
            )

            max_arousal_values.append(maxi)
            min_arousal_values.append(mini)

            sum_arousal += sum(current_emotion_data["arousal"])
            total_number_arosual += len(current_emotion_data["arousal"])
        else:
            max_arousal_values.append(0)
            min_arousal_values.append(0)

        if len(current_emotion_data["arousal"]) > 1:
            std_deviation_arousal.append(stats.stdev(current_emotion_data["arousal"]))
            kurtosis_FER_arousal.append(kurtosis(current_emotion_data["arousal"]))
        else:
            std_deviation_arousal.append(0)
            kurtosis_FER_arousal.append(0)

        final_mean_arousal.append(
            sum_arousal / total_number_arosual if total_number_arosual > 0 else 0
        )
        final_mean_valence.append(
            sum_valence / total_number_valence if total_number_valence > 0 else 0
        )

    features = {
        "participant": file_names,
        "video": values,
        "FER_Mean_Valence": final_mean_valence,
        "FER_Mean_Arousal": final_mean_arousal,
        "FER_Std_Valence": std_deviation_valence,
        "FER_Std_Arousal": std_deviation_arousal,
        "FER_Kurtosis_Valence": kurtosis_FER_valence,
        "FER_Kurtosis_Arousal": kurtosis_FER_arousal,
        "FER_Max_Valence": max_valence_values,
        "FER_Max_Arousal": max_arousal_values,
        "FER_Min_Valence": min_valence_values,
        "FER_Min_Arousal": min_arousal_values,
        "FER_Quality_Signals": fer_quality_signals["FER_Quality_Signals"],
        # 'Pupil_after_mean_normalize': pupil_data['after_mean_normalize'],
        # 'Pupil_after_min_normalize': pupil_data['after_min_normalize'],
        # 'Pupil_after_max_normalize': pupil_data['after_max_normalize'],
        # 'Pupil_after_skew_normalize': pupil_data['after_skew_normalize'],
        # 'Pupil_after_kurtosis_normalize': pupil_data['after_kurtosis_normalize'],
        # 'Pupil_after_std_normalize': pupil_data['after_std_normalize'],
        # 'Pupil_before_mean_normalize': pupil_data['before_mean_normalize'],
        # 'Pupil_before_min_normalize': pupil_data['before_min_normalize'],
        # 'Pupil_before_max_normalize': pupil_data['before_max_normalize'],
        # 'Pupil_before_skew_normalize': pupil_data['before_skew_normalize'],
        # 'Pupil_before_kurtosis_normalize': pupil_data['before_kurtosis_normalize'],
        # 'Pupil_before_std_normalize': pupil_data['before_std_normalize'],
        # "Pupil_Quality_Signals": pupilSize_quality_signals['PupilSize_Quality_Signals'],
        # 'ibi_after_mean_normalize': heartRate_data['ibi_after_mean_normalize'],
        # 'ibi_after_min_normalize': heartRate_data['ibi_after_min_normalize'],
        # 'ibi_after_max_normalize': heartRate_data['ibi_after_max_normalize'],
        # 'ibi_after_skew_normalize': heartRate_data['ibi_after_skew_normalize'],
        # 'ibi_after_kurtosis_normalize': heartRate_data['ibi_after_kurtosis_normalize'],
        # 'ibi_after_std_normalize': heartRate_data['ibi_after_std_normalize'],
        # 'ibi_before_mean_normalize': heartRate_data['ibi_before_mean_normalize'],
        # 'ibi_before_min_normalize': heartRate_data['ibi_before_min_normalize'],
        # 'ibi_before_max_normalize': heartRate_data['ibi_before_max_normalize'],
        # 'ibi_before_skew_normalize': heartRate_data['ibi_before_skew_normalize'],
        # 'ibi_before_kurtosis_normalize': heartRate_data['ibi_before_kurtosis_normalize'],
        # 'ibi_before_std_normalize': heartRate_data['ibi_before_std_normalize'],
        # 'heart_rate_after_mean_normalize': heartRate_data['heart_rate_after_mean_normalize'],
        # 'heart_rate_after_min_normalize': heartRate_data['heart_rate_after_min_normalize'],
        # 'heart_rate_after_max_normalize': heartRate_data['heart_rate_after_max_normalize'],
        # 'heart_rate_after_skew_normalize': heartRate_data['heart_rate_after_skew_normalize'],
        # 'heart_rate_after_kurtosis_normalize': heartRate_data['heart_rate_after_kurtosis_normalize'],
        # 'heart_rate_after_std_normalize': heartRate_data['heart_rate_after_std_normalize'],
        # 'heart_rate_before_mean_normalize': heartRate_data['heart_rate_before_mean_normalize'],
        # 'heart_rate_before_min_normalize': heartRate_data['heart_rate_before_min_normalize'],
        # 'heart_rate_before_max_normalize': heartRate_data['heart_rate_before_max_normalize'],
        # 'heart_rate_before_skew_normalize': heartRate_data['heart_rate_before_skew_normalize'],
        # 'heart_rate_before_kurtosis_normalize': heartRate_data['heart_rate_before_kurtosis_normalize'],
        # 'heart_rate_before_std_normalize': heartRate_data['heart_rate_before_std_normalize'],
        # 'ibi_after_rmssd_normalize': heartRate_data['ibi_after_rmssd_normalize'],
        # 'ibi_after_sdnn_normalize': heartRate_data['ibi_after_sdnn_normalize'],
        # 'ibi_before_rmssd_normalize': heartRate_data['ibi_before_rmssd_normalize'],
        # 'ibi_before_sdnn_normalize': heartRate_data['ibi_before_sdnn_normalize'],
        # "HeartRate_Quality_Signals": heartRate_quality_signals ['HeartRate_Quality_Signals'],
        # 'phasic_signal_after_mean_normalize': GSR_data['phasic_signal_after_mean_normalize'],
        # 'phasic_signal_after_median_normalize': GSR_data['phasic_signal_after_median_normalize'],
        # 'phasic_signal_after_min_normalize': GSR_data['phasic_signal_after_min_normalize'],
        # 'phasic_signal_after_max_normalize': GSR_data['phasic_signal_after_max_normalize'],
        # 'phasic_signal_after_skew_normalize': GSR_data['phasic_signal_after_skew_normalize'],
        # 'phasic_signal_after_kurtosis_normalize': GSR_data['phasic_signal_after_kurtosis_normalize'],
        # 'phasic_signal_after_std_normalize': GSR_data['phasic_signal_after_std_normalize'],
        # 'phasic_signal_after_variance_normalize': GSR_data['phasic_signal_after_variance_normalize'],
        # 'phasic_signal_after_mean_energy_normalize': GSR_data['phasic_signal_after_mean_energy_normalize'],
        # 'phasic_signal_after_peak_average_normalize': GSR_data['phasic_signal_after_peak_average_normalize'],
        # 'phasic_signal_after_peak_per_minute_normalize': GSR_data['phasic_signal_after_peak_per_minute_normalize'],
        # 'phasic_signal_before_mean_normalize': GSR_data['phasic_signal_before_mean_normalize'],
        # 'phasic_signal_before_median_normalize': GSR_data['phasic_signal_before_median_normalize'],
        # 'phasic_signal_before_min_normalize': GSR_data['phasic_signal_before_min_normalize'],
        # 'phasic_signal_before_max_normalize': GSR_data['phasic_signal_before_max_normalize'],
        # 'phasic_signal_before_skew_normalize': GSR_data['phasic_signal_before_skew_normalize'],
        # 'phasic_signal_before_kurtosis_normalize': GSR_data['phasic_signal_before_kurtosis_normalize'],
        # 'phasic_signal_before_std_normalize': GSR_data['phasic_signal_before_std_normalize'],
        # 'phasic_signal_before_variance_normalize': GSR_data['phasic_signal_before_variance_normalize'],
        # 'phasic_signal_before_mean_energy_normalize': GSR_data['phasic_signal_before_mean_energy_normalize'],
        # 'phasic_signal_before_peak_average_normalize': GSR_data['phasic_signal_before_peak_average_normalize'],
        # 'phasic_signal_before_peak_per_min_normalize': GSR_data['phasic_signal_before_peak_per_min_normalize'],
        # 'tonic_signal_after_mean_normalize': GSR_data['tonic_signal_after_mean_normalize'],
        # 'tonic_signal_after_median_normalize': GSR_data['tonic_signal_after_median_normalize'],
        # 'tonic_signal_after_min_normalize': GSR_data['tonic_signal_after_min_normalize'],
        # 'tonic_signal_after_max_normalize': GSR_data['tonic_signal_after_max_normalize'],
        # 'tonic_signal_after_skew_normalize': GSR_data['tonic_signal_after_skew_normalize'],
        # 'tonic_signal_after_kurtosis_normalize': GSR_data['tonic_signal_after_kurtosis_normalize'],
        # 'tonic_signal_after_std_normalize': GSR_data['tonic_signal_after_std_normalize'],
        # 'tonic_signal_after_variance_normalize': GSR_data['tonic_signal_after_variance_normalize'],
        # 'tonic_signal_after_mean_energy_normalize': GSR_data['tonic_signal_after_mean_energy_normalize'],
        # 'tonic_signal_before_mean_normalize': GSR_data['tonic_signal_before_mean_normalize'],
        # 'tonic_signal_before_median_normalize': GSR_data['tonic_signal_before_median_normalize'],
        # 'tonic_signal_before_min_normalize': GSR_data['tonic_signal_before_min_normalize'],
        # 'tonic_signal_before_max_normalize': GSR_data['tonic_signal_before_max_normalize'],
        # 'tonic_signal_before_skew_normalize': GSR_data['tonic_signal_before_skew_normalize'],
        # 'tonic_signal_before_kurtosis_normalize': GSR_data['tonic_signal_before_kurtosis_normalize'],
        # 'tonic_signal_before_std_normalize': GSR_data['tonic_signal_before_std_normalize'],
        # 'tonic_signal_before_variance_normalize': GSR_data['tonic_signal_before_variance_normalize'],
        # 'tonic_signal_before_mean_energy_normalize': GSR_data['tonic_signal_before_mean_energy_normalize'],
        # 'GSR_Quality_signals': gsr_quality_signals['GSR_Quality_Signals']
    }
    return features
