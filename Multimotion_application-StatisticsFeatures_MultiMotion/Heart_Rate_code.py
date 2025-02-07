

import pandas as pd
import numpy as np
import os

video_names = ['A1', 'A2', 'A3', 'A4', 'A', 'B', 'C', 'F', 'G', 'H', 'J', 'K', 'M', 'N', 'O', 'P', 'Q', 'U', 'V', 'W']


def compute_rmssd(intervals):
    if len(intervals) < 2:
        return 0
    nn_intervals = np.diff(intervals)
    return np.sqrt(np.mean(nn_intervals**2))


def compute_sdnn(intervals):
    if len(intervals) < 2:
        return 0
    return np.std(intervals)


def heartRate_statistics_data(df):
    fixation_data = df[df['SourceStimuliName'] == 'Fixation point']
    fixation_heart_rates = pd.to_numeric(fixation_data['Heart Rate PPG ALG']).dropna()
    fixation_ibi = pd.to_numeric(fixation_data['IBI PPG ALG']).dropna()
    fixation_rmssd = compute_rmssd(fixation_heart_rates)
    fixation_sdnn = compute_sdnn(fixation_heart_rates)


    data_frames = []

    video_differences = []

    for video_name in video_names:
        video_data = df[df['SourceStimuliName'] == video_name]
        video_heart_rates = pd.to_numeric(video_data['Heart Rate PPG ALG']).dropna()
        video_ibi = pd.to_numeric(video_data['IBI PPG ALG']).dropna()
        video_rmssd = compute_rmssd(video_ibi)
        video_sdnn = compute_sdnn(video_ibi)

        rmssd_after = (video_rmssd - fixation_rmssd) / fixation_rmssd
        sdnn_after = (video_sdnn - fixation_sdnn) / fixation_sdnn


        ibi_after_difference = (video_ibi.mean() - fixation_ibi.mean()) / fixation_ibi.mean()
        ibi_after_min_difference = (video_ibi.min() - fixation_ibi.min()) / fixation_ibi.min()
        ibi_after_max_difference = (video_ibi.max() - fixation_ibi.max()) / fixation_ibi.max()
        ibi_after_skewness = (video_ibi.skew() - fixation_ibi.skew()) / fixation_ibi.skew()
        ibi_after_kurtosis = (video_ibi.kurtosis() - fixation_ibi.kurtosis()) / fixation_ibi.kurtosis()
        ibi_after_std_dev = (video_ibi.std() - fixation_ibi.std()) / fixation_ibi.std()

        heart_rate_after_difference = (video_heart_rates.mean() - fixation_heart_rates.mean()) / fixation_heart_rates.mean()
        heart_rate_after_min_difference = (video_heart_rates.min() - fixation_heart_rates.min()) / fixation_heart_rates.min()
        heart_rate_after_max_difference = (video_heart_rates.max() - fixation_heart_rates.max()) / fixation_heart_rates.max()
        heart_rate_after_skewness = (video_heart_rates.skew() - fixation_heart_rates.skew()) / fixation_heart_rates.skew()
        heart_rate_after_kurtosis = (video_heart_rates.kurtosis() - fixation_heart_rates.kurtosis()) / fixation_heart_rates.kurtosis()
        heart_rate_after_std_dev = (video_heart_rates.std() - fixation_heart_rates.std()) / fixation_heart_rates.std()

        ibi_each_video_normalize = (video_ibi - fixation_ibi.mean()) / fixation_ibi.mean()
        heart_rates_each_video_normalize = (video_heart_rates - fixation_heart_rates.mean()) / fixation_heart_rates.mean()

        ibi_before_difference = np.mean(ibi_each_video_normalize)
        ibi_before_min_difference = np.min(ibi_each_video_normalize)
        ibi_before_max_difference = np.max(ibi_each_video_normalize)
        ibi_before_skewness = pd.Series(ibi_each_video_normalize).skew()
        ibi_before_kurtosis = pd.Series(ibi_each_video_normalize).kurtosis()
        ibi_before_std_dev = np.std(ibi_each_video_normalize)

        heart_rate_before_difference = np.mean(heart_rates_each_video_normalize)
        heart_rate_before_min_difference = np.min(heart_rates_each_video_normalize)
        heart_rate_before_max_difference = np.max(heart_rates_each_video_normalize)
        heart_rate_before_skewness = pd.Series(heart_rates_each_video_normalize).skew()
        heart_rate_before_kurtosis = pd.Series(heart_rates_each_video_normalize).kurtosis()
        heart_rate_before_std_dev = np.std(heart_rates_each_video_normalize)

        rmssd_before = compute_rmssd(ibi_each_video_normalize)
        sdnn_before = compute_sdnn(ibi_each_video_normalize)

        video_differences.append({
            'ibi_after_mean_normalize': ibi_after_difference,
            'ibi_after_min_normalize': ibi_after_min_difference,
            'ibi_after_max_normalize': ibi_after_max_difference,
            'ibi_after_skew_normalize': ibi_after_skewness,
            'ibi_after_kurtosis_normalize': ibi_after_kurtosis,
            'ibi_after_std_normalize': ibi_after_std_dev,
            'ibi_before_mean_normalize': ibi_before_difference,
            'ibi_before_min_normalize': ibi_before_min_difference,
            'ibi_before_max_normalize': ibi_before_max_difference,
            'ibi_before_skew_normalize': ibi_before_skewness,
            'ibi_before_kurtosis_normalize': ibi_before_kurtosis,
            'ibi_before_std_normalize': ibi_before_std_dev,

            'heart_rate_after_mean_normalize': heart_rate_after_difference,
            'heart_rate_after_min_normalize': heart_rate_after_min_difference,
            'heart_rate_after_max_normalize': heart_rate_after_max_difference,
            'heart_rate_after_skew_normalize': heart_rate_after_skewness,
            'heart_rate_after_kurtosis_normalize': heart_rate_after_kurtosis,
            'heart_rate_after_std_normalize': heart_rate_after_std_dev,
            'heart_rate_before_mean_normalize': heart_rate_before_difference,
            'heart_rate_before_min_normalize': heart_rate_before_min_difference,
            'heart_rate_before_max_normalize': heart_rate_before_max_difference,
            'heart_rate_before_skew_normalize': heart_rate_before_skewness,
            'heart_rate_before_kurtosis_normalize': heart_rate_before_kurtosis,
            'heart_rate_before_std_normalize': heart_rate_before_std_dev,

            'ibi_after_rmssd_normalize': rmssd_after,
            'ibi_after_sdnn_normalize': sdnn_after,
            'ibi_before_rmssd_normalize': rmssd_before,
            'ibi_before_sdnn_normalize': sdnn_before,
        })

    result_df = pd.DataFrame(video_differences)
    # result_df['video name'] = result_df['video name'].replace(column_name_mapping)
    data_frames.append(result_df)

    final_result = pd.concat(data_frames, ignore_index=True)
    return final_result

