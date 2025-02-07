import numpy as np
import pandas as pd
import os
from scipy.signal import find_peaks

video_names = ['A1', 'A2', 'A3', 'A4', 'A', 'B', 'C', 'F', 'G', 'H', 'J', 'K', 'M', 'N', 'O', 'P', 'Q', 'U', 'V', 'W']


def GSR_statisticsFeatures(df):
    fixation_data = df[df['SourceStimuliName'] == 'Fixation point']
    timestamp = fixation_data['Timestamp']
    fixation_phasic_signal = pd.to_numeric(fixation_data['Phasic Signal']).dropna()
    fixation_tonic_signal = pd.to_numeric(fixation_data['Tonic Signal']).dropna()
    fixation_phasic_signal_peakCount = len(find_peaks(fixation_phasic_signal)[0])
    result_dict = {'peak_average': lambda x: np.mean(x) if len(x) > 0 else 0}
    fixation_phasic_signal_peakAverage = result_dict['peak_average'](fixation_phasic_signal)

    aggregation_functions = {
        'Phasic_duration': ('Timestamp', lambda x: x.iloc[-1] - x.iloc[0]),
    }
    fixation_phasic_duration = aggregation_functions['Phasic_duration'][1](timestamp)
    fixation_phasic_peak_per_minute = fixation_phasic_signal_peakCount / fixation_phasic_duration

    data_frames = []

    video_differences = []

    for video_name in video_names:
        video_data = df[df['SourceStimuliName'] == video_name]
        video_phasic_signal = pd.to_numeric(video_data['Phasic Signal']).dropna()
        video_tonic_signal = pd.to_numeric(video_data['Tonic Signal']).dropna()
        video_phasic_signal_peak_count = len(find_peaks(video_phasic_signal)[0])

        peakAverageDict = {'peak_average': lambda x: np.mean(x) if len(x) > 0 else 0}
        video_phasic_signal_peak_average = peakAverageDict['peak_average'](video_phasic_signal)

        video_aggregare_func = {
            'video_phasic_duration': ('Timestamp', lambda x: x.iloc[-1] - x.iloc[0]),
        }
        video_phasic_signal_duration = video_aggregare_func['video_phasic_duration'][1](timestamp)
        video_phasic_signal_peak_per_minute = video_phasic_signal_peak_count / video_phasic_signal_duration

        phasic_signal_mean_after_difference = (
                                                          video_phasic_signal.mean() - fixation_phasic_signal.mean()) / fixation_phasic_signal.mean()
        phasic_signal_median_after_difference = (
                                                            video_phasic_signal.median() - fixation_phasic_signal.median()) / fixation_phasic_signal.median()
        phasic_signal_min_after_difference = (
                                                         video_phasic_signal.min() - fixation_phasic_signal.min()) / fixation_phasic_signal.min()
        phasic_signal_after_max_difference = (
                                                         video_phasic_signal.max() - fixation_phasic_signal.max()) / fixation_phasic_signal.max()
        phasic_signal_after_skewness = (
                                                   video_phasic_signal.skew() - fixation_phasic_signal.skew()) / fixation_phasic_signal.skew()
        phasic_signal_after_kurtosis = (
                                                   video_phasic_signal.kurtosis() - fixation_phasic_signal.kurtosis()) / fixation_phasic_signal.kurtosis()
        phasic_signal_after_std_dev = (
                                                  video_phasic_signal.std() - fixation_phasic_signal.std()) / fixation_phasic_signal.std()
        phasic_signal_after_variance = (
                                                   video_phasic_signal.var() - fixation_phasic_signal.var()) / fixation_phasic_signal.var()
        phasic_signal_after_mean_energy = (np.mean(np.square(video_phasic_signal)) - np.mean(
            np.square(fixation_phasic_signal))) / np.mean(np.square(fixation_phasic_signal))
        phasic_signal_after_peak_average = (
                                                       video_phasic_signal_peak_average - fixation_phasic_signal_peakAverage) / fixation_phasic_signal_peakAverage
        phasic_signal_after_peak_per_minute = (
                                                          video_phasic_signal_peak_per_minute - fixation_phasic_peak_per_minute) / fixation_phasic_peak_per_minute

        video_tonic_signal_mean_after_difference = (
                                                               video_tonic_signal.mean() - fixation_tonic_signal.mean()) / fixation_tonic_signal.mean()
        video_tonic_signal_median_after_difference = (
                                                                 video_tonic_signal.median() - fixation_tonic_signal.median()) / fixation_tonic_signal.median()
        video_tonic_signal_after_min_difference = (
                                                              video_tonic_signal.min() - fixation_tonic_signal.min()) / fixation_tonic_signal.min()
        video_tonic_signal_after_max_difference = (
                                                              video_tonic_signal.max() - fixation_tonic_signal.max()) / fixation_tonic_signal.max()
        video_tonic_signal_after_skewness = (
                                                        video_tonic_signal.skew() - fixation_tonic_signal.skew()) / fixation_tonic_signal.skew()
        video_tonic_signal_after_kurtosis = (
                                                        video_tonic_signal.kurtosis() - fixation_tonic_signal.kurtosis()) / fixation_tonic_signal.kurtosis()
        video_tonic_signal_after_std_dev = (
                                                       video_tonic_signal.std() - fixation_tonic_signal.std()) / fixation_tonic_signal.std()
        video_tonic_signal_after_variance = (
                                                        video_tonic_signal.var() - fixation_tonic_signal.var()) / fixation_tonic_signal.var()
        video_tonic_signal_after_mean_energy = (np.mean(np.square(video_tonic_signal)) - np.mean(
            np.square(fixation_tonic_signal))) / np.mean(np.square(fixation_tonic_signal))

        phasic_signal_each_video_normalize = (
                                                         video_phasic_signal - fixation_phasic_signal.mean()) / fixation_phasic_signal.mean()
        tonic_signal_each_video_normalize = (
                                                        video_tonic_signal - fixation_phasic_signal.mean()) / fixation_phasic_signal.mean()

        phasic_signal_before_mean_difference = np.mean(phasic_signal_each_video_normalize)
        phasic_signal_before_median_difference = np.median(phasic_signal_each_video_normalize)
        phasic_signal_before_min_difference = np.min(phasic_signal_each_video_normalize)
        phasic_signal_before_max_difference = np.max(phasic_signal_each_video_normalize)
        phasic_signal_before_skewness = pd.Series(phasic_signal_each_video_normalize).skew()
        phasic_signal_before_kurtosis = pd.Series(phasic_signal_each_video_normalize).kurtosis()
        phasic_signal_before_std_dev = np.std(phasic_signal_each_video_normalize)
        phasic_signal_before_variance = np.var(phasic_signal_each_video_normalize)
        phasic_signal_before_mean_energy = np.mean(np.square(phasic_signal_each_video_normalize))
        phasic_signal_before_peak_count = len(find_peaks(phasic_signal_each_video_normalize)[0])
        before_phasicPeakAverageDict = {'peak_average': lambda x: np.mean(x) if len(x) > 0 else 0}
        phasic_signal_before_peak_average = before_phasicPeakAverageDict['peak_average'](
            phasic_signal_each_video_normalize)

        video_before_aggregare_func = {
            'video_before_phasic_duration': ('Timestamp', lambda x: x.iloc[-1] - x.iloc[0]),
        }
        before_phasic_signal_duration = video_before_aggregare_func['video_before_phasic_duration'][1](timestamp)
        phasic_signal_before_peak_per_minute = phasic_signal_before_peak_count / before_phasic_signal_duration

        tonic_signal_before_mean_difference = np.mean(tonic_signal_each_video_normalize)
        tonic_signal_before_median_difference = np.median(tonic_signal_each_video_normalize)
        tonic_signal_before_min_difference = np.min(tonic_signal_each_video_normalize)
        tonic_signal_before_max_difference = np.max(tonic_signal_each_video_normalize)
        tonic_signal_before_skewness = pd.Series(tonic_signal_each_video_normalize).skew()
        tonic_signal_before_kurtosis = pd.Series(tonic_signal_each_video_normalize).kurtosis()
        tonic_signal_before_std_dev = np.std(tonic_signal_each_video_normalize)
        tonic_signal_before_variance = np.var(tonic_signal_each_video_normalize)
        tonic_signal_before_mean_energy = np.mean(np.square(tonic_signal_each_video_normalize))

        video_differences.append({
            'phasic_signal_after_mean_normalize': phasic_signal_mean_after_difference,
            'phasic_signal_after_median_normalize': phasic_signal_median_after_difference,
            'phasic_signal_after_min_normalize': phasic_signal_min_after_difference,
            'phasic_signal_after_max_normalize': phasic_signal_after_max_difference,
            'phasic_signal_after_skew_normalize': phasic_signal_after_skewness,
            'phasic_signal_after_kurtosis_normalize': phasic_signal_after_kurtosis,
            'phasic_signal_after_std_normalize': phasic_signal_after_std_dev,
            'phasic_signal_after_variance_normalize': phasic_signal_after_variance,
            'phasic_signal_after_mean_energy_normalize': phasic_signal_after_mean_energy,
            'phasic_signal_after_peak_average_normalize': phasic_signal_after_peak_average,
            'phasic_signal_after_peak_per_minute_normalize': phasic_signal_after_peak_per_minute,

            'phasic_signal_before_mean_normalize': phasic_signal_before_mean_difference,
            'phasic_signal_before_median_normalize': phasic_signal_before_median_difference,
            'phasic_signal_before_min_normalize': phasic_signal_before_min_difference,
            'phasic_signal_before_max_normalize': phasic_signal_before_max_difference,
            'phasic_signal_before_skew_normalize': phasic_signal_before_skewness,
            'phasic_signal_before_kurtosis_normalize': phasic_signal_before_kurtosis,
            'phasic_signal_before_std_normalize': phasic_signal_before_std_dev,
            'phasic_signal_before_variance_normalize': phasic_signal_before_variance,
            'phasic_signal_before_mean_energy_normalize': phasic_signal_before_mean_energy,
            'phasic_signal_before_peak_average_normalize': phasic_signal_before_peak_average,
            'phasic_signal_before_peak_per_min_normalize': phasic_signal_before_peak_per_minute,

            'tonic_signal_after_mean_normalize': video_tonic_signal_mean_after_difference,
            'tonic_signal_after_median_normalize': video_tonic_signal_median_after_difference,
            'tonic_signal_after_min_normalize': video_tonic_signal_after_min_difference,
            'tonic_signal_after_max_normalize': video_tonic_signal_after_max_difference,
            'tonic_signal_after_skew_normalize': video_tonic_signal_after_skewness,
            'tonic_signal_after_kurtosis_normalize': video_tonic_signal_after_kurtosis,
            'tonic_signal_after_std_normalize': video_tonic_signal_after_std_dev,
            'tonic_signal_after_variance_normalize': video_tonic_signal_after_variance,
            'tonic_signal_after_mean_energy_normalize': video_tonic_signal_after_mean_energy,

            'tonic_signal_before_mean_normalize': tonic_signal_before_mean_difference,
            'tonic_signal_before_median_normalize': tonic_signal_before_median_difference,
            'tonic_signal_before_min_normalize': tonic_signal_before_min_difference,
            'tonic_signal_before_max_normalize': tonic_signal_before_max_difference,
            'tonic_signal_before_skew_normalize': tonic_signal_before_skewness,
            'tonic_signal_before_kurtosis_normalize': tonic_signal_before_kurtosis,
            'tonic_signal_before_std_normalize': tonic_signal_before_std_dev,
            'tonic_signal_before_variance_normalize': tonic_signal_before_variance,
            'tonic_signal_before_mean_energy_normalize': tonic_signal_before_mean_energy
        })

    result_df = pd.DataFrame(video_differences)
    # result_df['video name'] = result_df['video name'].replace(column_name_mapping)
    data_frames.append(result_df)

    final_result = pd.concat(data_frames, ignore_index=True)
    #print(final_result)
    return final_result

