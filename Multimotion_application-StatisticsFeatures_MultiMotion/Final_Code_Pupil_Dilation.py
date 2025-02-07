import pandas as pd
import numpy as np
import os


def pupil_statistics_data(df):
    video_names = ['A1', 'A2', 'A3', 'A4', 'A', 'B', 'C', 'F', 'G', 'H', 'J', 'K', 'M', 'N', 'O', 'P', 'Q', 'U', 'V',
                   'W']
    data_frames = []

    video_differences = []
    fixation_data = df[df['SourceStimuliName'] == 'Fixation point']
    fixation_data = fixation_data[(fixation_data['ET_PupilLeft'].notnull()) & (fixation_data['ET_PupilRight'].notnull())]

    if not fixation_data.empty:
        average_fixation = (fixation_data['ET_PupilLeft'] + fixation_data['ET_PupilRight']) / 2

        for video in video_names:
            video_data = df[df['SourceStimuliName'] == video]
            video_data = video_data[(video_data['ET_PupilLeft'] != -1) & (video_data['ET_PupilRight'] != -1)]
            if not video_data.empty:  # Check if there are valid rows after excluding -1 values
                video_average = (video_data['ET_PupilLeft'] + video_data['ET_PupilRight']) / 2
                each_video_normalize = (video_average - average_fixation.mean()) / average_fixation.mean()

                after_difference = (video_average.mean() - average_fixation.mean()) / average_fixation.mean()
                after_min_difference = (video_average.min() - average_fixation.min()) / average_fixation.min()
                after_max_difference = (video_average.max() - average_fixation.max()) / average_fixation.max()
                after_skewness = (video_average.skew() - average_fixation.skew()) / average_fixation.skew()
                after_kurtosis = (video_average.kurtosis() - average_fixation.kurtosis()) / average_fixation.kurtosis()
                after_std_dev = (video_average.std() - average_fixation.std()) / average_fixation.std()

                before_difference = np.mean(each_video_normalize)
                before_min_difference = np.min(each_video_normalize)
                before_max_difference = np.max(each_video_normalize)
                before_skewness = pd.Series(each_video_normalize).skew()
                before_kurtosis = pd.Series(each_video_normalize).kurtosis()
                before_std_dev = np.std(each_video_normalize)

            video_differences.append({
                'after_mean_normalize': after_difference,
                'after_min_normalize': after_min_difference,
                'after_max_normalize': after_max_difference,
                'after_skew_normalize': after_skewness,
                'after_kurtosis_normalize': after_kurtosis,
                'after_std_normalize': after_std_dev,
                'before_mean_normalize': before_difference,
                'before_min_normalize': before_min_difference,
                'before_max_normalize': before_max_difference,
                'before_skew_normalize': before_skewness,
                'before_kurtosis_normalize': before_kurtosis,
                'before_std_normalize': before_std_dev
            })

        result_df = pd.DataFrame(video_differences)
        data_frames.append(result_df)

    final_result = pd.concat(data_frames, ignore_index=True)
    return final_result

# # To use this function:
# directory_path = r'C:\Users\440\PycharmProjects\pythonProject3\dataset'
# result = process_data(directory_path)
# result.to_csv('merged_results.csv', index=False)
