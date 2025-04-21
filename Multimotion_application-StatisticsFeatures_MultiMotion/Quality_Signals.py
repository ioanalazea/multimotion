import pandas as pd
import os



video_names = {
       'HN_1',      'HN_2',      'HN_4',         'HN_5',         'HN_6',        'LP_3',
        'LP_4',      'LP_6',      'LN_1',     'LN_2',      'LN_3',      'LN_4',    'LN_5',
         'LN_6',  'LN_7',      'LN_8',    'HP_1',   'HP_2', 'HP_3',  'HP_4',
         'HP_6',
         'HP_7',  'HN_3',  'HN_8',  'HN_7', 'LP_1',  'LP_2', 'LP_5',  'LP_7',
         'LP_8',   'HP_5',   'HP_8'
        }

def quality_signals_FER(df, filename):
    total_percentage = []
    file_names = []

    for video in video_names:
        file_names.append(filename.split(".")[0])

        data = df[df['SourceStimuliName'] == video][['Timestamp', 'Anger']]
        video_duration = (data['Timestamp'].max() - data['Timestamp'].min()) / 1000

        TimeStamp_each = []
        all_differences = []

        for index, row in data.iterrows():
            time, anger = row['Timestamp'], row['Anger']
            if pd.isna(anger):
                TimeStamp_each.append(time)
            else:
                TimeStamp_each.append(time)
                difference_time = (TimeStamp_each[-1] - TimeStamp_each[0]) / 1000
                all_differences.append(difference_time)
                TimeStamp_each = [time]

        sumIsGreater = []
        for time in all_differences:
            if time >= 0.5:
                sumIsGreater.append(time)

        percentage = (sum(sumIsGreater) / video_duration) * 100
        total_percentage.append(percentage)

    Quality_Signal_data = {
        "participant": file_names,
        "video": video_names,
        "FER_Quality_Signals": total_percentage,

    }
    return Quality_Signal_data

