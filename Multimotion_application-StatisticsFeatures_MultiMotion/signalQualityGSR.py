import pandas as pd

video_names = ['A1', 'A2', 'A3', 'A4', 'A', 'B', 'C', 'F', 'G', 'H', 'J', 'K', 'M', 'N', 'O', 'P', 'Q', 'U', 'V', 'W']


def GSRSignalQuality(file_path):
    file = pd.read_csv(file_path)

    # Filter rows where 'Label' is in video_names list
    file = file[file['Label'].isin(video_names)]

    # Create a custom categorical data type with specified order
    custom_order = pd.CategoricalDtype(video_names, ordered=True)
    file['Label'] = file['Label'].astype(custom_order)

    file.loc[file['Label'] == 'Fixation point', 'Signal-to-noise ratio (dB)'] = file.loc[
        file['Label'] == 'Fixation point', 'Signal-to-noise ratio (dB)'].apply(lambda x: 0 if x < 3 else x)

    fixation_point_zeros = file[file['Label'] == 'Fixation point'].groupby('Respondent Name')[
        'Signal-to-noise ratio (dB)'].apply(lambda x: (x == 0).any())

    for respondent in fixation_point_zeros[fixation_point_zeros].index:
        if respondent != 'Fixation point':
            file.loc[file['Respondent Name'] == respondent, 'Signal-to-noise ratio (dB)'] = 0

    # Sort 'Label' within each 'Respondent Name' group based on video_names order
    def custom_sort(x):
        return pd.Categorical(x, categories=video_names, ordered=True)

    file['Label'] = file.groupby('Respondent Name')['Label'].transform(lambda x: custom_sort(x))

    # Convert 'Respondent Name' to categorical and retain the original order
    file['Respondent Name'] = pd.Categorical(file['Respondent Name'], categories=file['Respondent Name'].unique(),
                                             ordered=True)

    # Sort the DataFrame based on 'Respondent Name' and 'Label'
    file = file.sort_values(by=['Respondent Name', 'Label'])

    columns_to_drop = ['Gender', 'Age', 'Group', 'Start (ms)', 'Duration (ms)']
    file = file.drop(columns_to_drop, axis=1)

    return file



