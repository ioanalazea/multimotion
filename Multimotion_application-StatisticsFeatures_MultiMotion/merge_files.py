import pandas as pd


def merge_file(combined_File_path, ground_truth_file_path, merge_file_path, GSR_quality_path):
    combined_file = pd.read_csv(combined_File_path)

    ground_truth_file = pd.read_csv(ground_truth_file_path)

    selected_columns_combined = combined_file.iloc[:, 2:]

    merged_data = merged_data.rename(columns={'Valence': 'Ground_Truth_valence', ' Arousal': 'Ground_Truth_arousal'})

    merged_data.to_csv(merge_file_path, index=False)
