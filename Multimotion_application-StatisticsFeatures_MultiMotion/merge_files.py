import pandas as pd
from signalQualityGSR import GSRSignalQuality


def merge_file(combined_File_path, ground_truth_file_path, merge_file_path, GSR_quality_path):
    combined_file = pd.read_csv(combined_File_path)
    gsr_quality_signals = GSRSignalQuality(GSR_quality_path)['Signal-to-noise ratio (dB)']
    gsr_quality_signals.reset_index(drop=True, inplace=True)

    ground_truth_file = pd.read_csv(ground_truth_file_path)

    selected_columns_combined = combined_file.iloc[:, 2:]
    if len(selected_columns_combined) == len(gsr_quality_signals):

        merged_data = pd.concat([ground_truth_file, selected_columns_combined], axis=1)

        merged_data['GSR_SignalToNoiseRatio'] = gsr_quality_signals
    else:
        print("Lengths do not match. Check your data.")

    merged_data = merged_data.rename(columns={'Valence': 'Ground_Truth_valence', ' Arousal': 'Ground_Truth_arousal'})

    merged_data.to_csv(merge_file_path, index=False)
