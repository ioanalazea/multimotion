import pandas as pd


def best_features(final_dataFile, bestFeatures_path):
    with open('best_features.txt', 'r') as file:
        feature_list = file.readlines()

    selected_features = [feature.split(',')[0].strip() for feature in feature_list if feature.strip().endswith(',1')]
    # print(selected_features)
    final_file = pd.read_csv(final_dataFile)
    bestFeatures = final_file[selected_features]
    features_file = bestFeatures_path + 'bestFeatures.csv'

    bestFeatures.to_csv(features_file, index=False)
