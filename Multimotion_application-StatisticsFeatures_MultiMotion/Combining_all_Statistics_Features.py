import os
import pandas as pd


def combining_statistics_features(path, combine_path_file):
    dfs = []
    directory_path = path
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path)
            dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    path = combine_path_file
    combined_df.to_csv(path, index=False)
