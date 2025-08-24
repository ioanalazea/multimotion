import os
import pandas as pd
import numpy as np
import chardet


def find_and_set_header(
    df, start_index=31, expected_header_indicator="Expected Header Indicator"
):
    header = None
    for i in range(start_index, len(df)):
        row = df.iloc[i]
        # take the respondent name from the file
        respondent_name = df.iloc[1, 1]

        if row[0] == expected_header_indicator:  # Check the first cell
            header = row
            df.columns = header  # Set this row as the header
            df = df[i + 1 :].reset_index(
                drop=True
            )  # Slice the DataFrame from the next row
            break
    return df, header, respondent_name


def imputation_files(directory_path, output_path):
    required_columns = [
        "Timestamp",
        "SourceStimuliName",
        "Anger",
        "Contempt",
        "Disgust",
        "Fear",
        "Joy",
        "Sadness",
        "Surprise",
        "ET_PupilLeft",
        "ET_PupilRight",
        "ET_GazeLeftx",
        "ET_GazeLefty",
    ]

    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            print(f"Processing file: {filename}")

            # Detect file encoding
            with open(file_path, "rb") as f:
                result = chardet.detect(f.read(100000))  # Read a portion of the file
                encoding = result["encoding"]

            # Read CSV
            df = pd.read_csv(file_path, encoding=encoding)

            filename = filename.split(".")[0]
            df, header, respondent_name = find_and_set_header(df, 0, "Row")

            # Ensure all required columns exist
            for col in required_columns:
                if col not in df.columns:
                    df[col] = pd.NA  # Assign NaN if missing

            # Convert necessary columns to numeric
            for col in [
                "Anger",
                "Contempt",
                "Disgust",
                "Fear",
                "Joy",
                "Sadness",
                "Surprise",
                "ET_PupilLeft",
                "ET_PupilRight",
                "ET_GazeLeftx",
                "ET_GazeLefty",
            ]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df["Anger"] = df["Anger"].interpolate(method="linear")
            df["Contempt"] = df["Contempt"].interpolate(method="linear")
            df["Disgust"] = df["Disgust"].interpolate(method="linear")
            df["Fear"] = df["Fear"].interpolate(method="linear")
            df["Joy"] = df["Joy"].interpolate(method="linear")
            df["Sadness"] = df["Sadness"].interpolate(method="linear")
            df["Surprise"] = df["Surprise"].interpolate(method="linear")

            # Add respondent_name column
            df["respondent_name"] = respondent_name

            # Keep only the required columns
            columns_to_keep = required_columns + ["respondent_name"]
            df = df[columns_to_keep]

            # Save the processed file
            output_file_path = os.path.join(output_path, f"{filename}.csv")
            df.to_csv(output_file_path, index=False)
