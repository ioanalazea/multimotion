import os
import pandas as pd
import numpy as np
import chardet


def find_and_set_header(df, start_index=31, expected_header_indicator='Expected Header Indicator'):
    header = None
    for i in range(start_index, len(df)):
        row = df.iloc[i]
        #take the respondent name from the file
        respondent_name = df.iloc[1,1]
        
        if row[0] == expected_header_indicator:  # Check the first cell
            header = row
            df.columns = header  # Set this row as the header
            df = df[i+1:].reset_index(drop=True)  # Slice the DataFrame from the next row
            break
    return df, header, respondent_name

def imputation_files(directory_path, output_path):
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            print(f"Processing file: {filename}")

            # Detect file encoding
            with open(file_path, "rb") as f:
                result = chardet.detect(f.read(100000))  # Read a portion of the file
                encoding = result["encoding"]

          
            df = pd.read_csv(file_path, encoding=encoding)
      
            filename = filename.split(".")[0]
            df, header, respondent_name = find_and_set_header(df, 0, "Row")
            
            df['Anger'] = pd.to_numeric(df['Anger'])
            df['Contempt'] = pd.to_numeric(df['Contempt'])
            df["Disgust"] = pd.to_numeric(df["Disgust"])
            df["Fear"] = pd.to_numeric(df["Fear"])
            df["Joy"] = pd.to_numeric(df["Joy"])
            df["Sadness"] = pd.to_numeric(df["Sadness"])
            df["Surprise"] = pd.to_numeric(df["Surprise"])
            # df['Heart Rate PPG ALG'] = pd.to_numeric(df['Heart Rate PPG ALG'])
            # df['IBI PPG ALG'] = pd.to_numeric(df['IBI PPG ALG'])
            df['ET_PupilLeft'] = pd.to_numeric(df['ET_PupilLeft'])
            df['ET_PupilRight'] = pd.to_numeric(df['ET_PupilRight'])
            df['ET_GazeLeftx'] = pd.to_numeric(df['ET_GazeLeftx'])
            df['ET_GazeLefty'] = pd.to_numeric(df['ET_GazeLefty'])
            # df['Phasic Signal'] = pd.to_numeric(df['Phasic Signal'])
            # df['Tonic Signal'] = pd.to_numeric(df['Tonic Signal'])

            # df['IBI PPG ALG'] = pd.to_numeric(df['IBI PPG ALG'], errors='coerce')
            # df['Heart Rate PPG ALG'].replace(-1, np.nan, inplace=True)
            # df['IBI PPG ALG'].replace(-1, np.nan, inplace=True)
            df['ET_PupilLeft'].replace(-1, np.nan, inplace=True)
            df['ET_PupilRight'].replace(-1, np.nan, inplace=True)
            df['ET_GazeLeftx'].replace(-1, np.nan, inplace=True)
            df['ET_GazeLefty'].replace(-1, np.nan, inplace=True)
            # df['Phasic Signal'].replace(-1, np.nan, inplace=True)
            # df['Tonic Signal'].replace(-1, np.nan, inplace=True)

            first_valid_index = df['Anger'].first_valid_index()
            df.at[first_valid_index, 'Anger'] = np.nan
            first_valid_index = df['Contempt'].first_valid_index()
            df.at[first_valid_index, 'Contempt'] = np.nan
            first_valid_index = df['Disgust'].first_valid_index()
            df.at[first_valid_index, 'Disgust'] = np.nan
            first_valid_index = df['Fear'].first_valid_index()
            df.at[first_valid_index, 'Fear'] = np.nan
            first_valid_index = df['Joy'].first_valid_index()
            df.at[first_valid_index, 'Joy'] = np.nan
            first_valid_index = df['Sadness'].first_valid_index()
            df.at[first_valid_index, 'Sadness'] = np.nan
            first_valid_index = df['Surprise'].first_valid_index()
            df.at[first_valid_index, 'Surprise'] = np.nan

            # first_valid_index = df['Heart Rate PPG ALG'].first_valid_index()
            # df.at[first_valid_index, 'Heart Rate PPG ALG'] = np.nan

            df['Anger'] = df['Anger'].interpolate(method='linear')
            df['Contempt'] = df['Contempt'].interpolate(method='linear')
            df['Disgust'] = df['Disgust'].interpolate(method='linear')
            df['Fear'] = df['Fear'].interpolate(method='linear')
            df['Joy'] = df['Joy'].interpolate(method='linear')
            df['Sadness'] = df['Sadness'].interpolate(method='linear')
            df['Surprise'] = df['Surprise'].interpolate(method='linear')
            # df['Heart Rate PPG ALG'] = df['Heart Rate PPG ALG'].interpolate(method='linear')
            # df['IBI PPG ALG'] = df['IBI PPG ALG'].interpolate(method='linear')
            df['ET_PupilLeft'].fillna(df['ET_PupilLeft'].mean(), inplace=True)
            df['ET_PupilRight'].fillna(df['ET_PupilRight'].mean(), inplace=True)
            df['ET_GazeLeftx'].fillna(df['ET_GazeLeftx'].mean(), inplace=True)
            df['ET_GazeLefty'].fillna(df['ET_GazeLefty'].mean(), inplace=True)
            # df['Phasic Signal'].fillna(df['Phasic Signal'].mean(), inplace=True)
            # df['Tonic Signal'].fillna(df['Tonic Signal'].mean(), inplace=True)
            df['respondent_name'] = respondent_name
            columns_to_keep = [
                'Timestamp', 'SourceStimuliName','Anger', 'Contempt', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise',
                # 'Heart Rate PPG ALG', 'IBI PPG ALG',
                'ET_PupilLeft', 'ET_PupilRight',
                'ET_GazeLeftx','ET_GazeLefty',
                'respondent_name'
                # 'Phasic Signal', 'Tonic Signal'
            ]
            df = df[columns_to_keep]

            output_file_path = output_path + rf"{filename}.csv"
            df.to_csv(output_file_path, index=False)

