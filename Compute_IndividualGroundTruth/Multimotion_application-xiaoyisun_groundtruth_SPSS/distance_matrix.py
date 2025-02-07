# -*- coding: utf-8 -*-
# @Time    : 2023/8/2 16:42
# @Author  : Xiaoyi Sun
# @Site    : 
# @File    : distance_matrix.py
# @Software: PyCharm

import pandas as pd
from scipy.spatial.distance import squareform, pdist

def process_and_combine_similarities(input_file, output_file):
    # Import the data from a CSV file
    df = pd.read_csv(input_file)

    # Define emotions for the first group of respondents (before "Aki")
    emotions_11 = ['Amused*', 'Angry*', 'Anxious*', 'Bored*', 'Calm*', 'Content*', 'Excited*', 'Fearful*',
                   'Happy*', 'Negative*', 'Positive*', 'Sad*']

    # Get unique respondents and stimuli (using drop_duplicates to ensure uniqueness)
    respondents = df['respondent'].drop_duplicates()
    stimuli = df['stimulus'].drop_duplicates()
    print(stimuli)
    # Initialize an empty dictionary to store individual similarity matrices for each respondent
    individual_similarity_matrices = {}

    # Iterate over each respondent
    for i, respondent in enumerate(respondents):
        respondent_df = df[df['respondent'] == respondent]
        emotions_to_use = emotions_11
        ratings = respondent_df[emotions_to_use]

        # Calculate the similarity matrix using Euclidean distance
        similarity_matrix = squareform(pdist(ratings, metric='euclidean'))

        # Store the similarity matrix in the dictionary with respondent's identifier as the key
        individual_similarity_matrices[respondent] = similarity_matrix

    # Create an ExcelWriter to save all the individual matrices into a single Excel file
    with pd.ExcelWriter(output_file) as writer:
        # Save individual similarity matrices to separate sheets in the Excel file
        for respondent, similarity_matrix in individual_similarity_matrices.items():
            # Create a DataFrame for the similarity matrix with proper indices and column names
            respondent_similarity_df = pd.DataFrame(similarity_matrix, index=stimuli, columns=stimuli)

            # Save the DataFrame to the Excel file with a sheet name based on the respondent
            sheet_name = f'Respondent_{respondent}'
            respondent_similarity_df.to_excel(writer, sheet_name=sheet_name)

    # Notify that the saving process is completed
    print(f"All individual similarity matrices are saved to '{output_file}' in a vertical order.")

    # Read the existing Excel file with multiple sheets
    excel_file = pd.ExcelFile(output_file)

    # Get the names of all sheets in the Excel file
    sheet_names = excel_file.sheet_names

    # Initialize an empty list to store the individual DataFrames from each sheet
    dataframes = []

    # Iterate through each sheet and read the data into individual DataFrames
    for sheet_name in sheet_names:
        df = excel_file.parse(sheet_name)
        dataframes.append(df)

    # Concatenate all individual DataFrames into a single DataFrame
    combined_similarity_df = pd.concat(dataframes, ignore_index=True)
    combined_similarity_df.drop(combined_similarity_df.columns[0], axis=1, inplace=True)

    # Save the combined DataFrame to a new sheet in the same Excel file
    with pd.ExcelWriter(output_file, mode='a', engine='openpyxl') as writer:
        combined_similarity_df.to_excel(writer, sheet_name='Combined_Sheet', index=False)

    # Notify that the saving process is completed
    print(f"All individual similarity matrices are combined and saved to '{output_file}' on a single sheet.")


