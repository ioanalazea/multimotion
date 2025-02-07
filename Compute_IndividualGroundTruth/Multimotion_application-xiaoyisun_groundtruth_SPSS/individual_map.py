# -*- coding: utf-8 -*-
# @Time    : 2023/8/2 16:27
# @Author  : Xiaoyi Sun
# @Site    : 
# @File    : individual_map.py
# @Software: PyCharm


import numpy as np
import pandas as pd


def process_and_save_data(output_file, group_space, subject_weights):
    # Create empty array to store individual maps
    individual_maps = np.empty((subject_weights.shape[0], group_space.shape[0], group_space.shape[1]))

    # Calculate individual maps
    for i in range(subject_weights.shape[0]):
        individual_maps[i] = group_space * subject_weights[i]

    # Save the individual maps to CSV
    header = "Valence, Arousal"
    np.savetxt(output_file, individual_maps.reshape(-1, 2), delimiter=",", header=header, comments="", fmt="%.6f")

    # Load the CSV file
    data = pd.read_csv(output_file)

    # Calculate participant numbers
    num_participants = data.shape[0] // 20
    participant_nums = []

    for i in range(num_participants):
        participant_nums += [i + 1] * 20

    # Add participant numbers as a new column
    data.insert(0, 'Participant', participant_nums)

    names = ["A1_LP", "A2_LP", "A3_LP", "A4_LP", "A_HN", "B_HN", "C_LN", "F_HN", "G_HP", "H_HP",
             "J_Ne", "K_Ne", "M_LN", "N_LN", "O_LN", "P_HP", "Q_HP", "U_Ne", "V_Ne", "W_HN"]

    # Create a new column with repeated values from "Stimulus Name"
    data["Stimulus_Name"] = names * num_participants
    new_column_order_index = [0, 3, 1, 2]  # Assuming the order: Stimulus_Name, Participant, Valence, Arousal

    # Use the iloc property to reorder the columns based on their index positions
    data = data.iloc[:, new_column_order_index]
    # Save the modified data to the same CSV file
    data.to_csv(output_file, index=False)


