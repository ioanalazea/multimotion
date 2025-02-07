# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np

from old_exp_data_converter import convert_old_experiment
from final_exp_data_converter import convert_experiment
from combine_data import concatenate_dataframes
from distance_matrix import process_and_combine_similarities
from subject_weights_converter import process_data_and_save_to_csv
from individual_map import process_and_save_data
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler

# Transfer SPSS data to csv file, replace when needed
group_space = """\
    1      A1_LP       .9196   -.7567
    2      A2_LP      1.0982   -.2382
    3      A3_LP       .7823    .9654
    4      A4_LP      1.1107   -.2626
    5      A_HN      -1.5092    .4290
    6      B_HN      -1.7222    .7918
    7      C_LN       -.7728  -1.1605
    8      F_HN       -.7472   1.3061
    9      G_HP        .8925   1.2453
   10      H_HP       1.0100   1.3979
   11      J_Ne        .3498  -1.1371
   12      K_Ne        .2503  -1.3021
   13      M_LN      -1.0307   -.7214
   14      N_LN      -1.1942   -.4970
   15      O_LN       -.8761  -1.1255
   16      P_HP        .8871   1.3746
   17      Q_HP       1.0534   1.3605
   18      U_Ne        .5605   -.9875
   19      V_Ne        .3830  -1.0587
   20      W_HN      -1.4451    .3768
"""


subject_weights = """\
      1     .4782    .0287    .0298
      2     .2316    .7499    .2332
      3     .1800    .6381    .3835
      4     .1905    .4478    .2738
      5     .1977    .4907    .3036
      6     .1643    .6333    .3710
      7     .4827    .0112    .0118
      8     .2350    .4302    .2832
      9     .5108    .1473    .1643
     10     .0370    .5924    .2522
     11     .0643    .7627    .3111
     12     .4340    .4971    .4709
     13     .3397    .8111    .2088
     14     .0027    .7298    .3279
     15     .1037    .6458    .3432
     16     .2621    .3466    .2389
     17     .2261    .4846    .3143
     18     .3524    .7440    .1872
     19     .0052    .6429    .2877
     20     .3679    .6823    .1668
     21     .4415    .1127    .1084
     22     .1387    .5309    .2984
     23     .4112    .7675    .1726
     24     .2617    .8470    .2503
     25     .2288    .7670    .2397
     26     .2195    .5976    .3834
     27     .1476    .6855    .3908
     28     .2607    .7369    .2181
     29     .2906    .7967    .2239
     30     .3006    .3019    .2225
     31     .0274    .7497    .3240
     32     .0644    .5621    .2806
     33     .1780    .6306    .3777
     34     .3390    .7265    .1873
     35     .4859    .1738    .1835
     36     .1106    .6337    .2401
     37     .1862    .5079    .3083
     38     .1596    .5538    .3219
     39     .2112    .8323    .2678
     40     .0347    .6736    .2878
     41     .3025    .8152    .2244
     42     .4232    .7620    .1674
     43     .1454    .6768    .3845
     44     .1886    .4091    .2494
     45     .2672    .6300    .1845
     46     .1575    .5430    .3146
"""

# Create a dictionary to store repositories
repository_data = {}

# Read repositories from the text file and store them in the dictionary
with open("repository.txt", "r") as txt_file:
    for line in txt_file:
        name, url = line.strip().split(',')
        repository_data[name] = url

# Access repositories using the dictionary
repository_name_1 = "raw_data_old"
raw_data_old = repository_data[repository_name_1]

repository_name_2 = "raw_data_new"
raw_data_new = repository_data[repository_name_2]

repository_name_3 = "processed_data"
processed_data = repository_data[repository_name_3]

repository_name_4 = "old_tata"
old_data = repository_data[repository_name_4]

repository_name_5 = "new_data"
new_data = repository_data[repository_name_5]

repository_name_11 = "old_new_combined_data"
old_new_combined_data = repository_data[repository_name_11]

repository_name_6 = "distance_matrix"
distance_matrix = repository_data[repository_name_6]

repository_name_7 = "group_space_data"
group_space_data = repository_data[repository_name_7]

repository_name_8 = "subject_weights_data"
subject_weights_data = repository_data[repository_name_8]

repository_name_9 = "individual_ground_truth"
individual_ground_truth = repository_data[repository_name_9]

repository_name_10 = "updated_data"
updated_data = repository_data[repository_name_10]


while True:
    print("\nChoose a function:")
    print("1. convert old raw data from txt to csv")
    print("2. convert final raw data from txt to csv")
    print("3. merge new data to existing dataframe")
    print("4. compute distance matrix")
    print("5. convert group space and subject weights from string to csv")
    print("6. compute individual ground_truth")
    print("7. Exit")
    print("8. compute individual ground_truth maps")
    print("9. compute group space map")

    choice = input("Enter the number of the function you want to choose: ")

    if choice == "1":
        convert_old_experiment(raw_data_old, processed_data)
    elif choice == "2":
        convert_experiment(raw_data_new, processed_data)
    elif choice == "3":
        concatenate_dataframes(old_data, new_data, old_new_combined_data)
    elif choice == "4":
        process_and_combine_similarities(processed_data, distance_matrix)
    elif choice == "5":
        process_data_and_save_to_csv(group_space, group_space_data, subject_weights, subject_weights_data)
        # Read the CSV file

    elif choice == "6":
        group_space = np.genfromtxt(group_space_data, delimiter=',')
        subject_weights = np.genfromtxt(subject_weights_data, delimiter=',')
        process_and_save_data(individual_ground_truth, group_space, subject_weights)

    elif choice == "8":
        # Create a folder to save images
        if not os.path.exists('individual_plots'):
            os.makedirs('individual_plots')
        # Read data from CSV
        data = pd.read_csv(individual_ground_truth)
        # Group data by respondent
        # Group data by respondent
        grouped_data = data.groupby(data.columns[0])  # Assuming the first column is 'Participant'

        # Create scatter plots for each respondent
        for participant, group in grouped_data:
            plt.figure(figsize=(8, 6))
            plt.scatter(group.iloc[:, 2], group.iloc[:, 3], c='blue', marker='o', label='Stimuli')

            for i, row in group.iterrows():
                plt.annotate(row[1], (row[2], row[3]))  # Assuming 'Stimulus_Name' is the second column

            plt.xlabel("Valence")
            plt.ylabel("Arousal")
            plt.title(f"Valence vs. Arousal for Respondent {participant}")
            plt.axhline(0, color='black', linewidth=0.5)
            plt.axvline(0, color='black', linewidth=0.5)
            plt.xlim(-1, 1)  # Set x-axis range to -1 to 1
            plt.ylim(-1, 1)  # Set y-axis range to -1 to 1
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            # Save the plot as an image in the folder
            plt.savefig(f'individual_plots/plot_{participant}.png')

            plt.close()  # Close the figure to free up memory
    elif choice == "7":
        print("Exiting the program.")
        break
    elif choice == "9":
        # Read the CSV file
        if not os.path.exists('group_space_plot'):
            os.makedirs('group_space_plot')
        df = pd.read_csv(group_space_data, header=None)

        # Normalize the data using Min-Max scaling
        scaler = MinMaxScaler(feature_range=(-1, 1))
        normalized_data = scaler.fit_transform(df)

        # Assign names to each row
        row_names = [
            "A1_LP", "A2_LP", "A3_LP", "A4_LP", "A_HN",
            "B_HN", "C_LN", "F_HN", "G_HP", "H_HP",
            "J_Ne", "K_Ne", "M_LN", "N_LN", "O_LN",
            "P_HP", "Q_HP", "U_Ne", "V_Ne", "W_HN"
        ]

        # Create a DataFrame with normalized data and row names
        df_normalized = pd.DataFrame(normalized_data, columns=['x', 'y'])
        df_normalized['names'] = row_names

        # Plot the normalized data points and label them
        plt.figure(figsize=(10, 8))
        plt.scatter(df_normalized['x'], df_normalized['y'], color='b', marker='o')

        for i, row in df_normalized.iterrows():
            plt.annotate(row['names'], (row['x'], row['y']), textcoords="offset points", xytext=(0, 10), ha='center')

        plt.xlabel('valence')
        plt.ylabel('arousal')
        plt.title('group_space')
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.grid(True)
        plt.savefig(f'group_space_plot/group_space_plot.png')
        plt.show()
        plt.close()  # Close the figure to free up memory

    else:
        print("Invalid choice.")


