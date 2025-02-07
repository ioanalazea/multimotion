from Statistics_Features import statistics_features
from Combining_all_Statistics_Features import combining_statistics_features
from merge_files import merge_file
from Imputation import imputation_files
from best_features import best_features
from final_model_pupil import emotion_data_calculation_all_participants
from final_model_pupil import Normal_RGB_model as RGB_N

if __name__ == "__main__":

    repository_data = {}
    with open("repository.txt", "r") as txt_file:
        for line in txt_file:
            name, url = line.strip().split(',')
            repository_data[name] = url

    # Access repositories using the dictionary
    repository_name_1 = "data_files"
    data_files_path = repository_data[repository_name_1]

    repository_name_2 = "Features_files"
    features_files_path = repository_data[repository_name_2]

    repository_name_3 = "Combined_File"
    combined_file_path = repository_data[repository_name_3]

    repository_name_4 = "Ground_truth_File"
    Ground_truth_File = repository_data[repository_name_4]

    repository_name_5 = "merge_File"
    merge_File = repository_data[repository_name_5]

    repository_name_6 = "imputation_Files"
    imputation_Files_path = repository_data[repository_name_6]

    repository_name_7 = "GSR_SignalNoise"
    GSR_QualitySignals_path = repository_data[repository_name_7]

    repository_name_8 = "best_features"
    best_features_path = repository_data[repository_name_8]
    
    repository_name_9 = "pupil_data"
    pupil_data_path = repository_data[repository_name_9]

    while True:
        print("\nChoose a function:")
        print("0. Imputation files")
        print("1. Convert Data File into features matrix")
        print("2. Combined all features data files")
        print("3. Merge all data with Ground Truth & GSR Signals to noise ratio")
        print("4. Extract Best features")
        print("5. Extract pupil features")
        choice = input("Enter the number of the function you want to choose: ")

        if choice == "0":
            print("\n Now go to the Respository.txt file, Change the Path of Line 1, where you have all data files or imputated data files"
                  "from the imotion\n "
                  ",and Line 6, change the path where you would like to have your imputated files")
            choice_2 = input("\nPress 1 to continue, or any other number to go back to previous statement:")
            if choice_2 == "1":
                imputation_files(data_files_path, imputation_Files_path,)
            else:
                continue

        if choice == "1":
            print("\nNow go to the Respository.txt file, Change the Paths of Line 1, where you have all the data files,"
                  " and Line 2, where u want all of you files after processed, ")
                 # "\nFinally Line 7 give the path of the file of GSR Signal Quality File")
            choice_2 = input("\nPress 1 to continue, or any other number to go back to previous statement:")
            if choice_2 == "1":
                statistics_features(data_files_path, features_files_path)
            else:
                continue
        if choice == "2":
            print("\n Now go to the Respository.txt file, Change the Path of Line 2, where you have all the "
                  "statistics features data files, "
                  " \nfrom Choice 1, And Line 3 path to where you want to add your combined feature file \n")
            choice_2 = input("\nPress 1 to continue, or any other number to go back to previous statement:")
            if choice_2 == "1":
                combining_statistics_features(features_files_path, combined_file_path)
            else:
                continue

        if choice == "3":
            print("\n Now go to the Respository.txt file, Change the Path of Line 3, where you have combined features "
                  "file from Choice 2, \n "
                  "Line 4 path where you have ground truth file"
                  "\n and line 5 where want your Final data file to be."
                  "\n line 7 where your GSR quality signal data")
            choice_2 = input("\nPress 1 to continue, or any other number to go back to previous statement:")
            if choice_2 == "1":
                merge_file(combined_file_path, Ground_truth_File, merge_File, GSR_QualitySignals_path)
            else:
                continue

        if choice == "4":
            print("\n Now go to the Respository.txt file, Change the Path of Line 5, where you have your final data "
                  "file from option 3 "
                  "\n line 8, add path where you want your features file")
            choice_2 = input("\nPress 1 to continue, or any other number to go back to previous statement:")
            if choice_2 == "1":
                best_features(merge_File, best_features_path)
            else:
                continue
        if choice == "5":
            print("\n Now go to the Respository.txt file, Change the Path of Line 1, where you have your initial data "
                  "\n line 9, add path where you want your pupil data output files")
            choice_2 = input("\nPress 1 to continue, or any other number to go back to previous statement:")
            if choice_2 == "1":
                emotion_data_calculation_all_participants.process_csv_files(data_files_path, pupil_data_path)
            else:
                continue



