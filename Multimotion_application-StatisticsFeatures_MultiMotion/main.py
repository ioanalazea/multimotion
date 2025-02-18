import json
import os
from Statistics_Features import statistics_features
from Combining_all_Statistics_Features import combining_statistics_features
from merge_files import merge_file
from Imputation import imputation_files
from best_features import best_features
from final_model_pupil.emotion_data_calculation_all_participants import process_csv_files

# Load repository paths dynamically
def load_paths():
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))  # Get script directory
        json_path = os.path.join(base_path, "multimotion_file_path.json")  # Adjust for structure
        
        parent_path = os.path.dirname(base_path)  # One level up

        with open(json_path, "r") as json_file:
            repository_data = json.load(json_file)

        # Prepend the dynamically found base path to all file paths
        repository_data = {key: os.path.join(parent_path, value) for key, value in repository_data.items()}
        return repository_data
    except FileNotFoundError:
        print("Error: 'multimotion_file_path.json' not found. Please ensure it exists.")
        exit(1)
    except json.JSONDecodeError:
        print("Error: 'multimotion_file_path.json' is not properly formatted.")
        exit(1)

def main():
    repository_data = load_paths()

    print(repository_data)
    
    while True:
        print("\n" + "=" * 50)
        print("Choose a function:")
        print("0. Imputation files")
        print("1. Convert Data File into features matrix")
        print("2. Combine all features data files")
        print("3. Merge all data with Ground Truth & GSR Signals to noise ratio")
        print("4. Extract Best features")
        print("5. Extract pupil features")
        print("6. Exit")
        print("=" * 50)

        choice = input("Enter the number of the function you want to choose: ")

        if choice == "0":
            print("\nEnsure 'multimotion_file_path.json' is correctly configured:")
            print(f"  - Data Files Path: {repository_data['data_files']}")
            print(f"  - Imputed Files Output Path: {repository_data['imputation_files']}")
            
            choice_2 = input("\nPress 1 to continue, or any other number to go back: ")
            if choice_2 == "1":
                imputation_files(repository_data["data_files"], repository_data["imputation_files"])
            else:
                continue

        elif choice == "1":
            print("\nEnsure 'multimotion_file_path.json' is correctly configured:")
            print(f"  - Data Files Path: {repository_data['data_files']}")
            print(f"  - Features Output Path: {repository_data['features_files']}")

            choice_2 = input("\nPress 1 to continue, or any other number to go back: ")
            if choice_2 == "1":
                statistics_features(repository_data["data_files"], repository_data["features_files"])
            else:
                continue

        elif choice == "2":
            print("\nEnsure 'multimotion_file_path.json' is correctly configured:")
            print(f"  - Features Files Path: {repository_data['features_files']}")
            print(f"  - Combined File Output Path: {repository_data['combined_file']}")

            choice_2 = input("\nPress 1 to continue, or any other number to go back: ")
            if choice_2 == "1":
                combining_statistics_features(repository_data["features_files"], repository_data["combined_file"])
            else:
                continue

        elif choice == "3":
            print("\nEnsure 'multimotion_file_path.json' is correctly configured:")
            print(f"  - Combined Features File: {repository_data['combined_file']}")
            print(f"  - Ground Truth File: {repository_data['ground_truth_file']}")
            print(f"  - Merged File Output Path: {repository_data['merge_file']}")
            print(f"  - GSR Quality Signals Path: {repository_data['gsr_signal_noise']}")

            choice_2 = input("\nPress 1 to continue, or any other number to go back: ")
            if choice_2 == "1":
                merge_file(repository_data["combined_file"], repository_data["ground_truth_file"], 
                           repository_data["merge_file"], repository_data["gsr_signal_noise"])
            else:
                continue

        elif choice == "4":
            print("\nEnsure 'multimotion_file_path.json' is correctly configured:")
            print(f"  - Merged File Path: {repository_data['merge_file']}")
            print(f"  - Best Features Output Path: {repository_data['best_features']}")

            choice_2 = input("\nPress 1 to continue, or any other number to go back: ")
            if choice_2 == "1":
                best_features(repository_data["merge_file"], repository_data["best_features"])
            else:
                continue

        elif choice == "5":
            print("\nEnsure 'multimotion_file_path.json' is correctly configured:")
            print(f"  - Initial Data Files Path: {repository_data['data_files']}")
            print(f"  - Pupil Data Output Path: {repository_data['pupil_data']}")

            choice_2 = input("\nPress 1 to continue, or any other number to go back: ")
            if choice_2 == "1":
                process_csv_files(repository_data["data_files"], repository_data["pupil_data"])
            else:
                continue

        elif choice == "6":
            print("Exiting program...")
            break

        else:
            print("Invalid choice. Please enter a valid option.")

if __name__ == "__main__":
    main()
