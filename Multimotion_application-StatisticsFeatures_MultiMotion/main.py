import json
import os
from Statistics_Features import statistics_features
from Imputation import imputation_files
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

def display_menu():
    print("\n" + "=" * 50)
    print("Welcome to the Data Processing Tool!")
    print("Please choose a function by entering the corresponding number:\n")
    print("0. Imputation of Missing Data Files")
    print("1. Feature Extraction (FER and Pupil Arousal)")
    print("2. Merge Data with Ground Truth")
    print("3. Extract Pupil Data Features (for timestamps)")
    print("4. Exit the Program")
    print("=" * 50)

def handle_imputation(repository_data):
    print("\nImputation - Please ensure the following paths are correctly configured:")
    print(f"  - Raw Data Input Path: {repository_data['raw_files']}")
    print(f"  - Imputed Data Output Path: {repository_data['imputation_files']}")
    
    choice_2 = input("\nPress 1 to continue with imputation, or any other key to go back: ")
    if choice_2 == "1":
        imputation_files(repository_data["raw_files"], repository_data["imputation_files"])
    else:
        print("Going back to the main menu...")

def handle_feature_extraction(repository_data):
    print("\nFeature Extraction - Please ensure the following paths are correctly configured:")
    print(f"  - Data Input Path: {repository_data['data_files']}")
    print(f"  - Pupil Data Input Path (for pupil size computation): {repository_data['pupil_data']}")
    print(f"  - Features Output Path: {repository_data['features_files']}")

    choice_2 = input("\nPress 1 to continue with feature extraction, or any other key to go back: ")
    if choice_2 == "1":
        statistics_features(repository_data["data_files"], repository_data["features_files"], repository_data["pupil_data"])
    else:
        print("Going back to the main menu...")

def handle_merge_data(repository_data):
    print("\nMerge Data - Please ensure the following paths are correctly configured:")
    print(f"  - Combined Features File Path: {repository_data['combined_file']}")
    print(f"  - Ground Truth File Path: {repository_data['ground_truth_file']}")

    choice_2 = input("\nPress 1 to continue with data merging, or any other key to go back: ")
    if choice_2 == "1":
        print("Merging data with ground truth...")
    else:
        print("Going back to the main menu...")

def handle_pupil_data_extraction(repository_data):
    print("\nPupil Data Extraction - Please ensure the following paths are correctly configured:")
    print(f"  - Data Input Path: {repository_data['data_files']}")
    print(f"  - Pupil Data Output Path: {repository_data['pupil_data']}")

    choice_2 = input("\nPress 1 to continue with pupil data extraction, or any other key to go back: ")
    if choice_2 == "1":
        process_csv_files(repository_data["data_files"], repository_data["pupil_data"])
    else:
        print("Going back to the main menu...")

def main():
    repository_data = load_paths()

    while True:
        display_menu()

        choice = input("Enter the number of the function you want to choose: ")

        if choice == "0":
            handle_imputation(repository_data)
        elif choice == "1":
            handle_feature_extraction(repository_data)
        elif choice == "2":
            handle_merge_data(repository_data)
        elif choice == "3":
            handle_pupil_data_extraction(repository_data)
        elif choice == "4":
            print("Exiting the program... Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a valid option.")

if __name__ == "__main__":
    main()
