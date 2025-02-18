# Introduction  
Neuromarketing combines neuroscience, psychology, AI, and marketing to study consumer behavior and decision-making. It uses advanced technologies like EEG, HRV, eye-tracking, and facial expression analysis to measure emotional and cognitive responses, overcoming the limitations of traditional methods like surveys. AI and machine learning, particularly deep learning models like CNNs, play a key role in automating emotion and attention detection from multimodal data (e.g., EEG, ECG). This project, a collaboration between CSEE, the Department of Psychology, and Essex Business School, aims to develop an AI-based tool for emotion recognition and attention detection to enhance neuromarketing research.

# Objective  
The primary objective is to develop an AI-based application capable of detecting and classifying emotions and attention levels using physiological and neurological data. Specific goals include:  
- Designing an emotion and attention detection system.  
- Leveraging multimodal data (e.g., EEG, ECG) for accurate classification.  
- Supporting neuromarketing research and clinical psychology applications. 

# Steps to Set Up and Run the Project

1. **Clone the Repository**  
   Clone the repository to your local machine.

2. **Download Survey Stimuli**  
   - Download the `survey_stimuli` folder from the Box link:  
     [Box Link](https://essexuniversity.app.box.com/folder/292287311065?box_action=go_to_item&box_source=legacy-notify_existing_collab_folder&s=g1xdjz8aucerskphjjuhmzeb0oobeujv)  
   - Place the downloaded files in the following directory:  
     `final_model_pupil/required_files/survey_stimuli`  
   - **Do not push these files to the GitHub repository** as they are too large.

3. **Change Paths**  
   Update the following paths in the specified files:  
   - **`repository.txt`**: Adjust the paths to match the locations of the files on your system.  
   - **`emotion_data_calculation_all_participants.py`**:  
     Change the `video_folder` to the location of your `survey_stimuli`.  
   - **`interval_decesion_for_all_participants_split_video.py`**:  
     Update the following paths:  
     - `home_dir`  
     - `relative_path`  
     - `interval_path`  
     - `gt_path`  

4. **Run `main.py`**  
   Execute the `main.py` script to start the process.

5. **Step 0: Imputation of Raw Files**  
   Compute the imputation of raw files using the appropriate option in the script.

6. **Generate Pupil CSV Files**  
   Once the imputation is complete, generate the pupil CSV files by selecting **option 5**.

7. **Compute Feature Matrix**  
   Finally, compute the feature matrix by selecting **option 1**.