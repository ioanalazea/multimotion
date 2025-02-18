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
   - Create Files folder
   - Place the downloaded files in the following directory:  
     `Files/survey_stimuli`
   - In a same way place all the data in `Files` folder
   - Like: `MyFiles`, `required_files` data inside `Files` folder.
   - Please Make sure the above steps are completed before running you code.
   - **Do not push these files to the GitHub repository** as they are too large and contains confidential data.

3. **Run `main.py`**  
   Execute the `main.py` script to start the process.

4. **Step 0: Imputation of Raw Files**  
   Compute the imputation of raw files using the appropriate option in the script.

5. **Generate Pupil CSV Files**  
   Once the imputation is complete, generate the pupil CSV files by selecting **option 5**.

6. **Compute Feature Matrix**  
   Finally, compute the feature matrix by selecting **option 1**.
