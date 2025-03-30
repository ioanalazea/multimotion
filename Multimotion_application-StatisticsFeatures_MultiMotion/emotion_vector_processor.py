# Import necessary libraries
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the emotions for each video stimulus
video_emotion = {
      'HN_1-1' : 'HN_1',         'HN_2-1' : 'HN_2',        'HN_4-1' : 'HN_4',        'HN_5-1' : 'HN_5',        'HN_6-1' : 'HN_6',        'LP_3-1' : 'LP_3',
        'LP_4-1' : 'LP_4',       'LP_6-1' : 'LP_6',      'LN_1-1' : 'LN_1',      'LN_2-1' : 'LN_2',     'LN_3-1' : 'LN_3',     'LN_4-1' : 'LN_4',    'LN_5-1' : 'LN_5',
        'LN_6-1' : 'LN_6',  'LN_7-1' : 'LN_7',     'LN_8-1' : 'LN_8',   'HP_1-1' : 'HP_1',  'HP_2-1' : 'HP_2', 'HP_3-1' : 'HP_3', 'HP_4-1' : 'HP_4', 'HP_6-1' : 'HP_6',
        'HP_7-1' : 'HP_7', 'HN_3-1' : 'HN_3', 'HN_8-1' : 'HN_8', 'HN_7-1' : 'HN_7', 'LP_1-1' : 'LP_1',  'LP_2-1' : 'LP_2', 'LP_5-1' : 'LP_5', 'LP_7-1' : 'LP_7',
        'LP_8-1' : 'LP_8',  'HP_5-1' : 'HP_5',  'HP_8-1' : 'HP_8'

}
# Define a class for emotion vector processing
class EmotionVectorProcessor:
    # Define constants for emotion angles
    EMOTION_ANGLES = {
        "Joy": 24.50,
        "Surprise": 73.08,
        "Anger": 99.16,
        "Fear": 117.73,
        "Disgust": 138.70,
        "Sadness": 207.70,
        "Contempt": 328.49
    }

    # Constructor
    def __init__(self, data, video_emotion):
        self.data = data
        self.video_emotion = video_emotion

    # Function to convert polar coordinates to cartesian coordinates
    def polar_to_cartesian(self, magnitude, angle):
        x = magnitude * math.cos(math.radians(angle))
        y = magnitude * math.sin(math.radians(angle))
        return x, y

    # Function to add multiple vectors
    def add_vectors(self, vectors):
        x_sum = sum([vec[0] for vec in vectors])
        y_sum = sum([vec[1] for vec in vectors])
        return x_sum, y_sum

    # Function to normalize a vector
    def normalize_vector(self, x, y):
        magnitude = math.sqrt(x ** 2 + y ** 2)
        normalized_x = x / magnitude
        normalized_y = y / magnitude
        return normalized_x, normalized_y

    # Function to check if a point is inside a circle
    def is_point_in_circle(self, x, y, circle_radius=1):
        squared_distance = x ** 2 + y ** 2
        squared_radius = circle_radius ** 2

        if squared_distance < squared_radius:
            return True
        elif squared_distance == squared_radius:
            return True
        else:
            return False

    # Function to process emotions and extract valence and arousal
    def process_emotions(self, emotion_data):
        valence, arousal = [], []
        vectors = []
        columns = emotion_data.columns

        # Iterate through the emotion data and compute the emotion vectors
        for row in emotion_data.values:
            for i, intensity in enumerate(row):
                emotion = columns[i]
                angle = self.EMOTION_ANGLES[emotion]
                magnitude = float(intensity) / 100
                x, y = self.polar_to_cartesian(magnitude, angle)
                vectors.append((x, y))

            x_sum, y_sum = self.add_vectors(vectors)

            # Clip the vector components to a range of [-1, 1] and normalize the vector if it lies outside the unit circle
            # x_sum = np.clip(x_sum, -1, 1)
            # y_sum = np.clip(y_sum, -1, 1)
            if not self.is_point_in_circle(x_sum, y_sum, circle_radius=1):
                x_sum, y_sum = self.normalize_vector(x_sum, y_sum)

            # Append the valence and arousal values to their respective lists
            valence.append(y_sum)
            arousal.append(x_sum)

        return valence, arousal

    # Function to visualize the emotion vectors
    def visualize_results(self, valence, arousal, title):
        fig = plt.figure(figsize=(7, 7))
        plt.scatter(arousal, valence)
        circle = plt.Circle((0, 0), 1, edgecolor="gray", facecolor="none", linestyle="--", linewidth=0.5)
        plt.gca().add_artist(circle)
        plt.axhline(y=0, color="k", linewidth=0.5)
        plt.axvline(x=0, color="k", linewidth=0.5)
        plt.xlabel("Arousal (X)")
        plt.ylabel("Valence (Y)")
        plt.title(title)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.show()

    # Function to process the emotion data for each stimulus
    def process(self, stimulus, visualize=False):
        output = {}

        # Process the emotion data for each stimulus and extract valence and arousal pairs
        # for stimulus, emotion in self.video_emotion.items():
        extracted_group = self.data[
                ["Anger", "Contempt", "Disgust", "Fear", "Joy", "Sadness", "Surprise"]]
        extracted_group = extracted_group.copy()  # Ensure it's a copy
        extracted_group.dropna(axis=0, how="any", inplace=True)
        extracted_group.reset_index(drop=True, inplace=True)
        extracted_group = extracted_group.astype("float")
        valence, arousal = self.process_emotions(extracted_group.iloc[:, :7])
        output = {
                "valence": valence,
                "arousal": arousal
        }

            # Visualize the emotion vectors if the visualize flag is set to True
        # if visualize:
        #     self.visualize_results(valence, arousal, title=f"Emotion Vector Visualization - {stimulus} - {emotion}")

        return output
    
    def process_path(self, visualize=False):
        output = {}

        # Process the emotion data for each stimulus and extract valence and arousal pairs
        for stimulus, emotion in self.video_emotion.items():
            extracted_group = self.data[self.data.SourceStimuliName == stimulus][
                ["Anger", "Contempt", "Disgust", "Fear", "Joy", "Sadness", "Surprise"]]
            extracted_group.dropna(axis=0, how="any", inplace=True)
            extracted_group.reset_index(drop=True, inplace=True)
            extracted_group = extracted_group.astype("float")
            valence, arousal = self.process_emotions(extracted_group.iloc[:, :7])
            output[stimulus] = {
                "valence": valence,
                "arousal": arousal
            }

            # Visualize the emotion vectors if the visualize flag is set to True
            if visualize:
                self.visualize_results(valence, arousal, title=f"Emotion Vector Visualization - {stimulus} - {emotion}")

        return output


# if __name__ == "__main__":

def output_vector_processor(df, stimulus):

    # Create an instance of the EmotionVectorProcessor class and process the emotion data
    emotion_vector_processor = EmotionVectorProcessor(df, video_emotion)
    arousal_valence_pairs = emotion_vector_processor.process(stimulus, visualize=False)
    return arousal_valence_pairs


def output_vector_processor_path(path):

    # Load the emotion data from a CSV file
    df = pd.read_csv(path)

    # Create an instance of the EmotionVectorProcessor class and process the emotion data
    emotion_vector_processor = EmotionVectorProcessor(df, video_emotion)
    arousal_valence_pairs = emotion_vector_processor.process_path(visualize=False)
    return arousal_valence_pairs