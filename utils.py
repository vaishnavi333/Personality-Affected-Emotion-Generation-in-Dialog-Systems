# Define Constants and Utility Dictionaries


# Define VAD vectors for basic emotions
EMOTION_TO_VAD = {
    'anger': [-0.51, 0.59, 0.25],
    'disgust': [-0.60, 0.35, 0.11],
    'fear': [-0.62, 0.82, -0.43],
    'joy': [0.81, 0.51, 0.46],
    'neutral': [0.00, 0.00, 0.00],
    'sadness': [-0.63, -0.27, -0.33],
    'surprise': [0.40, 0.67, -0.13]
}

# Define mood domains in VAD space
Mood_dict = {
    # M4 is neutral
    'M4': [0.0, 0.0, 0.0],
    'M1': [1.0, 1.0, 0.0],
    'M2': [-1.0, 1.0, 0.0],
    'M3': [-1.0, -1.0, 0.0]
}

# Map emotions to mood domains
Emotion_Mood = {
    'anger': 'M2',
    'sadness': 'M3',
    'neutral': 'M4',
    'joy': 'M1',
    'surprise': 'M1',
    'fear': 'M2',
    'disgust': 'M2'
}


def get_emotion_to_VAD(emotion):
    return EMOTION_TO_VAD[emotion.lower()]

def extract_initial_mood_state_VAD(row):
    return Mood_dict[Emotion_Mood[row['emotion_1']]]

import numpy as np
def load_nrc_vad(path="NRC-VAD-Lexicon-v2.1.txt"):
    vad_dict = {}
    with open(path, 'r') as f:
        next(f)  # skip header
        for line in f:
            word, val, ar, dom = line.strip().split('\t')
            vad_dict[word.lower()] = np.array([float(val), float(ar), float(dom)])
    return vad_dict