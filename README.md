# Hand Sign Detection
Summer Project 2024

A real-time system that uses Mediapipe and a Random Forest Classifier to detect and classify hand signs (A–Z) from live webcam input. It extracts hand landmarks, preprocesses data, and predicts gestures with accurate visual feedback.
Ideal for gesture-based interfaces, accessibility tools, and assistive technologies.

## Installation
Install the required Python libraries using pip:

!pip install opencv-python mediapipe scikit-learn matplotlib numpy
Workflow Overview
## 1. Data Collection (collect_images.py)
   1. Captures hand gesture images from the webcam.
   2. Press Q to begin image collection for each class.
   3. Stores 100 images for each of the 26 classes (A–Z) in separate directories under ./data/.

## 2. Data Preprocessing (preprocess_data.py)
   1. Uses Mediapipe Hands to extract 21 hand landmarks (x, y coordinates).
   2. Normalizes coordinates relative to the bounding box.
   3. Saves preprocessed data and labels to a file named data.pickle.

## 3. Model Training (train_model.py)
   1. Loads the landmark data and labels from data.pickle.
   2. Ensures consistent input shape using padding/truncation.
   3. Trains a Random Forest Classifier using scikit-learn.
   4. Evaluates model performance and saves the trained model as model.p.

## 4. Real-Time Prediction (predict_realtime.py)
   1. Captures live webcam feed.
   2. Detects hand landmarks using Mediapipe.
   3. Extracts and normalizes features.
   4. Loads the trained model to predict hand signs in real time.
   5. Displays predictions on-screen with bounding boxes and alphabet labels.

## Label Mapping
Each class number corresponds to an uppercase English alphabet letter:

| Label | Character |
|-------|-----------|
| 0     | A         |
| 1     | B         |
| 2     | C         |
| 3     | D         |
| 4     | E         |
| 5     | F         |
| 6     | G         |
| 7     | H         |
| 8     | I         |
| 9     | J         |
| 10    | K         |
| 11    | L         |
| 12    | M         |
| 13    | N         |
| 14    | O         |
| 15    | P         |
| 16    | Q         |
| 17    | R         |
| 18    | S         |
| 19    | T         |
| 20    | U         |
| 21    | V         |
| 22    | W         |
| 23    | X         |
| 24    | Y         |
| 25    | Z         |


## Suggestions for Improvement :
   1. Collect more data per class to improve model robustness.
   2. Use deep learning models such as CNNs or RNNs for higher accuracy.

## Tools & Concepts Used :
   1. OpenCV – For video capture and image handling
   2. Mediapipe – For real-time hand landmark detection
   3. scikit-learn – For training and evaluating machine learning models
   4. Random Forest – As the primary classification algorithm
   5. Pickle – For saving and loading models and data

## Acknowledgements :
   1. Mediapipe by Google – Hand tracking and landmark detection
   2. scikit-learn – Machine learning library used for model training and evaluation


