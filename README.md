# Hand_Sign_Detection

Summer Project 2024

A real-time system that uses Mediapipe and a Random Forest Classifier to detect and classify hand signs (A-Z) from live webcam input. It extracts hand landmarks, preprocesses data, and predicts gestures with accurate visual feedback. Ideal for gesture-based applications and assistive technologies.

Install dependencies using pip: 
pip install opencv-python mediapipe scikit-learn matplotlib numpy

Workflow :
1. Data Collection (collect_images.py)
Captures hand gesture images from webcam.

Press Q to begin image collection for each class.

Stores 100 images for each of the 26 classes (A-Z) in separate directories under ./data/.

2. Data Preprocessing (preprocess_data.py)
Uses Mediapipe Hands to extract 21 hand landmarks (x, y coordinates).

Normalizes coordinates relative to the bounding box.

Saves preprocessed data and labels in a data.pickle file.

3. Model Training (train_model.py)
Loads landmark data and labels.

Ensures consistent feature vector length (padding/truncating).

Trains a Random Forest Classifier using scikit-learn.

Evaluates accuracy and saves the trained model as model.p.

4. Real-Time Prediction (predict_realtime.py)
Captures webcam feed.

Detects hand landmarks using Mediapipe.

Extracts and normalizes features.

Loads trained model to predict hand gestures in real-time.

Displays predictions on the screen with bounding box overlays.

Label Mapping
Each class number (0–25) corresponds to an uppercase English alphabet letter as shown below:

Label	Character
0	A
1	B
2	C
3	D
4	E
5	F
6	G
7	H
8	I
9	J
10	K
11	L
12	M
13	N
14	O
15	P
16	Q
17	R
18	S
19	T
20	U
21	V
22	W
23	X
24	Y
25	Z

You can improve this by:

1. Collecting more data per class

2. Using a deep learning model (e.g., CNN or RNN)

Concepts and Tools Used :

1. OpenCV: Video capture, image handling

2. Mediapipe: Robust hand landmark detection

3. scikit-learn: Model training and evaluation

4. Random Forest: Lightweight and interpretable classification algorithm

5. Pickle: Saving and loading models/data

Acknowledgements : 

1. Mediapipe by Google
   
2. scikit-learn
