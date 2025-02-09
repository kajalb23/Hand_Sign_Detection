import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data from pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Inspect data structure for debugging
print(f"Type of data_dict['data']: {type(data_dict['data'])}")
print(f"Type of data_dict['labels']: {type(data_dict['labels'])}")
print(f"Number of samples in data: {len(data_dict['data'])}")

# Check each sample in data for consistency
max_length = 0
for i, sample in enumerate(data_dict['data']):
    print(f"Sample {i}: {type(sample)}, shape: {np.shape(sample)}")
    max_length = max(max_length, len(sample))  # Track the maximum length

print(f"Maximum sequence length in data: {max_length}")

# Preprocess data to ensure consistent shape (pad or truncate sequences)
data = np.array([
    np.pad(sample, (0, max_length - len(sample)), mode='constant') if len(sample) < max_length else sample[:max_length]
    for sample in data_dict['data']
])

# Convert labels to a NumPy array
labels = np.array(data_dict['labels'], dtype=np.int32)

# Debug the final shapes of data and labels
print(f"Shape of data: {data.shape}")
print(f"Shape of labels: {labels.shape}")

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize and train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions
y_predict = model.predict(x_test)

# Calculate accuracy
score = accuracy_score(y_test, y_predict)
print('{}% of samples were classified correctly!'.format(score * 100))

# Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Model has been saved successfully!")
