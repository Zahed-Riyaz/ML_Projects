import numpy as np
from ANN_model import ANN 
from layer import Layer 
# Generate dataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
y = y.reshape(-1, 1)  # Reshape labels to column vector

# Normalize features
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check shapes
print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_test:", y_test.shape)

# Initialize ANN
ann = ANN()
ann.add_layer(input_size=4, num_neurons=5)  # Hidden Layer with 5 neurons
ann.add_layer(input_size=5, num_neurons=1)  # Output Layer with 1 neuron (for binary classification)

# Train ANN
ann.train(X_train,y_train)
