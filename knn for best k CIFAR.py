import numpy as np
from tensorflow.keras import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# Reshape the data to be 2D (1D vectors)
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# Normalize the data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Select the best k from previous cross-validation results
best_k = 4  # Use the best k you identified from your previous experiments

# Training set sizes (percentages of the full training set, from 10% to 90%)
train_sizes = np.linspace(0.1, 0.9, 9)  # Ensuring train_size is in (0, 1) range

# Lists to store accuracy for each training set size
train_accuracies = []
test_accuracies = []

for size in train_sizes:
    # Determine the number of samples to use for this size
    x_train_subset, _, y_train_subset, _ = train_test_split(
        x_train, y_train, train_size=size, random_state=42
    )

    # Initialize k-NN with the best k
    knn = KNeighborsClassifier(n_neighbors=best_k)
    
    # Fit the model on the subset of the training data
    knn.fit(x_train_subset, y_train_subset.ravel())

    # Evaluate accuracy on the training subset
    y_train_pred = knn.predict(x_train_subset)
    train_accuracy = accuracy_score(y_train_subset, y_train_pred)
    train_accuracies.append(train_accuracy)
    
    # Evaluate accuracy on the full test set
    y_test_pred = knn.predict(x_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_accuracies.append(test_accuracy)

    print(f'Training size: {int(size * 100)}% - Train Accuracy: {train_accuracy:.4f} - Test Accuracy: {test_accuracy:.4f}')

# Plot learning curves (accuracy vs. training set size)
plt.figure(figsize=(10, 5))

# Plot training accuracy
plt.plot(train_sizes * 100, train_accuracies, marker='o', linestyle='-', color='r', label='Training Accuracy')
# Plot test accuracy
plt.plot(train_sizes * 100, test_accuracies, marker='o', linestyle='-', color='b', label='Test Accuracy')

plt.xlabel('Training Set Size (%)')
plt.ylabel('Accuracy')
plt.title(f'Accuracy vs. Training Set Size (k={best_k})')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
