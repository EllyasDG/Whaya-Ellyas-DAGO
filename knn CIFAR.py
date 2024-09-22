import numpy as np
from tensorflow.keras import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import matplotlib.pyplot as plt
import time

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# Reshape the data to be 2D (1D vectors)
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# Normalize the data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Set up cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Experiment with different k values and collect error rates
k_values = list(range(1, 11))  # Test k values from 1 to 10
train_error_rates = []
val_error_rates = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Cross-validation on the training set
    cv_scores = cross_val_score(knn, x_train, y_train.ravel(), cv=kf, scoring='accuracy')
    train_error_rate = 1 - cv_scores.mean()  # Error rate is 1 - accuracy
    train_error_rates.append(train_error_rate)
    
    # Fit the model and evaluate on the validation set
    knn.fit(x_train, y_train.ravel())
    y_val_pred = knn.predict(x_test)
    val_accuracy = accuracy_score(y_test, y_val_pred)
    val_error_rate = 1 - val_accuracy
    val_error_rates.append(val_error_rate)
    
    print(f'Error rate with k={k} (Cross-validation): {train_error_rate:.4f}')
    print(f'Error rate with k={k} (Validation): {val_error_rate:.4f}')

# Plot the error rates as a function of k
plt.figure(figsize=(10, 5))

# Plot training and validation error rates
plt.plot(k_values, train_error_rates, marker='o', linestyle='-', color='r', label='Training Error Rate (Cross-validation)')
plt.plot(k_values, val_error_rates, marker='o', linestyle='-', color='b', label='Validation Error Rate')
plt.xlabel('k value')
plt.ylabel('Error Rate')
plt.title('Training and Validation Error Rate vs. k (CIFAR-10)')
plt.xticks(k_values)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Select the best k (k with the lowest validation error rate)
best_k = k_values[np.argmin(val_error_rates)]

# Training and timing with the best k
knn_best = KNeighborsClassifier(n_neighbors=best_k)

start_time = time.time()
knn_best.fit(x_train, y_train.ravel())
y_test_pred = knn_best.predict(x_test)
end_time = time.time()

# Evaluate accuracy and error rate with the best k
test_accuracy = accuracy_score(y_test, y_test_pred)
test_error_rate = 1 - test_accuracy
execution_time = end_time - start_time

print(f'\nBest k: {best_k}')
print(f'Test accuracy with k={best_k}: {test_accuracy:.4f}')
print(f'Test error rate with k={best_k}: {test_error_rate:.4f}')
print(f'Execution time for k={best_k}: {execution_time:.2f} seconds')

# Plot the error rate for the best k
plt.figure(figsize=(6, 4))
plt.bar(['Test Error Rate'], [test_error_rate], color='blue')
plt.ylabel('Error Rate')
plt.title(f'Test Error Rate with Best k ({best_k})')
plt.tight_layout()
plt.show()
