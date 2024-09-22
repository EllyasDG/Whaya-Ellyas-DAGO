import numpy as np
from tensorflow.keras import datasets
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
import matplotlib.pyplot as plt
import time

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# Reduce the training set to only 1000 images (out of 50,000)
x_train = x_train[:1000].reshape(1000, -1)
y_train = y_train[:1000]

# Reshape the test set to 2D (1D vectors)
x_test = x_test.reshape(x_test.shape[0], -1)

# Normalize the data
x_train, x_test = x_train / 255.0, x_test / 255.0

print("Data loading and preprocessing completed.")

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Define the hyperparameters grid
param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],  # Gamma for RBF kernel
    'kernel': ['linear', 'rbf']  # Testing both linear and RBF kernels
}

# Grid search for the best hyperparameters
print("Starting Grid Search...")
svm = SVC()
grid_search = GridSearchCV(svm, param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
grid_search.fit(x_train, y_train.ravel())

# Best hyperparameters
best_params = grid_search.best_params_
best_C = best_params['C']
best_gamma = best_params['gamma']

print(f"Best hyperparameters: {best_params}")
print(f"Best cross-validated accuracy: {grid_search.best_score_:.4f}")

# Plotting the Performance in Terms of Hyperparameters
results = grid_search.cv_results_
C_values = np.array([params['C'] for params in results['params']])
gamma_values = np.array([params['gamma'] for params in results['params']])
accuracy_means = results['mean_test_score']

print("Grid search completed. Plotting hyperparameter performance...")

# Plotting the curves for each unique gamma value
plt.figure(figsize=(10, 6))
for gamma in np.unique(gamma_values):
    mask = gamma_values == gamma
    plt.plot(C_values[mask], accuracy_means[mask], marker='o', label=f'gamma={gamma}', linestyle='-')

plt.xscale('log')
plt.xlabel('C (Regularization parameter)')
plt.ylabel('Cross-validated Accuracy')
plt.title('SVM Performance: Accuracy vs C for Different Gamma Values (RBF Kernel)')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()

# Initialize the best SVM with the found hyperparameters
svm_best = SVC(C=best_C, gamma=best_gamma, kernel='rbf')

# Measure the execution time for training and evaluation
print("Training the model with the best parameters...")
start_time = time.time()

# Simulate image-by-image analysis by looping over the training set
for idx, (image, label) in enumerate(zip(x_train, y_train), 1):
    # Print progress every 100 images
    if idx % 100 == 0:
        print(f"Analyzed {idx}/1000 images")

# Fit the model on the entire training set
svm_best.fit(x_train, y_train.ravel())

# Predict on the test set
y_test_pred = svm_best.predict(x_test)

# Calculate accuracy and error rates
test_accuracy = accuracy_score(y_test, y_test_pred)
train_accuracy = accuracy_score(y_train, svm_best.predict(x_train))
test_error_rate = 1 - test_accuracy
train_error_rate = 1 - train_accuracy

# Perform cross-validation error rate
cv_scores = cross_val_score(svm_best, x_train, y_train.ravel(), cv=kf, scoring='accuracy')
cv_error_rate = 1 - np.mean(cv_scores)

end_time = time.time()
execution_time = end_time - start_time

print(f'Execution time for SVM with C={best_C} and gamma={best_gamma}: {execution_time:.4f} seconds')
print(f'Training accuracy: {train_accuracy:.4f}, Test accuracy: {test_accuracy:.4f}')
print(f'Training error rate: {train_error_rate:.4f}, Test error rate: {test_error_rate:.4f}')
print(f'Cross-validation error rate: {cv_error_rate:.4f}')

# Plotting the error rates
print("Plotting error rates...")
plt.figure(figsize=(10, 6))
error_rates = [train_error_rate, test_error_rate, cv_error_rate]
labels = ['Training Error Rate', 'Test Error Rate', 'Cross-Validation Error Rate']
plt.bar(labels, error_rates, color=['red', 'blue', 'green'])
plt.ylabel('Error Rate')
plt.title(f'Error Rates for SVM with C={best_C} and gamma={best_gamma}')
plt.ylim([0, 1])  # Set y-limit to 0-1 for better visualization
plt.grid(True)
plt.tight_layout()
plt.show()

print("Process completed.")
