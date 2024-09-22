import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold
import itertools
import time
from sklearn.model_selection import train_test_split


# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# Reshape and normalize the data
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

# Sample 100 images for faster execution
x_train, x_sample, y_train, y_sample = train_test_split(x_train, y_train, train_size=500, random_state=42)

# Initialize KFold for cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Parameters for analysis
n_estimators_range = [10, 50, 100]  # Number of weak learners
max_depth_range = [2, 4, 6]          # Depth of each tree

# Store results
train_accuracies = []
mean_cv_scores = []
train_error_rates = []
cv_error_rates = []

# Loop through both n_estimators and max_depth
for max_depth in max_depth_range:
    for n_estimators in n_estimators_range:
        start_time = time.time()
        model = XGBClassifier(n_estimators=n_estimators, learning_rate=0.1, max_depth=max_depth, random_state=42)

        # Train the model
        model.fit(x_train, y_train.ravel())

        # Predictions and evaluation for the training set
        y_train_pred = model.predict(x_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)

        # Cross-validation
        cv_scores = cross_val_score(model, x_train, y_train.ravel(), cv=kf, scoring='accuracy')
        mean_cv_score = cv_scores.mean()

        # Error rate (1 - Accuracy)
        train_error_rate = 1 - train_accuracy
        cv_error_rate = 1 - mean_cv_score

        # Store results
        train_accuracies.append((max_depth, n_estimators, train_accuracy))
        mean_cv_scores.append((max_depth, n_estimators, mean_cv_score))
        train_error_rates.append((max_depth, n_estimators, train_error_rate))
        cv_error_rates.append((max_depth, n_estimators, cv_error_rate))

        print(f'max_depth = {max_depth}, n_estimators = {n_estimators}')
        print(f'Training Accuracy: {train_accuracy:.4f}')
        print(f'Mean Cross-Validation Accuracy: {mean_cv_score:.4f}')
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution Time: {execution_time:.4f} seconds")
        print('---')

# Plotting Accuracy Curves for all combinations of max_depth and n_estimators
plt.figure(figsize=(14, 6))
colors = itertools.cycle(plt.cm.Set1.colors)

# Combine all accuracies in a single plot with different colors
for max_depth in max_depth_range:
    color = next(colors)
    train_accs = [acc[2] for acc in train_accuracies if acc[0] == max_depth]
    cv_accs = [acc[2] for acc in mean_cv_scores if acc[0] == max_depth]

    plt.plot(n_estimators_range, train_accs, marker='o', linestyle='-', label=f'Train Acc (max_depth={max_depth})', color=color)
    plt.plot(n_estimators_range, cv_accs, marker='o', linestyle='--', label=f'CV Acc (max_depth={max_depth})', color=color)

plt.xlabel('Number of Weak Learners (n_estimators)')
plt.ylabel('Accuracy')
plt.title('Training vs Cross-Validation Accuracy for Different max_depth Values')
plt.grid(True)

# Positioning the legend outside the plot
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()

# Plotting Error Rate Curves for all combinations of max_depth and n_estimators
plt.figure(figsize=(14, 6))
colors = itertools.cycle(plt.cm.Set1.colors)

# Combine all error rates in a single plot with different colors
for max_depth in max_depth_range:
    color = next(colors)
    train_errs = [err[2] for err in train_error_rates if err[0] == max_depth]
    cv_errs = [err[2] for err in cv_error_rates if err[0] == max_depth]

    plt.plot(n_estimators_range, train_errs, marker='o', linestyle='-', label=f'Train Error (max_depth={max_depth})', color=color)
    plt.plot(n_estimators_range, cv_errs, marker='o', linestyle='--', label=f'CV Error (max_depth={max_depth})', color=color)

plt.xlabel('Number of Weak Learners (n_estimators)')
plt.ylabel('Error Rate')
plt.title('Training vs Cross-Validation Error Rate for Different max_depth Values')
plt.grid(True)

# Positioning the legend outside the plot
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()
