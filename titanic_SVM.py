import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# 1. Data Cleaning
train_df = pd.read_csv('C:/Users/HP/OneDrive/Bureau/Ellyas/IMT_Atlantique/3A/GTMetz/Cours/ML/HW1/titanic/train.csv')

train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
train_df['Fare'] = train_df['Fare'].fillna(train_df['Fare'].median())
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])

# Encoding categorical variables
label_encoder = LabelEncoder()
train_df['Sex'] = label_encoder.fit_transform(train_df['Sex'])
train_df['Embarked'] = label_encoder.fit_transform(train_df['Embarked'])

# Normalization/Standardization
scaler = StandardScaler()
numeric_features = ['Age', 'Fare']
train_df[numeric_features] = scaler.fit_transform(train_df[numeric_features])

# Data Splitting
X = train_df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
y = train_df['Survived']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 4. Hyperparameter Tuning using GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],  # Gamma for RBF kernel
    'kernel': ['linear', 'rbf']  # Testing both linear and RBF kernels
}

# Use GridSearchCV to find the best hyperparameters
svm = SVC(random_state=42)
grid_search = GridSearchCV(svm, param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Display the best hyperparameters found
print(f"Best hyperparameters: {grid_search.best_params_}")
print(f"Best cross-validated accuracy: {grid_search.best_score_:.4f}")

# 5. Plotting the Performance in Terms of Hyperparameters
results = grid_search.cv_results_

# Extract relevant data
C_values = np.array([params['C'] for params in results['params']])
gamma_values = np.array([params['gamma'] if 'gamma' in params else 'N/A' for params in results['params']])
kernel_types = np.array([params['kernel'] for params in results['params']])
accuracy_means = results['mean_test_score']

# Create a plot
plt.figure(figsize=(10, 6))

# Plot accuracy for each kernel type and gamma
for kernel in np.unique(kernel_types):
    if kernel == 'linear':
        mask = kernel_types == kernel
        plt.plot(C_values[mask], accuracy_means[mask], marker='o', label=f'Kernel: {kernel} (gamma N/A)', linestyle='--')
    else:
        for gamma in np.unique(gamma_values[gamma_values != 'N/A']):
            mask = (kernel_types == kernel) & (gamma_values == gamma)
            plt.plot(C_values[mask], accuracy_means[mask], marker='o', label=f'Kernel: {kernel}, gamma={gamma}', linestyle='-')

plt.xscale('log')  # Logarithmic scale for C
plt.xlabel('C (Regularization parameter)')
plt.ylabel('Cross-validated Accuracy')
plt.title('SVM Performance: Accuracy vs C for Different Kernels and Gamma Values')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()



import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np

# Best hyperparameters found from GridSearchCV
best_C = 10
best_gamma = 'scale'

# Initialize the SVM with the best parameters
svm_best = SVC(C=best_C, gamma=best_gamma, kernel='rbf', random_state=42)

# Measure the execution time for training and evaluation
start_time = time.time()

# Train the model on the entire training set
svm_best.fit(X_train, y_train)

# Predict on the validation set
y_val_pred = svm_best.predict(X_val)

# Calculate accuracy and error rates
val_accuracy = accuracy_score(y_val, y_val_pred)
train_accuracy = accuracy_score(y_train, svm_best.predict(X_train))
val_error_rate = 1 - val_accuracy
train_error_rate = 1 - train_accuracy

# Perform cross-validation with 5-fold cross-validation
cv_scores = cross_val_score(svm_best, X_train, y_train, cv=5, scoring='accuracy')
cv_error_rate = 1 - np.mean(cv_scores)

end_time = time.time()
execution_time = end_time - start_time

print(f'Execution time for SVM with C={best_C} and gamma={best_gamma}: {execution_time:.4f} seconds')
print(f'Training accuracy: {train_accuracy:.4f}, Validation accuracy: {val_accuracy:.4f}')
print(f'Training error rate: {train_error_rate:.4f}, Validation error rate: {val_error_rate:.4f}')
print(f'Cross-validation error rate: {cv_error_rate:.4f}')

# Plotting the error rates
plt.figure(figsize=(10, 6))
error_rates = [train_error_rate, val_error_rate, cv_error_rate]
labels = ['Training Error Rate', 'Validation Error Rate', 'Cross-Validation Error Rate']
plt.bar(labels, error_rates, color=['red', 'blue', 'green'])
plt.ylabel('Error Rate')
plt.title(f'Error Rates for SVM with C={best_C} and gamma={best_gamma}')
plt.ylim([0, 1])  # Set y-limit to 0-1 for better visualization
plt.grid(True)
plt.tight_layout()
plt.show()
