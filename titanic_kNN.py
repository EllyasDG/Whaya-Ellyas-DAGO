import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. Data Cleaning
train_df = pd.read_csv('C:/Users/HP/OneDrive/Bureau/Ellyas/IMT_Atlantique/3A/GTMetz/Cours/ML/HW1/titanic/train.csv')

train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
train_df['Fare'] = train_df['Fare'].fillna(train_df['Fare'].median())
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])

# Encoding categorical variables
label_encoder = LabelEncoder()
train_df['Sex'] = label_encoder.fit_transform(train_df['Sex'])
train_df['Embarked'] = label_encoder.fit_transform(train_df['Embarked'])

# 2. Normalization/Standardization
scaler = StandardScaler()
numeric_features = ['Age', 'Fare']
train_df[numeric_features] = scaler.fit_transform(train_df[numeric_features])

# 3. Data Splitting
X = train_df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
y = train_df['Survived']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Training with k-Nearest Neighbors (k-NN)

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

k_values = list(range(1, 21))  # Values of k from 1 to 20
train_error_rates = []
val_error_rates = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Cross-validation on the training set
    cv_scores = cross_val_score(knn, X_train, y_train, cv=kf, scoring='accuracy')
    train_error_rate = 1 - cv_scores.mean()  # Error rate is 1 - accuracy
    train_error_rates.append(train_error_rate)
    
    # Training and evaluating on the validation set
    knn.fit(X_train, y_train)
    y_val_pred = knn.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_error_rate = 1 - val_accuracy  # Error rate is 1 - accuracy
    val_error_rates.append(val_error_rate)
    
    print(f'Error rate with k={k} (Cross-validation): {train_error_rate:.4f}')
    print(f'Error rate with k={k} (Validation): {val_error_rate:.4f}')

# Plot the error rates as a function of k
plt.figure(figsize=(14, 6))

# Plot training and validation error rates
plt.plot(k_values, val_error_rates, marker='o', linestyle='-', color='b', label='Validation Error Rate')
plt.plot(k_values, train_error_rates, marker='o', linestyle='-', color='r', label='Training Error Rate (Cross-validation)')
plt.xlabel('k value')
plt.ylabel('Error Rate')
plt.title('Training and Validation Error Rate vs. k')
plt.xticks(k_values)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()



import time
import numpy as np  # Import numpy to fix the error
from sklearn.model_selection import learning_curve, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Set the best k value (k=7)
best_k = 7
knn = KNeighborsClassifier(n_neighbors=best_k)

# Measure execution time for training and validation
start_time = time.time()

# Fit the model and predict
knn.fit(X_train, y_train)
y_val_pred = knn.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)

end_time = time.time()

# Calculate execution time
execution_time = end_time - start_time

print(f'Execution time for k={best_k}: {execution_time:.4f} seconds')
print(f'Validation accuracy for k={best_k}: {val_accuracy:.4f}')

# Cross-validation for k=7
cv_scores = cross_val_score(knn, X_train, y_train, cv=kf, scoring='accuracy')
cv_mean_accuracy = cv_scores.mean()
print(f'Cross-validated accuracy for k={best_k}: {cv_mean_accuracy:.4f}')

# Generate the learning curve
train_sizes, train_scores, val_scores = learning_curve(knn, X_train, y_train, cv=kf, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

# Compute the mean and standard deviation of the scores
train_scores_mean = train_scores.mean(axis=1)
val_scores_mean = val_scores.mean(axis=1)
train_scores_std = train_scores.std(axis=1)
val_scores_std = val_scores.std(axis=1)

# Plot the learning curve
plt.figure(figsize=(14, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training Accuracy')
plt.plot(train_sizes, val_scores_mean, 'o-', color='b', label='Validation Accuracy')

# Add shading to show standard deviation
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
plt.fill_between(train_sizes, val_scores_mean - val_scores_std, val_scores_mean + val_scores_std, alpha=0.1, color='b')

plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title(f'Learning Curve for k-NN (k={best_k})')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()
