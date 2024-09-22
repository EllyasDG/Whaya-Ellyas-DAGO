import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import itertools

# 1. Data Cleaning
train_df = pd.read_csv('C:/Users/HP/OneDrive/Bureau/Ellyas/IMT_Atlantique/3A/GTMetz/Cours/ML/HW1/titanic/train.csv')

# Impute missing values
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
train_df['Fare'] = train_df['Fare'].fillna(train_df['Fare'].median())
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])

# Encode categorical variables
label_encoder = LabelEncoder()
train_df['Sex'] = label_encoder.fit_transform(train_df['Sex'])
train_df['Embarked'] = label_encoder.fit_transform(train_df['Embarked'])

# Standardize numeric features
scaler = StandardScaler()
numeric_features = ['Age', 'Fare']
train_df[numeric_features] = scaler.fit_transform(train_df[numeric_features])

# Data Split
X = train_df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
y = train_df['Survived']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Gradient Boosting with Decision Trees
n_estimators_range = [10, 50, 100,500]  # Number of weak learners (trees)
max_depth_range = [2, 4, 6]  # Reduced depth of trees for simplicity

train_accuracies = []
mean_cv_scores = []
train_error_rates = []
cv_error_rates = []

# Generate unique colors for combinations of max_depth and n_estimators
colors = itertools.cycle(plt.cm.Set1.colors)  # Use Set1 colormap for unique colors

# Loop through both n_estimators and max_depth
for max_depth in max_depth_range:
    for n_estimators in n_estimators_range:
        boosting_model = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        
        # Train the model
        boosting_model.fit(X_train, y_train)
        
        # Predictions and evaluation for the training set
        y_train_pred = boosting_model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(boosting_model, X, y, cv=kf, scoring='accuracy')
        mean_cv_score = cv_scores.mean()
        
        # Error rate (1 - Accuracy)
        train_error_rate = 1 - train_accuracy
        cv_error_rate = 1 - mean_cv_score
        
        train_accuracies.append((max_depth, n_estimators, train_accuracy))
        mean_cv_scores.append((max_depth, n_estimators, mean_cv_score))
        train_error_rates.append((max_depth, n_estimators, train_error_rate))
        cv_error_rates.append((max_depth, n_estimators, cv_error_rate))
        
        print(f'max_depth = {max_depth}, n_estimators = {n_estimators}')
        print(f'Training Accuracy: {train_accuracy:.4f}')
        print(f'Mean Cross-Validation Accuracy: {mean_cv_score:.4f}')
        print('---')

# Plotting Accuracy Curves for all combinations of max_depth and n_estimators
plt.figure(figsize=(14, 6))

# Combine all accuracies in a single plot with different colors
for max_depth in max_depth_range:
    color = next(colors)  # Get the next color for this combination
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
plt.figure(figsize=(10, 6))

# Combine all error rates in a single plot with different colors
colors = itertools.cycle(plt.cm.Set1.colors)  # Reuse the same color scheme
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





import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time

# 1. Data Cleaning
train_df = pd.read_csv('C:/Users/HP/OneDrive/Bureau/Ellyas/IMT_Atlantique/3A/GTMetz/Cours/ML/HW1/titanic/train.csv')

# Impute missing values
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
train_df['Fare'] = train_df['Fare'].fillna(train_df['Fare'].median())
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])

# Encode categorical variables
label_encoder = LabelEncoder()
train_df['Sex'] = label_encoder.fit_transform(train_df['Sex'])
train_df['Embarked'] = label_encoder.fit_transform(train_df['Embarked'])

# Standardize numeric features
scaler = StandardScaler()
numeric_features = ['Age', 'Fare']
train_df[numeric_features] = scaler.fit_transform(train_df[numeric_features])

# Data Split
X = train_df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
y = train_df['Survived']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Measure the execution time for max_depth=4, n_estimators=50
start_time = time.time()

# Gradient Boosting Model with max_depth=4 and n_estimators=50
boosting_model = GradientBoostingClassifier(n_estimators=50, max_depth=4, random_state=42)

# Train the model
boosting_model.fit(X_train, y_train)

# Predictions and evaluation for the training set
y_train_pred = boosting_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Predictions and evaluation for the validation set
y_val_pred = boosting_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)

# Measure the time taken
end_time = time.time()
execution_time = end_time - start_time

# Display the results
print(f"Execution Time for max_depth=4, n_estimators=50: {execution_time:.4f} seconds")
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")
