import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasClassifier
import matplotlib.pyplot as plt
import time  # Import module to measure execution time
from tensorflow.keras.callbacks import EarlyStopping  # Import EarlyStopping for early stopping
import numpy as np  # This is the missing import for numpy

# Measure start time
start_time = time.time()

# Load and preprocess data
train_df = pd.read_csv('C:/Users/HP/OneDrive/Bureau/Ellyas/IMT_Atlantique/3A/GTMetz/Cours/ML/HW1/titanic/train.csv')

# Impute missing values
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
train_df['Fare'] = train_df['Fare'].fillna(train_df['Fare'].median())
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])

# Encode categorical variables
label_encoder = LabelEncoder()
train_df['Sex'] = label_encoder.fit_transform(train_df['Sex'])
train_df['Embarked'] = label_encoder.fit_transform(train_df['Embarked'])

# Normalize numeric columns
scaler = StandardScaler()
numeric_features = ['Age', 'Fare']
train_df[numeric_features] = scaler.fit_transform(train_df[numeric_features])

# Define features (X) and target (y)
X = train_df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
y = train_df['Survived']

# Function to create the neural network model
def create_model():
    model = Sequential()
    model.add(Input(shape=(X.shape[1],)))
    model.add(Dense(64, activation='relu'))
    #model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create Keras model for compatibility with Scikit-learn
keras_model = KerasClassifier(model=create_model, epochs=100, batch_size=32, verbose=0)

# Cross-validation setup
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(keras_model, X, y, cv=kf, scoring='accuracy')

# Display cross-validation results
print(f"Accuracy on each fold: {cv_scores}")
print(f"Average accuracy over 5 folds: {cv_scores.mean():.4f}")
print(f"Standard deviation of accuracy: {cv_scores.std():.4f}")

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = create_model()

# Add early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model and save history for plotting
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200, batch_size=32, callbacks=[early_stopping], verbose=0)

# Check if early stopping was triggered
if early_stopping.stopped_epoch > 0:
    print(f"Early stopping triggered after {early_stopping.stopped_epoch + 1} epochs.")
else:
    print("Early stopping was not triggered.")

# Measure end time
end_time = time.time()
execution_time = end_time - start_time

# Display execution time
print(f"Total execution time: {execution_time:.2f} seconds")

# Plot Loss and Accuracy Curves
plt.figure(figsize=(12, 5))

# Plot Training and Validation Loss
plt.subplot(1, 2, 1)
plt.plot(history.history.get('loss', []), label='Training Loss')
plt.plot(history.history.get('val_loss', []), label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plot Training and Validation Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history.get('accuracy', []), label='Training Accuracy')
plt.plot(history.history.get('val_accuracy', []), label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Additional Error/Accuracy and Learning Curve
plt.figure(figsize=(12, 5))

# Error/Accuracy Curve
plt.subplot(1, 2, 1)
plt.plot(1 - np.array(history.history.get('accuracy', [])), label='Training Error')
plt.plot(1 - np.array(history.history.get('val_accuracy', [])), label='Validation Error')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Error Rate Curve (Training and Validation)')
plt.legend()

# Learning Curve: Accuracy vs. Epochs
plt.subplot(1, 2, 2)
plt.plot(history.history.get('accuracy', []), label='Training Accuracy')
plt.plot(history.history.get('val_accuracy', []), label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Learning Curve (Accuracy vs. Epochs)')
plt.legend()

plt.tight_layout()
plt.show()
