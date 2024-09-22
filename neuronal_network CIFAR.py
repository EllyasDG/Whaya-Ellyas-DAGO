import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.model_selection import KFold

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# Normalize the data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Create the CNN model
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Start timing the cross-validation process
start_time = time.time()

# Cross-validation with KFold (5 splits)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_acc = []
cv_loss = []
train_histories = []

for train_idx, val_idx in kfold.split(x_train, y_train):
    model = create_model()
    
    # Split the training data
    x_train_fold, x_val_fold = x_train[train_idx], x_train[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
    
    # Train the model and save history
    history = model.fit(x_train_fold, y_train_fold, epochs=15, validation_data=(x_val_fold, y_val_fold),
                        verbose=2, callbacks=[early_stopping])
    
    # Save training history for plotting later
    train_histories.append(history)
    
    # Evaluate on validation fold
    val_loss, val_acc = model.evaluate(x_val_fold, y_val_fold, verbose=0)
    cv_acc.append(val_acc)
    cv_loss.append(val_loss)

# Stop timing after cross-validation is complete
end_time = time.time()
execution_time = end_time - start_time

# Find the minimum number of epochs across all histories
min_epochs = min([len(h.history['loss']) for h in train_histories])

# Truncate all histories to the shortest number of epochs
truncated_train_losses = [h.history['loss'][:min_epochs] for h in train_histories]
truncated_val_losses = [h.history['val_loss'][:min_epochs] for h in train_histories]
truncated_train_accuracies = [h.history['accuracy'][:min_epochs] for h in train_histories]
truncated_val_accuracies = [h.history['val_accuracy'][:min_epochs] for h in train_histories]

# Cross-validation results
print(f'\nCross-validation accuracy (mean): {np.mean(cv_acc):.4f}')
print(f'Cross-validation loss (mean): {np.mean(cv_loss):.4f}')
print(f'Execution time for cross-validation: {execution_time:.2f} seconds')

# Plot learning curves (averaged across folds)
plt.figure(figsize=(12, 5))

# Collect and average loss/accuracy over epochs across all folds
epochs = range(1, min_epochs + 1)
avg_train_loss = np.mean(truncated_train_losses, axis=0)
avg_val_loss = np.mean(truncated_val_losses, axis=0)
avg_train_acc = np.mean(truncated_train_accuracies, axis=0)
avg_val_acc = np.mean(truncated_val_accuracies, axis=0)

# Loss curve (train and validation)
plt.subplot(1, 3, 1)
plt.plot(epochs, avg_train_loss, label='Training Loss')
plt.plot(epochs, avg_val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Accuracy curve (train and validation)
plt.subplot(1, 3, 2)
plt.plot(epochs, avg_train_acc, label='Training Accuracy')
plt.plot(epochs, avg_val_acc, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Error rate curve (1 - accuracy)
plt.subplot(1, 3, 3)
train_error_rate = 1 - np.array(avg_train_acc)
val_error_rate = 1 - np.array(avg_val_acc)
plt.plot(epochs, train_error_rate, label='Training Error Rate')
plt.plot(epochs, val_error_rate, label='Validation Error Rate')
plt.xlabel('Epochs')
plt.ylabel('Error Rate')
plt.legend()

plt.tight_layout()
plt.show()
