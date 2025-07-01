import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import make_classification # Using make_classification now
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Data Preparation ---
print("--- Data Preparation ---")

# Define parameters to mimic your chord problem
NUM_SAMPLES = 5000          # Number of synthetic samples (adjust based on your real data size)
NUM_FEATURES = 12           # Corresponds to 12 chroma features
NUM_CLASSES = 36            # Corresponds to your 36 chord classes

# Generate a synthetic dataset for multi-class classification
# n_features: number of input features (like chroma bins)
# n_informative: number of features that actually contribute to the target
# n_redundant: number of features that are linear combinations of informative features
# n_repeated: number of features that are duplicates of informative features
# n_classes: number of output classes (your chord types)
# random_state: for reproducibility
X, y = make_classification(
    n_samples=NUM_SAMPLES,
    n_features=NUM_FEATURES,
    n_informative=NUM_FEATURES, # All features are informative
    n_redundant=0,
    n_repeated=0,
    n_classes=NUM_CLASSES,
    n_clusters_per_class=1, # One cluster of data points per class
    random_state=42
)

print(f"Generated dataset shape: X={X.shape}, y={y.shape}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y # Stratify to maintain class proportions
)
print(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Testing data shape: X_test={X_test.shape}, y_test={y_test.shape}")

# Scale features: Neural networks often perform better when input features are scaled
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled.")

# No direct visualization like make_moons for 12D data, but we can check class distribution
print("\nClass distribution in training set:")
unique, counts = np.unique(y_train, return_counts=True)
print(dict(zip(unique, counts)))


# --- 2. Model Definition (using Keras Sequential API) ---
print("\n--- Model Definition ---")
model = keras.Sequential([
    # Hidden Layer 1: Dense layer with 128 neurons (a common starting point)
    # input_shape=(NUM_FEATURES,) is crucial for the first layer, matching our 12 features
    keras.layers.Dense(128, activation='relu', input_shape=(NUM_FEATURES,)),
    
    # Hidden Layer 2: Another Dense layer with 64 neurons
    keras.layers.Dense(64, activation='relu'),
    
    # Output Layer: Dense layer with NUM_CLASSES neurons for multi-class classification
    # Softmax activation outputs a probability distribution over the classes
    keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

# Display a summary of the model's architecture
model.summary()

# --- 3. Model Compilation ---
print("\n--- Model Compilation ---")
# Compile the model
# loss: 'sparse_categorical_crossentropy' is used when labels are integers (0, 1, 2, ..., N-1)
#       If labels were one-hot encoded (e.g., [0,0,1,0] for class 2), you'd use 'categorical_crossentropy'
model.compile(
    optimizer='adam',                  # Adam optimizer is a good default
    loss='sparse_categorical_crossentropy', # Appropriate for integer labels
    metrics=['accuracy']               # Monitor accuracy
)
print("Model compiled.")

# --- 4. Model Training ---
print("\n--- Model Training ---")
# Train the model
history = model.fit(
    X_train_scaled, y_train,
    epochs=100,             # Train for 100 epochs (can be tuned)
    batch_size=32,          # Process 32 samples per batch
    validation_split=0.1,   # Use 10% of training data for validation during training
    verbose=1               # Show progress bar
)
print("Model training complete.")

# Plot training history (loss and accuracy over epochs)
plt.figure(figsize=(12, 5))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.grid(True)

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()


# --- 5. Model Evaluation ---
print("\n--- Model Evaluation ---")
# Evaluate the model on the unseen test data
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# --- 6. Model Prediction ---
print("\n--- Model Prediction ---")
# Make predictions on new data (e.g., the first 5 samples from the test set)
sample_data = X_test_scaled[:5]
predictions_probabilities = model.predict(sample_data)

# For multi-class classification with softmax, np.argmax gives the predicted class index
predictions_classes = np.argmax(predictions_probabilities, axis=1)

print("\nSample Predictions:")
for i in range(len(sample_data)):
    print(f"Input Features (first 5): {sample_data[i][:5]}, True Label: {y_test[i]}, "
          f"Predicted Probabilities: {predictions_probabilities[i]}, " # Show full probability distribution
          f"Predicted Class Index: {predictions_classes[i]}")

# You can also save the trained model for later use
# model.save('my_chord_nn_model.h5')
# print("\nModel saved to 'my_chord_nn_model.h5'")

# To load the model later:
# loaded_model = keras.models.load_model('my_chord_nn_model.h5')
# print("Model loaded from 'my_chord_nn_model.h5'")
