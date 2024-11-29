import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Flatten, TimeDistributed
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns

from utils.video_processing import split_video_to_frames
from utils.face_detection import extract_faces_from_frames

# Constants
DATASET_DIR = "dataset"
FRAME_DIR = "frames"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
TIME_STEPS = 10
EPOCHS = 20

def preprocess_frames(frame_folder):
    """Loads and preprocesses frames."""
    frames = []
    for img_file in sorted(os.listdir(frame_folder))[:TIME_STEPS]:
        img_path = os.path.join(frame_folder, img_file)
        if not img_file.endswith(".jpg"):
            continue
        img = load_img(img_path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        frames.append(img_array)
    if len(frames) < TIME_STEPS:
        print(f"Insufficient frames in {frame_folder}. Padding...")
        while len(frames) < TIME_STEPS:
            frames.append(np.zeros((*IMG_SIZE, 3)))
    return np.array(frames)

def load_dataset():
    """Loads the dataset and prepares it for training."""
    X, y = [], []
    for label, category in enumerate(["real", "fake"]):
        category_folder = os.path.join(FRAME_DIR, category)
        for video_folder in os.listdir(category_folder):
            frame_folder = os.path.join(category_folder, video_folder)
            if not os.path.isdir(frame_folder):
                continue
            video_frames = preprocess_frames(frame_folder)
            X.append(video_frames)
            y.append(label)
    X = np.array(X)
    y = np.array(y)

    # Oversample minority class
    X_flat = X.reshape(X.shape[0], -1)
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_flat, y)
    X_resampled = X_resampled.reshape(-1, TIME_STEPS, *IMG_SIZE, 3)
    return X_resampled, y_resampled

def build_model():
    """Builds the ResNet-LSTM model."""
    base_model = ResNet50(include_top=False, input_shape=(*IMG_SIZE, 3))
    resnet_model = Model(inputs=base_model.input, outputs=Flatten()(base_model.output))

    model = Sequential([
        TimeDistributed(resnet_model, input_shape=(TIME_STEPS, *IMG_SIZE, 3)),
        LSTM(128, return_sequences=False),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def plot_training_history(history):
    """Plots the training and validation accuracy/loss."""
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def train_model():
    """Trains the ResNet-LSTM model and evaluates its performance."""
    # Load dataset
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and compile model
    model = build_model()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint("models/resnet_lstm_model.h5", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=2)
    ]

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate model
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    accuracy = (y_pred.flatten() == y_test).mean() * 100
    print(f"Accuracy: {accuracy:.2f}%")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"], zero_division=1))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # Plot training history
    plot_training_history(history)

if __name__ == "__main__":
    # Step 1: Process dataset (split videos into frames and extract faces)
    for category in ["real", "fake"]:
        for video_file in os.listdir(os.path.join(DATASET_DIR, category)):
            video_path = os.path.join(DATASET_DIR, category, video_file)
            output_folder = os.path.join(FRAME_DIR, category, os.path.splitext(video_file)[0])
            split_video_to_frames(video_path, output_folder)
            extract_faces_from_frames(output_folder, output_folder)

    # Step 2: Train the model
    train_model()
