import os
import cv2
import tempfile
import numpy as np
import streamlit as st
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from utils.video_processing import split_video_to_frames
from utils.face_detection import extract_faces_from_frames

# Constants
IMG_SIZE = (224, 224)
TIME_STEPS = 10
FRAME_DIR = "temp_frames"

# Load model
MODEL_PATH = "models/resnet_lstm_model.h5"
model = load_model(MODEL_PATH)

# Add custom CSS for background image
def add_background(image_path):
    """Adds a background image to the Streamlit app."""
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url({image_path});
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def preprocess_frames(frame_folder):
    """Loads and preprocesses frames."""
    frames = []
    for img_file in sorted(os.listdir(frame_folder))[:TIME_STEPS]:
        img_path = os.path.join(frame_folder, img_file)
        img = load_img(img_path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        frames.append(img_array)
    return np.array(frames)

def predict_video(video_path):
    """Predicts if a video is real or fake."""
    # Step 1: Split video into frames
    split_video_to_frames(video_path, FRAME_DIR)
    
    # Step 2: Extract faces from frames
    extract_faces_from_frames(FRAME_DIR, FRAME_DIR)
    
    # Step 3: Preprocess frames
    frames = preprocess_frames(FRAME_DIR)
    if frames.shape[0] < TIME_STEPS:
        st.warning("Not enough valid frames detected.")
        return None

    # Step 4: Predict
    frames = np.expand_dims(frames, axis=0)  # Add batch dimension
    prediction = model.predict(frames)[0][0]
    return "Real" if prediction < 0.5 else "Fake", prediction

# Streamlit App
add_background("image.jpg")  # Replace with your image URL or local file path

st.title("Deepfake Detection")

st.sidebar.title("Upload Video")
uploaded_video = st.sidebar.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_video:
    # Save the video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_video.read())
        video_path = temp_file.name

    st.video(video_path)
    st.write("Processing video...")

    # Predict the video
    prediction_result = predict_video(video_path)

    if prediction_result:
        label, confidence = prediction_result
        st.success(f"The video is predicted to be: **{label}**")
        st.info(f"Prediction Confidence: {confidence:.2f}")
    else:
        st.error("Could not process the video.")

    # Clean up
    for file in os.listdir(FRAME_DIR):
        os.remove(os.path.join(FRAME_DIR, file))
    os.rmdir(FRAME_DIR)
