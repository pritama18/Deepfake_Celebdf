import os
import face_recognition
from PIL import Image

def extract_faces_from_frames(input_folder, output_folder):
    """Detects and extracts faces from frames."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        frame_path = os.path.join(input_folder, file_name)
        frame = face_recognition.load_image_file(frame_path)
        face_locations = face_recognition.face_locations(frame)

        for i, face_location in enumerate(face_locations):
            top, right, bottom, left = face_location
            face_image = frame[top:bottom, left:right]
            face_pil = Image.fromarray(face_image)
            face_pil.save(os.path.join(output_folder, f"{file_name}_face_{i}.jpg"))

    print(f"Faces extracted to {output_folder}")
