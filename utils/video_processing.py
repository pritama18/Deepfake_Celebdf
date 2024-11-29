import cv2
import os

def split_video_to_frames(video_path, output_folder):
    """Splits a video into frames."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames from {video_path}")

def compile_frames_to_video(frame_folder, output_video_path, fps=30):
    """Compiles frames into a video."""
    frames = sorted(os.listdir(frame_folder))
    if not frames:
        print("No frames found in folder.")
        return

    frame_path = os.path.join(frame_folder, frames[0])
    frame = cv2.imread(frame_path)
    height, width, _ = frame.shape
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for frame_file in frames:
        frame = cv2.imread(os.path.join(frame_folder, frame_file))
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved to {output_video_path}")
