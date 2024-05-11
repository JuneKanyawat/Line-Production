import cv2
import os

# Path to the input video file
video_path = "../datasets/video/video_code.mp4"

# Create a directory to save captured frames if it doesn't exist
output_dir = "../datasets/cap_frame"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Initialize variables
frame_count = 0
capture_count = 0

# Loop through the video frames
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Check if frame reading was successful
    if not ret:
        break

    frame_count += 1

    # Capture every 5th frame
    if frame_count % 5 == 0:
        # Save the captured frame
        cv2.imwrite(os.path.join(output_dir, f"frame_{capture_count}.jpg"), frame)
        capture_count += 1

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

print(f"Total frames in the video: {frame_count}")
print(f"Frames captured: {capture_count}")
print(f"Captured frames saved in {output_dir}")
