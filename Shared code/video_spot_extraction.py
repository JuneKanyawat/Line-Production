import cv2
import os
import re
import glob

def get_spots_boxes(connected_components):
    (totalLabels, label_ids, values, centroid) = connected_components
    slots = []
    coef = 1
    for i in range(1, totalLabels):

        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)

        slots.append([x1, y1, w, h])

    return slots

# Extract file name
def extract_filename(video_path):
    match = re.search(r"cam_video_(.*)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.mp4", video_path)
    if match:
        return match.group(1)
    else:
        return None

# Clear old images in the directory
def clear_old_images(output_dir):
    files = glob.glob(os.path.join(output_dir, '*.png'))
    for f in files:
        os.remove(f)

mask_path = '../dataset/mask_img_DY08_P1_S2.png'  # <---- input mask file path here
video_path = '../dataset/cam_video_DY08_P1_2024-06-28_08-01-43.mp4'  # <---- input video file path here

filename = extract_filename(video_path)

output_dir = f"../dataset/data_train_for_{filename}"
os.makedirs(output_dir, exist_ok=True)

# Clear old images in the output directory
clear_old_images(output_dir)

mask = cv2.imread(mask_path, 0)
cap = cv2.VideoCapture(video_path)

connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = get_spots_boxes(connected_components)

frame_index = 0
frame_counter = 0
process_every = 2  # Process every nth frame

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_counter += 1
    if frame_counter % process_every == 0:
        for spot_index, spot in enumerate(spots):
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]

            # Generate unique filename for spot crop image
            spot_filename = os.path.join(output_dir, f"frame_{frame_index}_{spot_index}.png")
            cv2.imwrite(spot_filename, spot_crop)
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)

        frame_index += 1

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
