import cv2
from util import get_spots_boxes, empty_or_not


mask_path = '../datasets/mask/frame_small.jpeg'
video_path = '../datasets/video/video_code1.mp4'

mask = cv2.imread(mask_path, 0)
cap = cv2.VideoCapture(video_path)

connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = get_spots_boxes(connected_components)

frame_index = 0
frame_counter = 0
process_every = 2  # Process every 8th frame

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Increment frame counter
    frame_counter += 1

    # Process frame only if it's the 8th frame
    if frame_counter % process_every == 0:
        for spot_index, spot in enumerate(spots):
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            spot_status = empty_or_not(spot_crop)

            # Generate unique filename for spot crop image
            spot_filename = f"spot_crop/frame_{frame_index}_spot_{spot_index}_{spot_status}.png"
            cv2.imwrite(spot_filename, spot_crop)

            # Draw bounding box based on spot status
            if spot_status:
                frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)
            else:
                frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)

        frame_index += 1

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
