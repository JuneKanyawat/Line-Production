import cv2
import numpy as np
import pickle
from skimage.transform import resize
from util import get_spots_boxes

# Load models
model1 = pickle.load(open("new_three_model.p", "rb"))
model2 = pickle.load(open("new_black_model.p", "rb"))

# Define constants for categories
EMPTY = 0
NOT_EMPTY = 1
OTHER = 2  # Add more categories as needed

def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))

def empty_or_not(spot_bgr, model):
    flat_data = []
    img_resized = resize(spot_bgr, (30, 10, 3))
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)
    y_output = model.predict(flat_data)
    print(y_output)
    return y_output[0]

# Update paths for new data
mask1 = 'datasets/mask/frame_1 (1).png'
mask2 = 'datasets/mask/frame_1.png'
video_path = 'datasets/video/videor_croxsde1.mp4'

# Load mask images
mask1_img = cv2.imread(mask1, 0)
mask2_img = cv2.imread(mask2, 0)

# Get spots from masks
connected_components_mask1 = cv2.connectedComponentsWithStats(mask1_img, 4, cv2.CV_32S)
spots1 = get_spots_boxes(connected_components_mask1)
connected_components_mask2 = cv2.connectedComponentsWithStats(mask2_img, 4, cv2.CV_32S)
spots2 = get_spots_boxes(connected_components_mask2)

# Initialize status and diffs arrays
spots_status1 = [None for _ in spots1]
spots_status2 = [None for _ in spots2]
diffs1 = [None for _ in spots1]
diffs2 = [None for _ in spots2]

previous_frame = None

# Open video capture
cap = cv2.VideoCapture(video_path)
frame_nmr = 0
ret = True
step = 10

while ret:
    ret, frame = cap.read()

    if frame_nmr % step == 0 and previous_frame is not None:
        for spot_indx, spot in enumerate(spots1):
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            diffs1[spot_indx] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])

        for spot_indx, spot in enumerate(spots2):
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            diffs2[spot_indx] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])

        print("Mask1 diffs:", [diffs1[j] for j in np.argsort(diffs1)][::-1])
        print("Mask2 diffs:", [diffs2[j] for j in np.argsort(diffs2)][::-1])

    if frame_nmr % step == 0:
        if previous_frame is None:
            arr_1 = range(len(spots1))
            arr_2 = range(len(spots2))
        else:
            arr_1 = [j for j in np.argsort(diffs1) if diffs1[j] / np.amax(diffs1) > 0.4]
            arr_2 = [j for j in np.argsort(diffs2) if diffs2[j] / np.amax(diffs2) > 0.4]

        for spot_indx in arr_1:
            spot = spots1[spot_indx]
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            spot_status = empty_or_not(spot_crop, model1)
            spots_status1[spot_indx] = spot_status

        for spot_indx in arr_2:
            spot = spots2[spot_indx]
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            spot_status = empty_or_not(spot_crop, model2)
            spots_status2[spot_indx] = spot_status

    if frame_nmr % step == 0:
        previous_frame = frame.copy()

    for spot_indx, spot in enumerate(spots1):
        spot_status = spots_status1[spot_indx]
        x1, y1, w, h = spots1[spot_indx]
        if spot_status == EMPTY:
            color = (0, 0, 255)  # Red
        elif spot_status == NOT_EMPTY:
            color = (0, 255, 0)  # Green
        else:
            color = (255, 0, 0)  # Blue or another color for additional categories
        frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)

    for spot_indx, spot in enumerate(spots2):
        spot_status = spots_status2[spot_indx]
        x1, y1, w, h = spots2[spot_indx]
        if spot_status == EMPTY:
            color = (0, 0, 255)  # Red
        elif spot_status == NOT_EMPTY:
            color = (0, 255, 0)  # Green
        else:
            color = (255, 0, 0)  # Blue or another color for additional categories
        frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)

    # Display the frame
    cv2.rectangle(frame, (80, 20), (300, 80), (255, 255, 255), -1)
    cv2.putText(frame, '(Mask1): {} / {}'.format(str(sum([1 for s in spots_status1 if s == NOT_EMPTY])), str(len(spots_status1))), (100, 40),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1)
    cv2.putText(frame, '(Mask2): {} / {}'.format(str(sum([1 for s in spots_status2 if s == NOT_EMPTY])), str(len(spots_status2))), (100, 80),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1)

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    frame_nmr += 1

cap.release()
cv2.destroyAllWindows()