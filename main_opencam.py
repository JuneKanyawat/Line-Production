import cv2
import numpy as np
import pickle
from skimage.transform import resize
import time
from datetime import datetime

# Function to get spots boxes from the mask
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

# Load models
model1 = pickle.load(open("dataset/model_DY08_P1SI.p", "rb"))
model2 = pickle.load(open("dataset/model_DY08_P1SO.p", "rb"))

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
    return y_output[0]

# Update paths for new data
mask1 = 'dataset/mask_img_DY08_P1SI.png'
mask2 = 'dataset/mask_img_DY08_P1SO.png'

# Load mask images
mask1_img = cv2.imread(mask1, 0)
mask2_img = cv2.imread(mask2, 0)

# Get spots from masks
connected_components_mask1 = cv2.connectedComponentsWithStats(mask1_img, 4, cv2.CV_32S)
spots1 = get_spots_boxes(connected_components_mask1)
connected_components_mask2 = cv2.connectedComponentsWithStats(mask2_img, 4, cv2.CV_32S)
spots2 = get_spots_boxes(connected_components_mask2)

# Get dimensions of the mask images
mask_height, mask_width = mask1_img.shape

# Initialize status and diffs arrays
spots_status1 = [None for _ in spots1]
spots_status2 = [None for _ in spots2]
diffs1 = [None for _ in spots1]
diffs2 = [None for _ in spots2]

# Initialize timestamps and flags for stages
start_time = time.time()
t_E = t_S = t_prev_E = None
e_detected = s_detected = False

previous_frame = None
previous_spots_status1 = [None for _ in spots1]
previous_spots_status2 = [None for _ in spots2]

# Variables to store cycle time and assembly time
cycle_time = "N/A"
assembly_time = "N/A"

# Open video capture from the webcam
cap = cv2.VideoCapture(5)
frame_nmr = 0
ret = True
step = 10
cycle_counter = 1  # Initialize cycle counter
arr = []
cycle_time_start = time.time()

while ret:
    ret, frame = cap.read()

    if not ret:
        break

    # Resize frame to match mask dimensions
    frame = cv2.resize(frame, (mask_width, mask_height))

    if frame_nmr % step == 0 and previous_frame is not None:
        for spot_indx, spot in enumerate(spots1):
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            diffs1[spot_indx] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])

        for spot_indx, spot in enumerate(spots2):
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            diffs2[spot_indx] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])

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
        not_empty_count2 = sum(1 for s in spots_status2 if s == NOT_EMPTY)
        not_empty_count1 = sum(1 for s in spots_status1 if s == NOT_EMPTY)

        # Detect S stage (decrement in mask 1)
        if not s_detected and not_empty_count1 < sum(1 for s in previous_spots_status1 if s == NOT_EMPTY):
            t_S = time.time() - start_time
            s_detected = True
            e_detected = False  # Reset E detection flag for a new cycle
            print("S stage detected at: {:.1f} seconds".format(t_S))

        # Detect E stage (increment in mask 2 down box)
        if not e_detected and not_empty_count2 > sum(1 for s in previous_spots_status2 if s == NOT_EMPTY):
            t_E = time.time() - start_time
            e_detected = True
            print("E stage detected at: {:.1f} seconds".format(t_E))

            # Calculate and store time between S and E stages if S was detected
            if s_detected:
                assembly_time = "{:.1f} s".format(t_E - t_S)
                print("Assembly time {}: {}".format(cycle_counter, assembly_time))
                cycle_counter += 1
                s_detected = False  # Reset S detection flag for a new cycle

            # Calculate and store time between two E stages
            if t_prev_E is not None:
                cycle_time = "{:.1f} s".format(t_E - t_prev_E)
                print("Cycle time: {}".format(cycle_time))

            t_prev_E = t_E  # Update the previous E stage timestamp

        # Store cycle time every 1 second
        current_time = time.time()
        if current_time - cycle_time_start >= 1:
            if cycle_time == "N/A":
                arr.append(0)
            else:
                arr.append(cycle_time)
            cycle_time_start = current_time

    if frame_nmr % step == 0:
        previous_frame = frame.copy()
        previous_spots_status1 = spots_status1.copy()
        previous_spots_status2 = spots_status2.copy()

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
    current_time = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, current_time, (frame.shape[1] - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                cv2.LINE_AA)
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        print(arr)
        break
    frame_nmr += 1

cap.release()
print(arr)
cv2.destroyAllWindows()
