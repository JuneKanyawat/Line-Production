import cv2
import pickle
from skimage.transform import resize
import numpy as np


EMPTY = False
NOT_EMPTY = True

model1 = pickle.load(open("model/model.p", "rb"))
model2 = pickle.load(open("model/bl_model.p", "rb"))

def empty_or_not(spot_bgr):
    flat_data = []

    img_resized = resize(spot_bgr, (30, 10, 3))
    # img_resized = resize(spot_bgr, (15, 15, 3))
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)

    # Print the shape and contents of flat_data
    # print("Shape of flat_data:", flat_data.shape)
    # print("Contents of flat_data:", flat_data)

    y_output = model1.predict(flat_data)
    # print(y_output)

    if y_output == 0:
        return EMPTY
    else:
        return NOT_EMPTY


def get_spots_boxes(connected_components):
    (totalLabels, label_ids, values, centroid) = connected_components

    slots = []
    coef = 1
    for i in range(1, totalLabels):

        # Now extract the coordinate points
        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)

        slots.append([x1, y1, w, h])

    return slots