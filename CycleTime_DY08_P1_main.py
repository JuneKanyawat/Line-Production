import cv2
import numpy as np
import pickle
import os
from skimage.transform import resize
import time
from datetime import datetime
import tkinter as tk
import threading
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageDraw
import csv
import os
from tkinter import messagebox

app = None

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

filename = os.path.basename(__file__)
filename = filename.split('_')[1] + '_' + filename.split('_')[2]

model1_path = f"dataset/model_{filename}_S1.p"
model2_path = f"dataset/model_{filename}_S2.p"

# Load models
model1 = pickle.load(open(model1_path, "rb"))
model2 = pickle.load(open(model2_path, "rb"))

EMPTY = 0
NOT_EMPTY = 1
OTHER = 2

def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))

def empty_or_not(spot_bgr, model):
    flat_data = []
    img_resized = resize(spot_bgr, (30, 10, 3))
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)
    y_output = model.predict(flat_data)
    return y_output[0]

mask1 = f"dataset/mask_img_{filename}_S1.png"
mask2 = f"dataset/mask_img_{filename}_S2.png"

model_file1 = os.path.basename(mask1).replace('mask_img_', '').replace('.png', '')
model_file2 = os.path.basename(mask2).replace('mask_img_', '').replace('.png', '')

mask1_img = cv2.imread(mask1, 0)
mask2_img = cv2.imread(mask2, 0)

connected_components_mask1 = cv2.connectedComponentsWithStats(mask1_img, 4, cv2.CV_32S)
spots1 = get_spots_boxes(connected_components_mask1)
connected_components_mask2 = cv2.connectedComponentsWithStats(mask2_img, 4, cv2.CV_32S)
spots2 = get_spots_boxes(connected_components_mask2)

# Extract the main box number from the last character of the model file names
main_box1 = int(model_file1[-1])
main_box2 = int(model_file2[-1])

Config_data = []

def create_config_data(spots, main_box, model_file):
    config_data = []
    for i, spot in enumerate(spots):
        x, y, w, h = spot
        sub_box = i + 1  # Sub box index starts from 1
        config_data.append([main_box, sub_box, model_file, x, y, w, h])
    return config_data

# Create Config_data entries for spots1 and spots2
Config_data.extend(create_config_data(spots1, main_box1, model_file1))
Config_data.extend(create_config_data(spots2, main_box2, model_file2))
# print(Config_data)

mask_height, mask_width = mask1_img.shape
image_size = (mask_width,mask_height)

spots_status1 = [None for _ in spots1]
spots_status2 = [None for _ in spots2]
diffs1 = [None for _ in spots1]
diffs2 = [None for _ in spots2]

start_time = time.time()
t_E = t_S = t_prev_E = None
e_detected = s_detected = False

previous_frame = None
previous_spots_status1 = [None for _ in spots1]
previous_spots_status2 = [None for _ in spots2]

cycle_time = "N/A"
assembly_time = "N/A"

original_spots1 = spots1.copy()
original_spots2 = spots2.copy()

def create_box_mask(boxes, image_size, background_color, output_path):
    # Create a new image with the specified background color
    image = Image.new("RGB", image_size, background_color)
    draw = ImageDraw.Draw(image)

    # Draw each box on the image
    for box in boxes:
        x, y, w, h = box
        draw.rectangle([x, y, x + w, y + h], fill=(255, 255, 255))  # White filled rectangle

    # Save the image to the specified output path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)
    print(f"Image saved to {output_path}")


def boxes_overlap(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    if (x1 < x2 + w2 and x1 + w1 > x2 and
            y1 < y2 + h2 and y1 + h1 > y2):
        return True
    return False

def any_boxes_overlap(boxes):
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes_overlap(boxes[i], boxes[j]):
                return True
    return False

background_color = (0, 0, 0)
output_path1 = "dataset/mask_image1.png"
output_path2 = "dataset/mask_image2.png"
# Tkinter GUI setup
class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

        self.show_idx = False

        self.is_adjust = False
        self.is_recording = False
        self.video_writer = None

        self.adjust_window = None
        self.config_window = None

    def create_widgets(self):
        
        self.record_button = tk.Button(self)
        self.record_button["text"] = "Record Video"
        self.record_button["command"] = self.record_video
        self.record_button.pack(side="top")

        self.adjust_button = tk.Button(self)
        self.adjust_button["text"] = "Adjust Position"
        self.adjust_button["command"] = self.adjust_position
        self.adjust_button.pack(side="top")

        self.config_button = tk.Button(self)
        self.config_button["text"] = "Configuration"
        self.config_button["command"] = self.configuration
        self.config_button.pack(side="top")

        self.show_index_button = tk.Button(self)
        self.show_index_button["text"] = "Show Index"
        self.show_index_button["command"] = self.show_index
        self.show_index_button.pack(side="top")

    def show_index(self):
        if not self.show_idx:
            self.show_idx = True
        else:
            self.show_idx = False

    def record_video(self):
        global app
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        global app
        self.is_recording = True
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        video_filename = f"dataset/cam_video_{current_datetime}.mp4"
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        frame_rate = 20.0
        self.video_writer = cv2.VideoWriter(video_filename, codec, frame_rate, (mask_width, mask_height))
        print(f"Recording started: {video_filename}")

    def stop_recording(self):
        global app
        self.is_recording = False
        if self.video_writer is not None:
            self.video_writer.release()
            print("Recording stopped.")

    def adjust_position(self):
        if not self.is_adjust:
            self.is_adjust = True
            self.create_adjust_window()

    def create_adjust_window(self):
        self.adjust_window = tk.Toplevel(self.master)
        self.adjust_window.title("Adjust Position")
        self.adjust_window.geometry("100x50")

        save_button = Button(self.adjust_window,text="Save",command=self.save_adjustments)
        save_button.pack(side="top", padx=10, pady=10)

        reset_button = Button(self.adjust_window,text="Reset", command=self.reset_adjustments)
        reset_button.pack(side="bottom", padx=10, pady=10)

    def save_adjustments(self):
        global Config_data

        if any_boxes_overlap(spots1) or any_boxes_overlap(spots2):
            messagebox.showinfo("Message", "Boxes overlap")
        else:
            Config_data.clear()  # Clear existing data
            Config_data.extend(create_config_data(spots1, main_box1, model_file1))
            Config_data.extend(create_config_data(spots2, main_box2, model_file2))

            create_box_mask(spots1, image_size, background_color, output_path1)
            create_box_mask(spots2, image_size, background_color, output_path2)

            if hasattr(self, 'adjust_window') and self.adjust_window:
                self.adjust_window.destroy()
            self.is_adjust = False

            csv_filename = f"dataset/config_data_{filename}.csv"

            # Check if the file already exists and delete it if it does
            if os.path.exists(csv_filename):
                os.remove(csv_filename)

            # Write Config_data to CSV
            with open(csv_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Main Box', 'Sub Box', 'Model File', 'X', 'Y', 'W', 'H'])  # Write header
                writer.writerows(Config_data)

            print(f"Config_data saved to {csv_filename}")

            # After saving adjustments, update the Treeview in configuration window
            if hasattr(self, 'config_window') and self.config_window:
                self.update_treeview()

    def reset_adjustments(self):
        global spots1, spots2
        self.is_adjust = True
        spots1 = original_spots1.copy()
        spots2 = original_spots2.copy()

    def update_treeview(self):
        if hasattr(self, 'my_table') and self.my_table:
            # Clear existing items in the Treeview
            for item in self.my_table.get_children():
                self.my_table.delete(item)

            # Insert new data from Config_data
            for i, entry in enumerate(Config_data):
                main_box, sub_box, model_file, x, y, w, h = entry
                self.my_table.insert(parent='', index='end', iid=f'entry_{i}', text='',
                                    values=(main_box, sub_box, model_file, x, y, w, h))

    def configuration(self):
        self.config_window = tk.Toplevel(self.master)
        self.config_window.title("Configuration")
        self.config_window.geometry('500x250')
        self.config_window['bg'] = '#AC99F2'

        table_frame = tk.Frame(self.config_window)
        table_frame.pack()

        self.my_table = ttk.Treeview(table_frame)
        self.my_table['columns'] = ('main_box', 'sub_box', 'model_file', 'x', 'y', 'w', 'h')

        self.my_table['columns'] = ('main_box', 'sub_box', 'model_file', 'x', 'y', 'w', 'h')
        self.my_table.column("#0", width=0, stretch=NO)
        self.my_table.column("main_box", anchor=CENTER, width=40)
        self.my_table.column("sub_box", anchor=CENTER, width=40)
        self.my_table.column("model_file", anchor=CENTER, width=80)
        self.my_table.column("x", anchor=CENTER, width=70)
        self.my_table.column("y", anchor=CENTER, width=70)
        self.my_table.column("w", anchor=CENTER, width=70)
        self.my_table.column("h", anchor=CENTER, width=70)

        self.my_table.heading("#0", text="", anchor=CENTER)
        self.my_table.heading("main_box", text="Main Box", anchor=CENTER)
        self.my_table.heading("sub_box", text="Sub Box", anchor=CENTER)
        self.my_table.heading("model_file", text="Model File", anchor=CENTER)
        self.my_table.heading("x", text="X", anchor=CENTER)
        self.my_table.heading("y", text="Y", anchor=CENTER)
        self.my_table.heading("w", text="W", anchor=CENTER)
        self.my_table.heading("h", text="H", anchor=CENTER)

        self.update_treeview()

        scrollbar = tk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.my_table.yview)
        self.my_table.configure(yscrollcommand=scrollbar.set)

        self.my_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

def update_gui():
    global app
    root = tk.Tk()
    root.title("Control panel")
    app = Application(master=root)
    app.mainloop()

gui_thread = threading.Thread(target=update_gui)
gui_thread.start()

cap = cv2.VideoCapture(0)
frame_nmr = 0
ret = True
step = 10
cycle_counter = 1
arr = []
cycle_time_start = time.time()

dragging = False
selected_object_index = None
step_size = 5

def mouse_events(event, x, y, flags, param):
    global dragging, selected_object_index, spots1, spots2


    if app is not None and app.is_adjust:
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, spot in enumerate(spots1):
                x1, y1, w, h = spot
                if x1 <= x <= x1 + w and y1 <= y <= y1 + h:
                    if selected_object_index == i:
                        selected_object_index = None
                    else:
                        selected_object_index = i
                    dragging = True
                    break
            else:
                for i, spot in enumerate(spots2):
                    x2, y2, w, h = spot
                    if x2 <= x <= x2 + w and y2 <= y <= y2 + h:
                        selected_object_index = i + len(spots1)  # Adjust index for spots2
                        dragging = True
                        break
                else:
                    selected_object_index = None

        elif event == cv2.EVENT_MOUSEMOVE:
            if dragging and selected_object_index is not None:
                if selected_object_index < len(spots1):
                    x1, y1, w, h = spots1[selected_object_index]
                    spots1[selected_object_index] = [x - w // 2, y - h // 2, w, h]
                else:
                    real_index = selected_object_index - len(spots1)
                    x2, y2, w, h = spots2[real_index]
                    spots2[real_index] = [x - w // 2, y - h // 2, w, h]

        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False

cv2.namedWindow('Monitor', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Monitor', mouse_events)

while ret:
    ret, frame = cap.read()

    if not ret:
        break

    copy_frame = frame.copy()
    frame = cv2.resize(frame, (mask_width, mask_height))
    copy_frame = cv2.resize(copy_frame, (mask_width, mask_height))

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

        if not s_detected and not_empty_count1 < sum(1 for s in previous_spots_status1 if s == NOT_EMPTY):
            t_S = time.time() - start_time
            s_detected = True
            e_detected = False
            print("S stage detected at: {:.1f} seconds".format(t_S))

        if not e_detected and not_empty_count2 > sum(1 for s in previous_spots_status2 if s == NOT_EMPTY):
            t_E = time.time() - start_time
            e_detected = True
            print("E stage detected at: {:.1f} seconds".format(t_E))

            if s_detected:
                assembly_time = "{:.1f} s".format(t_E - t_S)
                print("Assembly time {}: {}".format(cycle_counter, assembly_time))
                cycle_counter += 1
                s_detected = False

            if t_prev_E is not None:
                cycle_time = "{:.1f} s".format(t_E - t_prev_E)
                print("Cycle time: {}".format(cycle_time))

            t_prev_E = t_E

    if frame_nmr % step == 0:
        previous_frame = frame.copy()
        previous_spots_status1 = spots_status1.copy()
        previous_spots_status2 = spots_status2.copy()

    for spot_indx, spot in enumerate(spots1):
        spot_status = spots_status1[spot_indx]
        x1, y1, w, h = spots1[spot_indx]
        if spot_status == EMPTY:
            color = (0, 0, 255)
        elif spot_status == NOT_EMPTY:
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
        frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)
        if app.show_idx:
            cv2.putText(frame, str(spot_indx+1), (x1 + w // 2, y1 + h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    for spot_indx, spot in enumerate(spots2):
        spot_status = spots_status2[spot_indx]
        x1, y1, w, h = spots2[spot_indx]
        if spot_status == EMPTY:
            color = (0, 0, 255)
        elif spot_status == NOT_EMPTY:
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
        frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)
        if app.show_idx:
            cv2.putText(frame, str(spot_indx+1), (x1 + w // 2, y1 + h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    current_time = datetime.now().strftime("%H:%M:%S")
    if app.is_adjust:
        cv2.putText(frame, "Adjust position enable", (frame.shape[1] - 230, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


    cv2.putText(frame, current_time, (frame.shape[1] - 200, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                cv2.LINE_AA)

    cv2.imshow('Monitor', frame)

    if app is not None and app.is_recording:
        if app.video_writer is not None:
            app.video_writer.write(copy_frame)

    # Keyboard events for adjusting positions
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and app.is_adjust and selected_object_index is not None:
        if selected_object_index < len(spots1):
            x1, y1, w, h = spots1[selected_object_index]
            spots1[selected_object_index] = [x1, y1 + step_size, w, h]
        else:
            real_index = selected_object_index - len(spots1)
            x2, y2, w, h = spots2[real_index]
            spots2[real_index] = [x2, y2 + step_size, w, h]
    elif key == ord('w') and app.is_adjust and selected_object_index is not None:
        if selected_object_index < len(spots1):
            x1, y1, w, h = spots1[selected_object_index]
            spots1[selected_object_index] = [x1, y1 - step_size, w, h]
        else:
            real_index = selected_object_index - len(spots1)
            x2, y2, w, h = spots2[real_index]
            spots2[real_index] = [x2, y2 - step_size, w, h]
    elif key == ord('a') and app.is_adjust and selected_object_index is not None:
        if selected_object_index < len(spots1):
            x1, y1, w, h = spots1[selected_object_index]
            spots1[selected_object_index] = [x1 - step_size, y1, w, h]
        else:
            real_index = selected_object_index - len(spots1)
            x2, y2, w, h = spots2[real_index]
            spots2[real_index] = [x2 - step_size, y2, w, h]
    elif key == ord('d') and app.is_adjust and selected_object_index is not None:
        if selected_object_index < len(spots1):
            x1, y1, w, h = spots1[selected_object_index]
            spots1[selected_object_index] = [x1 + step_size, y1, w, h]
        else:
            real_index = selected_object_index - len(spots1)
            x2, y2, w, h = spots2[real_index]
            spots2[real_index] = [x2 + step_size, y2, w, h]

    if cv2.waitKey(25) & 0xFF == ord('q'):
        print(arr)
        break

    frame_nmr += 1

cap.release()
print(arr)
cv2.destroyAllWindows()
