import cv2
import numpy as np
import pickle
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


def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))


def empty_or_not(spot_bgr, model):
    flat_data = []
    img_resized = resize(spot_bgr, (30, 10, 3))
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)
    y_output = model.predict(flat_data)
    return y_output[0]


# Read csv file to get Main box, Sub box data structure
def GetStructureFromCSV(file_path):
    # Initialize a dictionary to store the count of sub boxes for each main box
    main_box_counts = {}
    # Open the CSV file
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        # Skip the title row
        next(csv_reader)
        # Read each row in the CSV file
        for row in csv_reader:
            main_box = int(row[0])
            if main_box in main_box_counts:
                main_box_counts[main_box] += 1
            else:
                main_box_counts[main_box] = 1

    result = [[main_box, count] for main_box, count in main_box_counts.items()]
    return result

def create_config_data(spots, main_box, model_file):
    config_data = []
    for i, spot in enumerate(spots):
        x, y, w, h = spot
        sub_box = i + 1  # Sub box index starts from 1
        config_data.append([main_box, sub_box, model_file, x, y, w, h])
    return config_data

def create_box_mask(boxes, image_size, background_color, output_path):
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
            y1 < y2 + h2  and y1 + h1 > y2):
        return True
    return False

def any_boxes_overlap(boxes):
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes_overlap(boxes[i], boxes[j]):
                return True
    return False

filename = os.path.basename(__file__)
filename = filename.split('_')[1] + '_' + filename.split('_')[2]

MainSub = GetStructureFromCSV(f"dataset/config_data_{filename}.csv")
print(MainSub)

EMPTY = 0
NOT_EMPTY = 1
OTHER = 2

Config_data = []

for S in [sublist[0] for sublist in MainSub]:
    # Construct the model path
    model_path = f"dataset/model_{filename}_S{S}.p"

    # Check if the model file exists before loading
    if os.path.exists(model_path):
        try:
            model = pickle.load(open(model_path, "rb"))
        except Exception as e:
            continue
    else:
        model_path = 'dataset/model_default.p'
        model = pickle.load(open(model_path, "rb"))

    mask = f"dataset/mask_img_{filename}_S{S}.png"
    model_file = model_path.split('/', 1)[-1]
    mask_img = cv2.imread(mask, 0)
    spots = get_spots_boxes(cv2.connectedComponentsWithStats(mask_img, 4, cv2.CV_32S))

    spots_status = [None for _ in spots]
    diffs = [None for _ in spots]
    previous_spots_status = [None for _ in spots]
    original_spots = spots.copy()

    # Create Config_data entries for spots1 and spots2
    Config_data.extend(create_config_data(spots, S, model_file))

    # Dynamically create variable names like model1, model2, etc.
    globals()[f'model{S}'] = model
    globals()[f'mask{S}'] = mask
    globals()[f'model_file{S}'] = model_file
    globals()[f'mask{S}_img'] = mask_img
    globals()[f'spots{S}'] = spots

    globals()[f'spots_status{S}'] = spots_status
    globals()[f'diffs{S}'] = diffs
    globals()[f'previous_spots_status{S}'] = previous_spots_status
    globals()[f'original_spots{S}'] = original_spots

cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

mask_img_size = cv2.imread(f"dataset/mask_img_{filename}_S{MainSub[0][0]}.png",0)
mask_height, mask_width = mask_img_size.shape
image_size = (frame_width, frame_height)

start_time = time.time()
t_E = t_S = t_prev_E = None
e_detected = s_detected = False

previous_frame = None
cycle_time = "N/A"
assembly_time = "N/A"

background_color = (0, 0, 0)
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
        self.master.geometry("350x150+0+0")

        # Create container frame
        self.container = tk.Frame(self.master)
        self.container.pack(fill="both", expand=True)

        self.frames = {}
        for F in (StartPage, RecordPage, AdjustPage):
            page_name = F.__name__
            frame = F(parent=self.container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")

    def show_frame(self, page_name):
        if page_name == "StartPage":
            self.is_adjust = False
            for S in [sublist[0] for sublist in MainSub]:
                globals()[f'spots{S}'] = globals()[f'original_spots{S}'].copy()
        if page_name == "AdjustPage":
            self.is_adjust = True
        frame = self.frames[page_name]
        frame.tkraise()

    def show_index(self):
        self.show_idx = not self.show_idx

    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            video_filename = f"dataset/cam_video_{filename}_{current_datetime}.mp4"
            codec = cv2.VideoWriter_fourcc(*'mp4v')
            frame_rate = 20.0
            self.video_writer = cv2.VideoWriter(video_filename, codec, frame_rate, (mask_width, mask_height))
            print(f"Recording started: {video_filename}")

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            if self.video_writer is not None:
                self.video_writer.release()
                print("Recording stopped.")
            messagebox.showinfo("Recording Stopped", "Recording has stopped.")
        else:
            print("No active recording to stop.")

    def save_adjustments(self):
        global Config_data

        if any(any_boxes_overlap(globals()[f'spots{sublist[0]}']) for sublist in MainSub):
            overlapping_sublist = next(
                sublist for sublist in MainSub if any_boxes_overlap(globals()[f'spots{sublist[0]}']))
            messagebox.showinfo("Message", f"Boxes overlap on Main box {overlapping_sublist[0]}")

        else:
                Config_data.clear()  # Clear existing data
                for S in [sublist[0] for sublist in MainSub]:
                    Config_data.extend(create_config_data(globals()[f'spots{S}'], S, globals()[f'model_file{S}']))

                for S in [sublist[0] for sublist in MainSub]:
                    image_size = (480,640)
                    create_box_mask(globals()[f'spots{S}'], image_size, background_color, globals()[f'mask{S}'])

                for S in [sublist[0] for sublist in MainSub]:
                    globals()[f'original_spots{S}'] = globals()[f'spots{S}'].copy()

                csv_filename = f"dataset/config_data_{filename}.csv"
                # Check if the file already exists and delete it if it does
                if os.path.exists(csv_filename):
                    try:
                        os.remove(csv_filename)
                        print(f"{csv_filename} has been removed successfully.")
                    except PermissionError as e:
                        print(f"Error: {e}")

                # Write Config_data to CSV
                with open(csv_filename, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Main Box', 'Sub Box', 'Model File', 'X', 'Y', 'W', 'H'])  # Write header
                    writer.writerows(Config_data)

                print(f"Config_data saved to {csv_filename}")
                self.is_adjust = True
                self.update_treeview()

    def reset_adjustments(self):
        self.is_adjust = True

        for S in [sublist[0] for sublist in MainSub]:
            globals()[f'spots{S}'] = globals()[f'original_spots{S}'].copy()

    def add_box(self):
        adjust_page = self.frames["AdjustPage"]
        selected_main = adjust_page.selected_main.get()

        if selected_main == "new":
            # Handle creation of new main box
            new_main_index = max([i[0] for i in MainSub]) + 1
            MainSub.append([new_main_index, 0])

            # Initialize necessary globals for the new main box
            globals()[f'model{new_main_index}'] = None  # Example: Initialize model as None
            globals()[f'spots{new_main_index}'] = []
            globals()[f'spots_status{new_main_index}'] = []
            globals()[f'original_spots{new_main_index}'] = []
            globals()[f'diffs{new_main_index}'] = []
            globals()[f'model_file{new_main_index}'] = 'dataset/model_default'
            globals()[f'mask{new_main_index}'] = f"dataset/mask_img_{filename}_S{new_main_index}.png"

            # Create an example image for the mask
            img_size = (540, 960)  # Example image size
            bg_color = (0, 0, 0)  # Example background color (black)
            image = Image.new("RGB", img_size, bg_color)

            # Save the image to the specified mask path
            os.makedirs(os.path.dirname(globals()[f'mask{new_main_index}']), exist_ok=True)
            image.save(globals()[f'mask{new_main_index}'])

            # Update choices list with new main box index
            choices = [str(sublist[0]) for sublist in MainSub] + ["new"]
            adjust_page.selected_main.set(choices[0])  # Reset dropdown to first option

            # Update OptionMenu with new choices
            menu = adjust_page.dropdown["menu"]
            menu.delete(0, "end")  # Clear existing options
            for choice in choices:
                menu.add_command(label=choice, command=tk._setit(adjust_page.selected_main, choice))

            spots_list = globals()[f'spots{new_main_index}']
            selected_main = new_main_index
        else:
            selected_main = int(selected_main)
            spots_list = globals().get(f'spots{selected_main}', [])

        # Add a new spot to spots_list
        if spots_list:
            first_spot = spots_list[0]  # Assuming spots_list is not empty
            new_spot = [0, 0, first_spot[2], first_spot[3]]  # Example new spot
        else:
            new_spot = [10, 10, 50, 50]  # Default new spot dimensions

        spots_list.append(new_spot)

        # Update the global lists
        globals()[f'spots{selected_main}'] = spots_list
        globals()[f'spots_status{selected_main}'].append(new_spot)
        globals()[f'diffs{selected_main}'].append(new_spot)

        self.save_adjustments()  # Save adjustments or changes

    def delete_box(self):
        global MainSub
        adjust_page = self.frames["AdjustPage"]
        selected_main = adjust_page.selected_main.get()
        selected_main_int = int(selected_main)

        selected_index = globals()[f'selected_object_index{selected_main_int}']
        if selected_index is not None:
            spots_list = globals()[f'spots{selected_main_int}']
            spots_list.pop(selected_index)
            globals()[f'spots_status{selected_main_int}'].pop(selected_index)
            globals()[f'diffs{selected_main_int}'].pop(selected_index)
            globals()[f'previous_spots_status{selected_main_int}'].pop(selected_index)
            globals()[f'original_spots{selected_main_int}'].pop(selected_index)
            globals()[f'selected_object_index{selected_main_int}'] = None

            self.save_adjustments()
            print(f"Deleted spot at index {selected_index} from Main box {selected_main_int}")

            MainSub = GetStructureFromCSV(f"dataset/config_data_{filename}.csv")
            print(MainSub)
            # Update OptionMenu with new choices
            menu = adjust_page.dropdown["menu"]
            menu.delete(0, "end")  # Clear existing options
            choices = [str(sublist[0]) for sublist in MainSub] + ["new"]
            for choice in choices:
                menu.add_command(label=choice, command=tk._setit(adjust_page.selected_main, choice))
        else:
            print("No box selected to delete.")

    def delete_main_box(self):
        global MainSub
        adjust_page = self.frames["AdjustPage"]
        selected_main = adjust_page.selected_main.get()
        selected_main_int = int(selected_main) - 1

        if 0 <= selected_main_int < len(MainSub):
            # Pop-up message to confirm deletion
            confirm = messagebox.askyesno("Confirm Deletion", f"Are you sure you want to delete the item ?")
            if confirm:
                # Delete the item at selected_main_int index from MainSub
                del MainSub[selected_main_int]
                self.save_adjustments()
                print(f"Deleted from Main box {selected_main_int}")
                choices = [str(sublist[0]) for sublist in MainSub] + ["new"]
                adjust_page.selected_main.set(choices[0])  # Reset dropdown to first option

                # Update OptionMenu with new choices
                menu = adjust_page.dropdown["menu"]
                menu.delete(0, "end")  # Clear existing options
                for choice in choices:
                    menu.add_command(label=choice, command=tk._setit(adjust_page.selected_main, choice))
        else:
            print(f"Index {selected_main_int} is out of range.")


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
        def create_table_columns(table, columns):
            table['columns'] = columns
            for col in columns:
                table.column(col, anchor=tk.CENTER,
                             width=60 if col not in ['main_box', 'sub_box', 'model_file'] else 40 if col in ['main_box',
                                                                                                             'sub_box'] else 80)
                table.heading(col, text=col.replace('_', ' ').title(), anchor=tk.CENTER)
            table.column("#0", width=0, stretch=tk.NO)
            table.heading("#0", text="", anchor=tk.CENTER)

        self.config_window = tk.Toplevel(self.master)
        self.config_window.title("Configuration")
        self.config_window.geometry('420x250+0+200')
        self.config_window['bg'] = '#AC99F2'

        table_frame = tk.Frame(self.config_window)
        table_frame.pack()

        self.my_table = ttk.Treeview(table_frame)
        columns = ('main_box', 'sub_box', 'model_file', 'x', 'y', 'w', 'h')
        create_table_columns(self.my_table, columns)

        self.update_treeview()

        scrollbar = tk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.my_table.yview)
        self.my_table.configure(yscrollcommand=scrollbar.set)
        self.my_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.create_widgets()

    def create_widgets(self):
        external_padx = 10
        external_pady = 10
        internal_padx = 10
        internal_pady = 5

        # Show Index Button
        self.show_index_button = tk.Button(self, text="Show Index", command=self.controller.show_index,
                                           padx=internal_padx, pady=internal_pady)
        self.show_index_button.grid(row=0, column=0, padx=external_padx, pady=external_pady)

        # Record Video Button
        self.record_button = tk.Button(self, text="Record Video",
                                       command=lambda: self.controller.show_frame("RecordPage"),
                                       padx=internal_padx, pady=internal_pady)
        self.record_button.grid(row=1, column=1, padx=external_padx, pady=external_pady)

        # Adjust Position Button
        self.adjust_button = tk.Button(self, text="Adjust Position",
                                       command=lambda: self.controller.show_frame("AdjustPage"),
                                       padx=internal_padx, pady=internal_pady)
        self.adjust_button.grid(row=1, column=0, padx=external_padx, pady=external_pady)

        # Configuration Button
        self.config_button = tk.Button(self, text="Configuration", command=self.controller.configuration,
                                       padx=internal_padx, pady=internal_pady)
        self.config_button.grid(row=0, column=1, padx=external_padx, pady=external_pady)


class AdjustPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.selected_main = tk.StringVar(self)
        self.create_widgets()

    def create_widgets(self):
        external_padx = 10
        external_pady = 10
        internal_padx = 10
        internal_pady = 5

        # Extract the first numbers from MainSub
        choices = [str(sublist[0]) for sublist in MainSub]+ ["new"]
        self.selected_main.set(choices[0])  # Default selection
        self.dropdown = tk.OptionMenu(self, self.selected_main, *choices)

        self.dropdown.grid(row=0, column=0, padx=external_padx, pady=external_pady)

        add_button = tk.Button(self, text="Add", command=self.controller.add_box, padx=internal_padx,
                               pady=internal_pady)
        add_button.grid(row=0, column=1, padx=external_padx, pady=external_pady)

        delete_button = tk.Button(self, text="Delete Sub", command=self.controller.delete_box, padx=internal_padx,
                                  pady=internal_pady)
        delete_button.grid(row=0, column=2, padx=external_padx, pady=external_pady)

        delete_mainbutton = tk.Button(self, text="Delete Main", command=self.controller.delete_main_box, padx=internal_padx,
                                  pady=internal_pady)
        delete_mainbutton.grid(row=0, column=3, padx=external_padx, pady=external_pady)

        # Back Button to return to the StartPage
        back_button = tk.Button(self, text="Back", command=lambda: self.controller.show_frame("StartPage"),
                                padx=internal_padx, pady=internal_pady)
        back_button.grid(row=1, column=0, padx=external_padx, pady=external_pady)

        # Save Button
        start_button = tk.Button(self, text="Save", command=self.controller.save_adjustments, padx=internal_padx,
                                 pady=internal_pady)
        start_button.grid(row=1, column=1, padx=external_padx, pady=external_pady)

        # Reset Button
        stop_button = tk.Button(self, text="Reset", command=self.controller.reset_adjustments, padx=internal_padx,
                                pady=internal_pady)
        stop_button.grid(row=1, column=2, padx=external_padx, pady=external_pady)


class RecordPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.create_widgets()

    def create_widgets(self):
        external_padx = 10
        external_pady = 10
        internal_padx = 10
        internal_pady = 5

        # Start Recording Button
        start_button = tk.Button(self, text="Start Recording", command=self.controller.start_recording,
                                 padx=internal_padx, pady=internal_pady)
        start_button.pack(padx=external_padx, pady=external_pady)

        # Stop Recording Button
        stop_button = tk.Button(self, text="Stop Recording", command=self.controller.stop_recording, padx=internal_padx,
                                pady=internal_pady)
        stop_button.pack(padx=external_padx, pady=external_pady)

        # Back Button to return to the StartPage
        back_button = tk.Button(self, text="Back", command=lambda: self.controller.show_frame("StartPage"),
                                padx=internal_padx, pady=internal_pady)
        back_button.pack(padx=external_padx, pady=external_pady)


def update_gui():
    global app
    root = tk.Tk()
    root.title("Control panel")
    app = Application(master=root)
    app.mainloop()


gui_thread = threading.Thread(target=update_gui)
gui_thread.start()

frame_nmr = 0
ret = True
step = 10
cycle_counter = 1
cycle_time_start = time.time()

for S in [sublist[0] for sublist in MainSub]:
    globals()[f'selected_object_index{S}'] = None
    dragging = False
    resizing = False
    resizing_edge = None

step_size = 5
resize_margin = 10

def mouse_events(event, x, y, flags, param):
    global dragging, resizing, resizing_edge
    if app is not None and app.is_adjust:
        for S in [sublist[0] for sublist in MainSub]:
            if event == cv2.EVENT_LBUTTONDOWN:
                # Check if the click is on the edge of any object for resizing
                for i, spot in enumerate(globals()[f'spots{S}']):
                    x1, y1, w, h = spot
                    left_edge = x1
                    right_edge = x1 + w
                    top_edge = y1
                    bottom_edge = y1 + h

                    if (left_edge - resize_margin <= x <= left_edge + resize_margin and
                            top_edge <= y <= bottom_edge):
                        globals()[f'selected_object_index{S}'] = i
                        resizing = True
                        resizing_edge = 'left'
                        break
                    elif (right_edge - resize_margin <= x <= right_edge + resize_margin and
                          top_edge <= y <= bottom_edge):
                        globals()[f'selected_object_index{S}'] = i
                        resizing = True
                        resizing_edge = 'right'
                        break
                    elif (top_edge - resize_margin <= y <= top_edge + resize_margin and
                          left_edge <= x <= right_edge):
                        globals()[f'selected_object_index{S}'] = i
                        resizing = True
                        resizing_edge = 'top'
                        break
                    elif (bottom_edge - resize_margin <= y <= bottom_edge + resize_margin and
                          left_edge <= x <= right_edge):
                        globals()[f'selected_object_index{S}'] = i
                        resizing_edge = 'bottom'
                        resizing = True
                        break

                # If not resizing, check if the click is inside any object for dragging
                if not resizing:
                    for i, spot in enumerate(globals()[f'spots{S}']):
                        x1, y1, w, h = spot
                        if x1 <= x <= x1 + w and y1 <= y <= y1 + h:
                            if globals()[f'selected_object_index{S}'] == i:
                                globals()[f'selected_object_index{S}'] = None
                            else:
                                globals()[f'selected_object_index{S}'] = i
                                # print(globals()[f'selected_object_index{S}'])
                            dragging = True
                            break
                    else:
                        globals()[f'selected_object_index{S}'] = None

            elif event == cv2.EVENT_MOUSEMOVE:
                if dragging and globals()[f'selected_object_index{S}'] is not None:
                    if globals()[f'selected_object_index{S}'] is not None and dragging:
                        x1, y1, w, h = globals()[f'spots{S}'][globals()[f'selected_object_index{S}']]
                        globals()[f'spots{S}'][globals()[f'selected_object_index{S}']] = [x - w // 2, y - h // 2, w, h]
                elif resizing and globals()[f'selected_object_index{S}'] is not None:
                    x1, y1, w, h = globals()[f'spots{S}'][globals()[f'selected_object_index{S}']]
                    if resizing_edge == 'left':
                        new_w = (x1 + w) - x
                        if new_w > 10:
                            globals()[f'spots{S}'][globals()[f'selected_object_index{S}']] = [x, y1, new_w, h]
                    elif resizing_edge == 'right':
                        new_w = x - x1
                        if new_w > 10:
                            globals()[f'spots{S}'][globals()[f'selected_object_index{S}']][2] = new_w
                    elif resizing_edge == 'top':
                        new_h = (y1 + h) - y
                        if new_h > 10:
                            globals()[f'spots{S}'][globals()[f'selected_object_index{S}']] = [x1, y, w, new_h]
                    elif resizing_edge == 'bottom':
                        new_h = y - y1
                        if new_h > 10:
                            globals()[f'spots{S}'][globals()[f'selected_object_index{S}']][3] = new_h

            elif event == cv2.EVENT_LBUTTONUP:
                dragging = False
                resizing = False
                resizing_edge = None

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
        for S in [sublist[0] for sublist in MainSub]:
            for spot_indx, spot in enumerate(globals()[f'spots{S}']):
                x1, y1, w, h = spot
                spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
                globals()[f'diffs{S}'][spot_indx] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])

    if frame_nmr % step == 0:
        if previous_frame is None:
            for S in [sublist[0] for sublist in MainSub]:
                arr_ = range(len(globals()[f'spots{S}']))
                globals()[f'arr_{S}'] = arr_

        else:
            for S in [sublist[0] for sublist in MainSub]:
                diffs = globals()[f'diffs{S}']
                arr_ = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4]

                globals()[f'arr_{S}'] = arr_

        for S in [sublist[0] for sublist in MainSub]:
            model = globals().get(f'model{S}', None)
            if model is None:
                continue

            for spot_indx in globals()[f'arr_{S}']:
                spots = globals()[f'spots{S}']
                spot = spots[spot_indx]
                x1, y1, w, h = spot
                spot_crop = frame[y1:y1 + h, x1:x1 + w, :]

                spot_status = empty_or_not(spot_crop, model)
                globals()[f'spots_status{S}'][spot_indx] = spot_status

            # for spot_indx in arr_1:
            #     spot = spots1[spot_indx]
            #     x1, y1, w, h = spot
            #     spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            #     spot_status = empty_or_not(spot_crop, model1)
            #     spots_status1[spot_indx] = spot_status

            # for spot_indx in arr_2:
            #     spot = spots2[spot_indx]
            #     x1, y1, w, h = spot
            #     spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            #     spot_status = empty_or_not(spot_crop, model2)
            #     spots_status2[spot_indx] = spot_status

        # # Below is the application that is different depend on which DYXX or PX
        #
        # # if frame_nmr % step == 0:
        # not_empty_count2 = sum(1 for s in spots_status2 if s == NOT_EMPTY)
        # not_empty_count1 = sum(1 for s in spots_status1 if s == NOT_EMPTY)
        #
        # if not s_detected and not_empty_count1 < sum(1 for s in previous_spots_status1 if s == NOT_EMPTY):
        #     t_S = time.time() - start_time
        #     s_detected = True
        #     e_detected = False
        #     print("S stage detected at: {:.1f} seconds".format(t_S))
        #
        # if not e_detected and not_empty_count2 > sum(1 for s in previous_spots_status2 if s == NOT_EMPTY):
        #     t_E = time.time() - start_time
        #     e_detected = True
        #     print("E stage detected at: {:.1f} seconds".format(t_E))
        #
        #     if s_detected:
        #         assembly_time = "{:.1f} s".format(t_E - t_S)
        #         print("Assembly time {}: {}".format(cycle_counter, assembly_time))
        #         cycle_counter += 1
        #         s_detected = False
        #
        #     if t_prev_E is not None:
        #         cycle_time = "{:.1f} s".format(t_E - t_prev_E)
        #         print("Cycle time: {}".format(cycle_time))
        #
        #     t_prev_E = t_E
        #
        # # Above is the application that is different depend on which DYXX or PX

        # if frame_nmr % step == 0:
        previous_frame = frame.copy()

        for S in [sublist[0] for sublist in MainSub]:
            spots_status = globals()[f'spots_status{S}']
            globals()[f'previous_spots_status{S}'] = spots_status.copy()

            # previous_spots_status1 = spots_status1.copy()
            # previous_spots_status2 = spots_status2.copy()

    for S in [sublist[0] for sublist in MainSub]:
        spots = globals()[f'spots{S}']
        spots_status = globals()[f'spots_status{S}']

        for spot_indx, spot in enumerate(spots):
            spot_status = spots_status[spot_indx]
            x1, y1, w, h = spots[spot_indx]
            if spot_status == EMPTY:
                color = (0, 0, 255)
            elif spot_status == NOT_EMPTY:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)
            if app.show_idx:
                texts = str(S) + "/" + str(spot_indx + 1)
                textDim = cv2.getTextSize(texts, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 2)
                cv2.putText(frame, texts, (int(x1 + w // 2 - (textDim[0][0])/2), int(y1 + h // 2 + (textDim[0][1])/2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
            if globals()[f'selected_object_index{S}'] == spot_indx:
                cv2.rectangle(frame, (x1 - 2, y1 - 2), (x1 + w + 2, y1 + h + 2), (0, 255, 255), 3)

    current_time = datetime.now().strftime("%H:%M:%S")
    if app.is_adjust:
        cv2.putText(frame, "Adjust position enable", (frame.shape[1] - 230, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, current_time, (frame.shape[1] - 200, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                cv2.LINE_AA)

    cv2.imshow('Monitor', frame)

    if app is not None and app.is_recording:
        if app.video_writer is not None:
            app.video_writer.write(copy_frame)


    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    frame_nmr += 1

cap.release()
cv2.destroyAllWindows()