import cv2

# Initializing variables
dragging = False
selected_object_index = None
objects = [{'position': (100, 100), 'size': (40, 30), 'color': (0, 255, 0)},
           {'position': (200, 200), 'size': (50, 40), 'color': (255, 0, 0)}]
step_size = 5  # Step size for keyboard adjustments

# Mouse callback function
def mouse_events(event, x, y, flags, param):
    global dragging, selected_object_index, objects
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if the click is inside any object
        for i, obj in enumerate(objects):
            ox, oy = obj['position']
            w, h = obj['size']
            if ox - w // 2 <= x <= ox + w // 2 and oy - h // 2 <= y <= oy + h // 2:
                if selected_object_index == i:
                    # If the object is already selected, unselect it
                    selected_object_index = None
                else:
                    # Select the new object
                    selected_object_index = i
                dragging = True
                break
        else:
            # If no object was selected, clear the selection
            selected_object_index = None
    
    elif event == cv2.EVENT_MOUSEMOVE:
        # If mouse is moving, update the position of the object if dragging
        if dragging and selected_object_index is not None:
            objects[selected_object_index]['position'] = (x, y)
            
    elif event == cv2.EVENT_LBUTTONUP:
        # If left button is released, stop dragging
        dragging = False

# Create a blank image
image = cv2.imread('path_to_your_image.jpg')  # Replace with your image path
original_image = image.copy()

# Create a window and set the mouse callback
cv2.namedWindow('Interactive Window')
cv2.setMouseCallback('Interactive Window', mouse_events)

while True:
    # Copy the original image to work on
    display_image = original_image.copy()
    
    # Draw the objects
    for i, obj in enumerate(objects):
        x, y = obj['position']
        w, h = obj['size']
        if i == selected_object_index:
            # Highlight the selected object by drawing a border around it
            cv2.rectangle(display_image, (x - w // 2 - 5, y - h // 2 - 5), 
                          (x + w // 2 + 5, y + h // 2 + 5), (0, 255, 255), 2)  # Add a border
        cv2.rectangle(display_image, (x - w // 2, y - h // 2), 
                      (x + w // 2, y + h // 2), obj['color'], 2)  # Border only

    # Show the image
    cv2.imshow('Interactive Window', display_image)
    
    # Handle keyboard inputs for fine-tuning position
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif selected_object_index is not None:
        if key == ord('w'):  # Up key
            x, y = objects[selected_object_index]['position']
            objects[selected_object_index]['position'] = (x, y - step_size)
        elif key == ord('s'):  # Down key
            x, y = objects[selected_object_index]['position']
            objects[selected_object_index]['position'] = (x, y + step_size)
        elif key == ord('a'):  # Left key
            x, y = objects[selected_object_index]['position']
            objects[selected_object_index]['position'] = (x - step_size, y)
        elif key == ord('d'):  # Right key
            x, y = objects[selected_object_index]['position']
            objects[selected_object_index]['position'] = (x + step_size, y)

# Clean up
cv2.destroyAllWindows()
