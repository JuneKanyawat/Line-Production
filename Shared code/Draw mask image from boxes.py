from PIL import Image, ImageDraw
import os

def create_box_mask(boxes, image_size, background_color, output_path):
    """
    Create an RGB mask image with white boxes and a background color.
    
    Parameters:
        boxes (list): List of boxes where each box is defined as [x, y, w, h].
        image_size (tuple): Size of the output image (width, height).
        background_color (tuple): Background color as an (R, G, B) tuple.
        output_path (str): Path to save the generated image.
    """
    # Create a new image with the background color
    image = Image.new("RGB", image_size, background_color)
    draw = ImageDraw.Draw(image)
    
    # Draw each box on the image
    for box in boxes:
        x, y, w, h = box
        draw.rectangle([x, y, x + w, y + h], fill=(255, 255, 255))  # White boxes
    
    # Save the image to the specified directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)
    print(f"Image saved to {output_path}")

# Example usage
boxes = [
    [[10, 10, 50, 50], [100, 100, 30, 60]],  # First set of boxes
    [[200, 200, 40, 40]]                    # Second set of boxes
]

# Flatten the list of boxes
flattened_boxes = [box for sublist in boxes for box in sublist]

# Define image size and background color
image_size = (300, 300)
background_color = (0, 0, 0)  # Black

# Output path
output_path = "output/mask_image.png"

# Create and save the image
create_box_mask(flattened_boxes, image_size, background_color, output_path)
