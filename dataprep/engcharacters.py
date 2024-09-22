import os
from PIL import Image
import numpy as np

def process_image(input_path, output_path):
    """
    Processes a binary image:
    - Crops it to a square area containing the foreground object.
    - Resizes it to 100x100 pixels with smoothing.
    - Saves the processed image.
    """
    # Load the image and convert to grayscale
    img = Image.open(input_path).convert('L')
    
    # Convert image to numpy array
    img_np = np.array(img)
    
    # Assuming background is white (255) and foreground is black (0)
    # Find coordinates of foreground pixels
    foreground_pixels = np.argwhere(img_np < 255)
    
    if foreground_pixels.size == 0:
        # No foreground pixels found
        print(f"No foreground found in {input_path}. Skipping.")
        return
    
    # Get the bounding box of the foreground object
    y_min, x_min = foreground_pixels.min(axis=0)
    y_max, x_max = foreground_pixels.max(axis=0)
    
    # Compute the center of the bounding box
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    
    # Compute the size of the square crop area
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    square_size = max(bbox_width, bbox_height)
    
    # Optionally, add a margin around the object (e.g., 10%)
    margin = 0.1  # 10% margin
    square_size = int(square_size * (1 + margin))
    half_size = square_size / 2
    
    # Image dimensions
    img_width, img_height = img.size
    
    # Calculate the crop coordinates
    x_start = x_center - half_size
    y_start = y_center - half_size
    x_end = x_center + half_size
    y_end = y_center + half_size
    
    # Adjust the crop area if it goes beyond the image boundaries
    if x_start < 0:
        x_end -= x_start  # Shift x_end to maintain size
        x_start = 0
    if y_start < 0:
        y_end -= y_start
        y_start = 0
    if x_end > img_width:
        x_start -= x_end - img_width
        x_end = img_width
    if y_end > img_height:
        y_start -= y_end - img_height
        y_end = img_height
    
    # Ensure the adjusted coordinates are within image boundaries
    x_start = max(0, x_start)
    y_start = max(0, y_start)
    x_end = min(img_width, x_end)
    y_end = min(img_height, y_end)
    
    # Convert coordinates to integers
    x_start = int(round(x_start))
    y_start = int(round(y_start))
    x_end = int(round(x_end))
    y_end = int(round(y_end))
    
    # Crop the image
    img_cropped = img.crop((x_start, y_start, x_end, y_end))

    # Resize the image to 100x100 pixels with smoothing
    img_resized = img_cropped.resize((100, 100), Image.LANCZOS)
    
    # Save the processed image
    img_resized.save(output_path)

# Directories containing input and output images
input_dir = 'path_to_input_images'
output_dir = 'path_to_output_images'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process each image in the input directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        process_image(input_path, output_path)