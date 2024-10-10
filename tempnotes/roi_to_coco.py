import os
import json
import argparse
from roifile import ImagejRoi
from PIL import Image
import re

def extract_number(filename):
    """
    Extracts the first occurring number in a filename for sorting purposes.
    """
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else -1

def get_sorted_image_files(image_folderpath):
    """
    Retrieves and sorts image files numerically based on numbers in filenames.
    """
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    files = [f for f in os.listdir(image_folderpath) if os.path.splitext(f.lower())[1] in supported_extensions]
    files.sort(key=extract_number)
    return files

def read_rois(roi_filepath):
    """
    Reads ROIs from a .zip file using roifile.
    Assumes all ROIs are rectangles.
    """
    rois = []
    with ImagejRoi(roi_filepath) as roi_zip:
        for roi in roi_zip:
            if roi.shape != 'rectangle':
                continue  # Skip non-rectangle ROIs
            rois.append({
                'z_position': roi.z,  # Assuming z starts at 1
                'x': roi.left,
                'y': roi.top,
                'width': roi.width,
                'height': roi.height
            })
    return rois

def create_coco_annotations(image_folderpath, roi_filepath, output_json_path=None, category_name="d1"):
    """
    Converts ImageJ ROIs to COCO-format annotations.
    """
    # Set default output path if not provided
    if output_json_path is None:
        output_json_path = os.path.join(os.path.dirname(roi_filepath), 'annotations.json')
    
    # Get sorted image files
    image_files = get_sorted_image_files(image_folderpath)
    
    if not image_files:
        raise ValueError("No supported image files found in the specified image folder.")
    
    # Read ROIs
    rois = read_rois(roi_filepath)
    
    # Organize ROIs by z_position
    rois_by_z = {}
    for roi in rois:
        z = roi['z_position']
        if z not in rois_by_z:
            rois_by_z[z] = []
        rois_by_z[z].append(roi)
    
    # Initialize COCO structure
    coco = {
        "info": {
            "description": "Converted from ImageJ ROIs",
            "version": "1.0",
            "year": 2024,
            "contributor": "",
            "date_created": "2024-10-09"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Add category
    category = {
        "id": 1,
        "name": category_name,
        "supercategory": "none"
    }
    coco["categories"].append(category)
    
    annotation_id = 1  # Start annotation IDs at 1
    for idx, filename in enumerate(image_files):
        image_path = os.path.join(image_folderpath, filename)
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Error opening image {filename}: {e}")
            continue
        
        image_id = idx + 1  # COCO image IDs start at 1
        coco_image = {
            "id": image_id,
            "file_name": filename,
            "height": height,
            "width": width
        }
        coco["images"].append(coco_image)
        
        # z_position is assumed to correspond to image_id
        z = image_id  # Adjust if z_position starts at 0 in your ROI files
        if z in rois_by_z:
            for roi in rois_by_z[z]:
                # Ensure ROI is within image boundaries
                x = max(0, roi['x'])
                y = max(0, roi['y'])
                width_roi = min(roi['width'], width - x)
                height_roi = min(roi['height'], height - y)
                
                if width_roi <=0 or height_roi <=0:
                    print(f"Invalid ROI for image {filename}: {roi}")
                    continue
                
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x, y, width_roi, height_roi],
                    "area": width_roi * height_roi,
                    "iscrowd": 0
                }
                coco["annotations"].append(annotation)
                annotation_id += 1
    
    # Save COCO JSON
    with open(output_json_path, 'w') as json_file:
        json.dump(coco, json_file, indent=4)
    
    print(f"COCO annotations saved to {output_json_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert ImageJ ROIs to COCO-format annotations.")
    parser.add_argument('--image_folderpath', type=str, required=True, help='Path to the input image folder.')
    parser.add_argument('--roi_filepath', type=str, required=True, help='Path to the .zip file of ROIs saved from ImageJ.')
    parser.add_argument('--output_json_path', type=str, default=None, help='Path to save the output COCO JSON file. Defaults to the same folder as roi_filepath.')
    parser.add_argument('--category_name', type=str, default="d1", help='Name of the category. Default is "d1".')
    
    args = parser.parse_args()
    if len(args) == 0:
        args = argparse.Namespace() # For when used as a script
    
    create_coco_annotations(
        image_folderpath=args.image_folderpath,
        roi_filepath=args.roi_filepath,
        output_json_path=args.output_json_path,
        category_name=args.category_name
    )

if __name__ == "__main__":
    main()
