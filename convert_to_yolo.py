import os
import pandas as pd
from pathlib import Path

def convert_bbox_to_yolo(xmin, ymin, xmax, ymax, img_width, img_height):
    """Convert bounding box from Pascal VOC to YOLO format"""
    # Calculate center coordinates and dimensions
    x_center = ((xmin + xmax) / 2) / img_width
    y_center = ((ymin + ymax) / 2) / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    
    return x_center, y_center, width, height

def convert_csv_to_yolo(csv_path, images_dir, output_dir, class_name="ambulance"):
    """Convert CSV annotations to YOLO format"""
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique class names from CSV (handles both "ambulance" and "emergency-vehicle")
    unique_classes = df['class'].unique()
    print(f"Found classes in CSV: {unique_classes}")
    
    # Class mapping (map all classes to 0 for ambulance detection)
    class_map = {cls: 0 for cls in unique_classes}
    
    # Group by filename
    grouped = df.groupby('filename')
    
    for filename, group in grouped:
        # Get image dimensions (should be same for all annotations of same image)
        img_width = group.iloc[0]['width']
        img_height = group.iloc[0]['height']
        
        # Create output annotation file
        base_name = os.path.splitext(filename)[0]
        txt_filename = base_name + '.txt'
        txt_path = os.path.join(output_dir, txt_filename)
        
        # Convert all bounding boxes for this image
        with open(txt_path, 'w') as f:
            for _, row in group.iterrows():
                xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
                class_name = row['class']
                
                # Convert to YOLO format
                x_center, y_center, width, height = convert_bbox_to_yolo(
                    xmin, ymin, xmax, ymax, img_width, img_height
                )
                
                # Get class ID
                class_id = class_map.get(class_name, 0)
                
                # Write to file (format: class_id x_center y_center width height)
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        print(f"Converted: {filename} -> {txt_filename}")

if __name__ == "__main__":
    # Paths
    base_dir = Path("dataset")
    
    # Convert train, valid, and test sets
    for split in ['train', 'valid', 'test']:
        csv_path = base_dir / split / "_annotations.csv"
        images_dir = base_dir / split
        output_dir = base_dir / split
        
        if csv_path.exists():
            print(f"\nConverting {split} set...")
            convert_csv_to_yolo(csv_path, images_dir, output_dir)
            print(f"✓ {split} set converted successfully!")
        else:
            print(f"⚠ Warning: {csv_path} not found, skipping {split} set")
