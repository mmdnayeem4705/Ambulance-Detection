"""
Dataset Augmentation for Ambulance Detection
Adds diversity to training data (lighting, angles, weather conditions)
"""

import cv2
import numpy as np
from pathlib import Path
import random
import shutil
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

def augment_image(image, annotation):
    """Apply random augmentations to image while preserving bounding boxes"""
    
    # Define augmentation pipeline
    transform = A.Compose([
        # Geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=15, p=0.5),
        A.Affine(scale=(0.8, 1.2), translate_percent=(-0.1, 0.1), p=0.5),
        
        # Weather effects
        A.RandomRain(p=0.1),
        A.RandomFog(p=0.1),
        A.GaussNoise(p=0.1),
        
        # Lighting changes
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.RandomGamma(p=0.3),
        A.GaussBlur(blur_limit=3, p=0.1),
        
        # Color augmentations
        A.Blur(blur_limit=3, p=0.1),
        A.MotionBlur(blur_limit=3, p=0.1),
        A.ISONoise(p=0.1),
        
        # HSV adjustments
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    
    # Apply augmentation
    try:
        height, width = image.shape[:2]
        bboxes = parse_annotation(annotation, width, height)
        class_labels = [0] * len(bboxes)  # All are ambulance (class 0)
        
        if bboxes:
            transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            augmented_image = transformed['image']
            augmented_bboxes = transformed['bboxes']
            
            # Convert bboxes back to YOLO format
            new_annotation = bboxes_to_yolo_annotation(augmented_bboxes, width, height)
            return augmented_image, new_annotation
    except Exception as e:
        print(f"Warning: Augmentation failed: {e}")
    
    return image, annotation

def parse_annotation(annotation_file, img_width, img_height):
    """Parse YOLO annotation (x_center, y_center, width, height in normalized coords) to pascal_voc format"""
    
    try:
        bboxes = []
        with open(annotation_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # YOLO format: class_id x_center y_center width height (normalized)
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * img_width
                    y_center = float(parts[2]) * img_height
                    width = float(parts[3]) * img_width
                    height = float(parts[4]) * img_height
                    
                    # Convert to pascal_voc format: (x_min, y_min, x_max, y_max)
                    x_min = x_center - width / 2
                    y_min = y_center - height / 2
                    x_max = x_center + width / 2
                    y_max = y_center + height / 2
                    
                    # Clip to image boundaries
                    x_min = max(0, min(img_width, x_min))
                    y_min = max(0, min(img_height, y_min))
                    x_max = max(0, min(img_width, x_max))
                    y_max = max(0, min(img_height, y_max))
                    
                    bboxes.append((x_min, y_min, x_max, y_max))
        
        return bboxes
    except Exception as e:
        print(f"Error parsing annotation: {e}")
        return []

def bboxes_to_yolo_annotation(bboxes, img_width, img_height):
    """Convert pascal_voc bboxes back to YOLO format"""
    
    annotation = ""
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        
        # Convert to YOLO format
        x_center = ((x_min + x_max) / 2) / img_width
        y_center = ((y_min + y_max) / 2) / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        
        # Clamp to [0, 1]
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        annotation += f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
    
    return annotation

def augment_dataset(input_dir="dataset/train", output_dir="dataset/train_augmented", augmentations_per_image=3):
    """Create augmented copies of training images"""
    
    print("\n" + "="*60)
    print("DATASET AUGMENTATION")
    print("="*60)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory not found: {input_path}")
        return
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
    print(f"\nFound {len(image_files)} images in {input_dir}")
    print(f"Generating {augmentations_per_image} variations per image...")
    
    augmented_count = 0
    
    for img_file in image_files:
        # Read image
        image = cv2.imread(str(img_file))
        if image is None:
            print(f"Warning: Could not read image {img_file}")
            continue
        
        # Get corresponding annotation
        ann_file = img_file.with_suffix('.txt')
        if not ann_file.exists():
            print(f"Warning: Annotation not found for {img_file}")
            continue
        
        base_name = img_file.stem
        
        # Create augmented versions
        for i in range(augmentations_per_image):
            aug_image, aug_annotation = augment_image(image, str(ann_file))
            
            # Save augmented image
            aug_img_name = f"{base_name}_aug{i}.jpg"
            aug_img_path = output_path / aug_img_name
            cv2.imwrite(str(aug_img_path), aug_image)
            
            # Save augmented annotation
            aug_ann_name = f"{base_name}_aug{i}.txt"
            aug_ann_path = output_path / aug_ann_name
            with open(aug_ann_path, 'w') as f:
                f.write(aug_annotation)
            
            augmented_count += 1
    
    print(f"\n✅ Generated {augmented_count} augmented images in {output_dir}")
    print("\nNext steps:")
    print("1. Copy augmented images to dataset/train/")
    print("2. Update data.yaml to use augmented dataset")
    print("3. Retrain model: python train.py")

def merge_augmented_to_train(augmented_dir="dataset/train_augmented", train_dir="dataset/train"):
    """Merge augmented images into training directory"""
    
    print(f"\nMerging augmented images from {augmented_dir} to {train_dir}...")
    
    aug_path = Path(augmented_dir)
    train_path = Path(train_dir)
    
    if not aug_path.exists():
        print(f"Error: Augmented directory not found: {aug_path}")
        return
    
    train_path.mkdir(parents=True, exist_ok=True)
    
    # Copy augmented files
    aug_images = list(aug_path.glob("*.jpg")) + list(aug_path.glob("*.png"))
    aug_annotations = list(aug_path.glob("*.txt"))
    
    print(f"Copying {len(aug_images)} images and {len(aug_annotations)} annotations...")
    
    for img_file in aug_images:
        shutil.copy(str(img_file), str(train_path / img_file.name))
    
    for ann_file in aug_annotations:
        shutil.copy(str(ann_file), str(train_path / ann_file.name))
    
    print(f"✅ Merged {len(aug_images)} augmented images into {train_dir}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "merge":
        merge_augmented_to_train()
    else:
        # Augment dataset
        num_augmentations = int(sys.argv[1]) if len(sys.argv) > 1 else 3
        augment_dataset(augmentations_per_image=num_augmentations)
        
        print("\n" + "="*60)
        print("AUGMENTATION RECOMMENDATIONS")
        print("="*60)
        print("To incorporate augmented data:")
        print(f"  python augment_dataset.py merge")
        print("\nThen retrain:")
        print("  python train.py")
        print("="*60)
