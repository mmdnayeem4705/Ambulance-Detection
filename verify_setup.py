"""
Verify that the dataset is properly set up for YOLO training
"""
from pathlib import Path
import os

def verify_setup():
    """Verify dataset structure and files"""
    print("Verifying YOLO dataset setup...\n")
    
    issues = []
    warnings = []
    
    # Check dataset directory
    dataset_dir = Path("dataset")
    if not dataset_dir.exists():
        issues.append("ERROR: 'dataset' directory not found!")
        return issues, warnings
    
    print("OK: Dataset directory exists")
    
    # Check data.yaml
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        issues.append("ERROR: 'dataset/data.yaml' not found!")
    else:
        print("OK: data.yaml exists")
    
    # Check train, valid, test directories
    for split in ['train', 'valid', 'test']:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            warnings.append(f"WARNING: '{split}' directory not found")
            continue
        
        # Count images
        images = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png"))
        txt_files = list(split_dir.glob("*.txt"))
        
        print(f"OK: {split}: {len(images)} images, {len(txt_files)} annotation files")
        
        if len(images) == 0:
            warnings.append(f"WARNING: No images found in {split}/")
        
        if len(txt_files) == 0:
            warnings.append(f"WARNING: No YOLO annotation files (.txt) found in {split}/")
            warnings.append(f"  -> Run 'python convert_to_yolo.py' to create annotations")
        
        # Check if annotation count matches image count
        if len(images) > 0 and len(txt_files) > 0:
            if len(images) != len(txt_files):
                warnings.append(f"WARNING: {split}: Image count ({len(images)}) doesn't match annotation count ({len(txt_files)})")
    
    # Check requirements
    req_path = Path("requirements.txt")
    if req_path.exists():
        print("OK: requirements.txt exists")
    else:
        warnings.append("WARNING: requirements.txt not found")
    
    # Summary
    print("\n" + "="*50)
    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
        print("\nPlease fix these issues before training.")
    
    if warnings:
        print("\nWARNINGS:")
        for warning in warnings:
            print(f"  {warning}")
    
    if not issues and not warnings:
        print("All checks passed! Ready to train.")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Train model: python train.py")
    
    return issues, warnings

if __name__ == "__main__":
    verify_setup()
