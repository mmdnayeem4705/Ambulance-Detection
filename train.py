"""
YOLO Training Script for Ambulance Detection
This script trains a YOLOv8 model on the ambulance detection dataset
"""
from ultralytics import YOLO
import os
import torch
from pathlib import Path

def train_ambulance_detector():
    """Train YOLO model for ambulance detection"""
    # Check if dataset configuration exists
    config_path = Path("dataset/data.yaml")
    if not config_path.exists():
        print("ERROR: dataset/data.yaml not found!")
        print("Please run convert_to_yolo.py first to prepare the dataset.")
        return
    
    # Check if we can resume from a previous training
    resume_path = Path("runs/detect/ambulance_detector/weights/last.pt")
    if resume_path.exists():
        print(f"Found previous training checkpoint at {resume_path}")
        print("Resuming training from checkpoint...")
        model = YOLO(str(resume_path))  # Resume from checkpoint
        resume = True
    else:
        print("Initializing YOLOv8s model (improved accuracy over nano)...")
        model = YOLO("yolov8s.pt")  # Use 'small' variant for better accuracy (was yolov8n - nano)
        resume = False
    
    # Auto-detect device (GPU or CPU)
    if torch.cuda.is_available():
        device = 0
        batch_size = 32  # Larger batch for GPU
        print("Using GPU for training")
    else:
        device = "cpu"
        batch_size = 16  # Larger batch than before
        print("Using CPU for training (this will be slower)")
    
    # Training parameters (optimized for accuracy)
    if device == "cpu":
        epochs = 100  # More epochs for better convergence
        patience = 30  # Later early stopping
        imgsz = 640   # Full resolution images
        print("CPU mode: Using 100 epochs, 640px images for better accuracy")
    else:
        epochs = 200   # More epochs on GPU
        patience = 50  # Later early stopping
        imgsz = 640    # Full resolution
    
    print("Starting training with improved hyperparameters...")
    results = model.train(
        data=str(config_path),           # Path to dataset config
        epochs=epochs,                    # More epochs
        imgsz=imgsz,                      # Full image size
        batch=batch_size,                 # Larger batch
        name="ambulance_detector",        # Project name
        patience=patience,                # Later early stopping
        save=True,                        # Save checkpoints
        plots=True,                       # Generate training plots
        val=True,                         # Validate during training
        device=device,                    # Auto-detected device
        workers=8 if device != "cpu" else 2,
        project="runs/detect",            # Project directory
        exist_ok=True,                    # Overwrite existing project
        resume=resume,                    # Resume from checkpoint if available
        # Improved hyperparameters
        optimizer='SGD',                  # SGD often better than default Adam
        lr0=0.01,                         # Initial learning rate
        lrf=0.01,                         # Final LR ratio
        momentum=0.937,                   # SGD momentum
        weight_decay=0.0005,              # Regularization
        warmup_epochs=3,                  # Warmup period
        mosaic=1.0,                       # Data augmentation - mixup
        mixup=0.1,                        # Image mixing
        augment=True,                     # Enable augmentation
        flipud=0.5,                       # Vertical flip probability
        fliplr=0.5,                       # Horizontal flip probability
        degrees=10,                       # Random rotation degrees
        translate=0.1,                    # Random translation
        scale=0.5,                        # Random scale
        hsv_h=0.015,                      # HSV hue augmentation
        hsv_s=0.7,                        # HSV saturation augmentation
        hsv_v=0.4,                        # HSV value augmentation
    )
    
    print("\nTraining completed!")
    print(f"Model saved in: {results.save_dir}")
    print(f"Best model: {results.save_dir}/weights/best.pt")
    print(f"Last model: {results.save_dir}/weights/last.pt")
    
    # Validate the model
    print("\nRunning detailed validation...")
    metrics = model.val()
    print(f"Precision: {metrics.box.mp:.4f}")      # TP / (TP+FP)
    print(f"Recall: {metrics.box.mr:.4f}")         # TP / (TP+FN)
    print(f"mAP50: {metrics.box.map50:.4f}")       # 50% IoU accuracy
    print(f"mAP50-95: {metrics.box.map:.4f}")      # Strict accuracy metric
    
    return model, results

if __name__ == "__main__":
    train_ambulance_detector()
