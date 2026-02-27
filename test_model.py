"""
Quick test script to evaluate the trained ambulance detection model
"""
from ultralytics import YOLO
from pathlib import Path
import os

def test_model():
    """Test the trained model on test dataset"""
    model_path = Path("runs/detect/ambulance_detector/weights/best.pt")
    
    if not model_path.exists():
        print("ERROR: Trained model not found!")
        print("Please train the model first using train.py")
        return
    
    print("Loading trained model...")
    model = YOLO(str(model_path))
    
    # Test on a few sample images
    test_dir = Path("dataset/test")
    test_images = list(test_dir.glob("*.jpg"))[:5]  # Test first 5 images
    
    if not test_images:
        print("No test images found!")
        return
    
    print(f"\nTesting on {len(test_images)} sample images...")
    print("="*60)
    
    total_detections = 0
    for img_path in test_images:
        results = model.predict(
            source=str(img_path),
            conf=0.25,
            verbose=False
        )
        
        detections = len(results[0].boxes)
        total_detections += detections
        
        if detections > 0:
            confidences = [f"{box.conf[0].item()*100:.1f}%" for box in results[0].boxes]
            print(f"✓ {img_path.name}: {detections} ambulance(s) detected - Confidence: {', '.join(confidences)}")
        else:
            print(f"✗ {img_path.name}: No ambulance detected")
    
    print("="*60)
    print(f"\nSummary: Detected ambulances in {sum(1 for _ in test_images if len(model.predict(str(_), conf=0.25, verbose=False)[0].boxes) > 0)}/{len(test_images)} images")
    print(f"Average detections per image: {total_detections/len(test_images):.2f}")
    
    # Run full validation
    print("\nRunning full validation on test set...")
    metrics = model.val(data="dataset/data.yaml", split="test")
    print(f"\nTest Set Performance:")
    print(f"  Precision: {float(metrics.box.p):.4f}")
    print(f"  Recall: {float(metrics.box.r):.4f}")
    print(f"  mAP50: {float(metrics.box.map50):.4f}")
    print(f"  mAP50-95: {float(metrics.box.map):.4f}")

if __name__ == "__main__":
    test_model()
