"""
Validate and Monitor Ambulance Detection Model Accuracy
Comprehensive metrics for model performance evaluation
"""

from ultralytics import YOLO
from pathlib import Path
import json
from datetime import datetime
import os

def validate_model(model_path="runs/detect/ambulance_detector/weights/best.pt"):
    """Validate model and display comprehensive metrics"""
    
    print("\n" + "="*60)
    print("AMBULANCE DETECTION MODEL - VALIDATION REPORT")
    print("="*60)
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Please train the model first using: python train.py")
        return None
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    model = YOLO(model_path)
    
    # Run validation
    print("\nValidating model on test set...")
    metrics = model.val()
    
    # Extract key metrics
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map_95 = float(metrics.box.map)
    
    print("\n" + "-"*60)
    print("KEY PERFORMANCE METRICS")
    print("-"*60)
    print(f"✅ Precision: {precision:.4f} (TP / (TP+FP)) - Low false positives")
    print(f"✅ Recall:    {recall:.4f} (TP / (TP+FN)) - Catches ambulances")
    print(f"✅ mAP50:     {map50:.4f} (50% IoU threshold)")
    print(f"✅ mAP50-95:  {map_95:.4f} (Strict accuracy metric)")
    
    # Interpret results
    print("\n" + "-"*60)
    print("INTERPRETATION")
    print("-"*60)
    
    if map50 > 0.85:
        print("🟢 EXCELLENT accuracy! Model ready for production.")
    elif map50 > 0.70:
        print("🟡 GOOD accuracy. Consider collecting more diverse data.")
    else:
        print("🔴 NEEDS IMPROVEMENT. Retrain with more data/epochs.")
    
    if precision > 0.90:
        print("🟢 LOW FALSE POSITIVES - Few unnecessary traffic light activations")
    elif precision > 0.75:
        print("🟡 MODERATE FALSE POSITIVES - May trigger lights occasionally")
    else:
        print("🔴 HIGH FALSE POSITIVES - Consider increasing confidence threshold")
    
    if recall > 0.85:
        print("🟢 HIGH RECALL - Catches most ambulances")
    elif recall > 0.70:
        print("🟡 MODERATE RECALL - May miss some ambulances")
    else:
        print("🔴 LOW RECALL - Missing ambulances, needs more training data")
    
    print("\n" + "-"*60)
    print("RECOMMENDATIONS")
    print("-"*60)
    
    recommendations = []
    
    if precision < 0.80:
        recommendations.append("• Increase confidence threshold (currently 0.75) during inference")
        recommendations.append("• Collect negative examples (fire trucks, police cars, etc.)")
    
    if recall < 0.80:
        recommendations.append("• Collect more diverse ambulance images")
        recommendations.append("• Increase training epochs in train.py")
        recommendations.append("• Add more augmentation variations")
    
    if map50 < 0.75:
        recommendations.append("• Use larger model (yolov8m instead of yolov8s)")
        recommendations.append("• Retrain with more epochs (200+)")
        recommendations.append("• Check dataset for labeling errors")
    
    if recommendations:
        for rec in recommendations:
            print(rec)
    else:
        print("✅ Model performance is excellent!")
    
    # Log metrics
    log_metrics(precision, recall, map50, map_95)
    
    # Test on sample images
    print("\n" + "-"*60)
    print("TESTING ON SAMPLE IMAGES")
    print("-"*60)
    
    test_dir = Path("dataset/test")
    if test_dir.exists():
        test_images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
        if test_images:
            print(f"\nTesting on {len(test_images)} test images...")
            results = model.predict(source=str(test_dir), conf=0.75, verbose=False)
            
            for result in results[:5]:  # Show first 5 results
                ambulances = len(result.boxes) if result.boxes is not None else 0
                print(f"  {Path(result.path).name}: {ambulances} ambulance(s) detected")
    
    print("\n" + "="*60)
    print("Validation complete!")
    print("="*60 + "\n")
    
    return {
        'precision': precision,
        'recall': recall,
        'map50': map50,
        'map_95': map_95
    }

def log_metrics(precision, recall, map50, map_95):
    """Log metrics to file for historical tracking"""
    
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'model': 'yolov8s',
        'confidence_threshold': 0.75,
        'iou_threshold': 0.55,
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'mAP50': round(map50, 4),
        'mAP50-95': round(map_95, 4)
    }
    
    log_file = 'accuracy_log.json'
    
    # Append to log file
    try:
        with open(log_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
        print(f"\n✅ Metrics logged to {log_file}")
    except Exception as e:
        print(f"Warning: Could not log metrics: {e}")

def compare_models(old_model, new_model):
    """Compare performance of two models"""
    
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    old_metrics = validate_model(old_model)
    print("\n" + "-"*60 + "\n")
    new_metrics = validate_model(new_model)
    
    if old_metrics and new_metrics:
        print("\nIMPROVEMENT SUMMARY:")
        print("-"*60)
        
        precision_gain = (new_metrics['precision'] - old_metrics['precision']) * 100
        recall_gain = (new_metrics['recall'] - old_metrics['recall']) * 100
        map50_gain = (new_metrics['map50'] - old_metrics['map50']) * 100
        
        print(f"Precision: {precision_gain:+.2f}%")
        print(f"Recall:    {recall_gain:+.2f}%")
        print(f"mAP50:     {map50_gain:+.2f}%")
        
        total_gain = (precision_gain + recall_gain + map50_gain) / 3
        print(f"\nAverage Improvement: {total_gain:+.2f}%")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 2:
        # Compare two models
        old_model = sys.argv[1]
        new_model = sys.argv[2]
        compare_models(old_model, new_model)
    else:
        # Validate single model
        model_path = sys.argv[1] if len(sys.argv) > 1 else "runs/detect/ambulance_detector/weights/best.pt"
        validate_model(model_path)
