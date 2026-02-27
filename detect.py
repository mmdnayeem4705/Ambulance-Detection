"""
Ambulance Detection Script
This script performs real-time or batch detection on images/videos using the trained YOLO model
"""
from ultralytics import YOLO
import cv2
import argparse
from pathlib import Path
import os

def detect_ambulance_image(model_path, image_path, output_dir="detections", conf_threshold=0.25):
    """Detect ambulances in a single image"""
    # Load the trained model
    model = YOLO(model_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run inference
    results = model.predict(
        source=image_path,
        save=True,
        save_txt=True,
        conf=conf_threshold,
        project=output_dir,
        name="predictions"
    )
    
    # Display results
    for result in results:
        print(f"\n📸 Image: {result.path}")
        print(f"🚑 Detected {len(result.boxes)} ambulance(s)")
        
        for i, box in enumerate(result.boxes):
            conf = box.conf[0].item()
            print(f"  Ambulance {i+1}: Confidence = {conf:.2%}")
    
    return results

def detect_ambulance_video(model_path, video_path, output_dir="detections", conf_threshold=0.25):
    """Detect ambulances in a video"""
    # Load the trained model
    model = YOLO(model_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run inference
    results = model.predict(
        source=video_path,
        save=True,
        conf=conf_threshold,
        project=output_dir,
        name="predictions"
    )
    
    print(f"\n✅ Video processed and saved to {output_dir}/predictions/")
    return results

def detect_ambulance_webcam(model_path, conf_threshold=0.25):
    """Real-time ambulance detection using webcam"""
    # Load the trained model
    model = YOLO(model_path)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return
    
    print("📹 Starting webcam detection...")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference
        results = model.predict(
            source=frame,
            conf=conf_threshold,
            verbose=False
        )
        
        # Draw results on frame
        annotated_frame = results[0].plot()
        
        # Display frame
        cv2.imshow("Ambulance Detection", annotated_frame)
        
        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Ambulance Detection using YOLO")
    parser.add_argument("--model", type=str, default="runs/detect/ambulance_detector/weights/best.pt",
                        help="Path to trained model weights")
    parser.add_argument("--source", type=str, required=True,
                        help="Path to image, video, or 'webcam' for live detection")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (0-1)")
    parser.add_argument("--output", type=str, default="detections",
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"ERROR: Model not found at {args.model}")
        print("Please train the model first using train.py")
        return
    
    # Check if source file exists (unless it's webcam)
    if args.source.lower() != "webcam" and not Path(args.source).exists():
        print(f"ERROR: Source file not found: {args.source}")
        print("Please provide a valid path to an image or video file")
        return
    
    # Run detection based on source type
    if args.source.lower() == "webcam":
        detect_ambulance_webcam(args.model, args.conf)
    elif args.source.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        detect_ambulance_image(args.model, args.source, args.output, args.conf)
    elif args.source.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        detect_ambulance_video(args.model, args.source, args.output, args.conf)
    else:
        print(f"ERROR: Unsupported source type: {args.source}")
        print("Supported: images (.jpg, .png, etc.), videos (.mp4, .avi, etc.), or 'webcam'")

if __name__ == "__main__":
    main()
