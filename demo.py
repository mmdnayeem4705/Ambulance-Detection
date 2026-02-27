"""
Real-time Ambulance Detection Demo
Plays a video with live object detection and bounding boxes
Demonstrates the trained YOLOv8 model detecting ambulances in traffic scenes
"""
import cv2
import os
from pathlib import Path
from ultralytics import YOLO
import argparse
from datetime import datetime

# Configuration
MODEL_PATH = Path("runs/detect/ambulance_detector/weights/best.pt")
CONFIDENCE_THRESHOLD = 0.60
MAX_DETECTIONS = 10
AMBULANCE_CLASS_ID = 0

def load_model():
    """Load the trained YOLO model"""
    if not MODEL_PATH.exists():
        print(f"❌ ERROR: Model not found at {MODEL_PATH}")
        print("Please train the model first using: python train.py")
        return None
    
    print(f"✅ Loading model from {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    print("✅ Model loaded successfully!")
    return model

def play_video_with_detection(video_path, model, output_path=None, show_stats=True):
    """
    Process video with real-time ambulance detection (Headless mode - no display)
    
    Args:
        video_path: Path to video file
        model: Loaded YOLO model
        output_path: Optional path to save output video
        show_stats: Whether to show detection statistics
    """
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"❌ ERROR: Video file not found: {video_path}")
        return False
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ ERROR: Could not open video: {video_path}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n{'='*60}")
    print(f"📹 Video Properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.2f} seconds")
    print(f"{'='*60}\n")
    
    # Setup video writer if output path provided
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            print(f"⚠️  Warning: Could not open video writer. Trying alternative codec...")
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"💾 Output will be saved to: {output_path}\n")
    
    # Detection statistics
    frame_count = 0
    total_detections = 0
    max_confidence = 0
    detection_frames = []
    
    print(f"🚀 Starting detection (Headless Mode - Processing Video)...")
    print(f"⏳ This may take a few minutes depending on video length...\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("\n✅ Video processing completed!")
            break
        
        # Run inference
        results = model.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )
        
        # Get detections
        detections = results[0]
        annotated_frame = detections.plot()
        
        # Count detections in this frame
        frame_detections = len(detections.boxes)
        if frame_detections > 0:
            total_detections += frame_detections
            detection_frames.append(frame_count + 1)
        
        frame_count += 1
        
        # Get max confidence
        if len(detections.boxes) > 0:
            confidences = detections.boxes.conf
            frame_max_conf = max(confidences).item()
            if frame_max_conf > max_confidence:
                max_confidence = frame_max_conf
        
        # Add statistics to frame
        if show_stats:
            # Create semi-transparent overlay for better text visibility
            overlay = annotated_frame.copy()
            
            # Top-left info box background
            cv2.rectangle(overlay, (5, 5), (400, 120), (0, 0, 0), -1)
            annotated_frame = cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0)
            
            # Frame counter
            cv2.putText(
                annotated_frame,
                f"Frame: {frame_count}/{total_frames}",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
            
            # Detections in this frame
            if frame_detections > 0:
                # Highlight ambulance detection
                cv2.putText(
                    annotated_frame,
                    f"🚑 AMBULANCE DETECTED: {frame_detections}",
                    (15, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    3
                )
                
                # Show detection rate
                cv2.putText(
                    annotated_frame,
                    f"Total Detections: {total_detections}",
                    (15, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )
                
                # Enhanced labeling for each detection
                for i, box in enumerate(detections.boxes):
                    conf = box.conf[0].item()
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Draw thick bounding box
                    color = (0, 255, 0) if conf > 0.80 else (0, 165, 255)  # Green if high confidence, orange otherwise
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                    
                    # Label background
                    label = f"Ambulance {i+1}"
                    conf_text = f"Conf: {conf:.1%}"
                    
                    # Text background for better visibility
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1 - 60),
                        (x1 + text_size[0] + 10, y1 - 10),
                        (0, 0, 0),
                        -1
                    )
                    
                    # Label and confidence
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1 + 5, y1 - 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                    cv2.putText(
                        annotated_frame,
                        conf_text,
                        (x1 + 5, y1 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2
                    )
                    
                    # Dimensions info
                    width_box = x2 - x1
                    height_box = y2 - y1
                    size_text = f"Size: {width_box}x{height_box}px"
                    cv2.putText(
                        annotated_frame,
                        size_text,
                        (x1, y2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )
            else:
                # No detections
                cv2.putText(
                    annotated_frame,
                    "No Ambulances Detected",
                    (15, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 165, 255),
                    2
                )
        
        # Write to output video if saving
        if writer:
            writer.write(annotated_frame)
        
        # Print progress every 30 frames
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"  ⏳ Progress: {frame_count}/{total_frames} frames ({progress:.1f}%) - Detections so far: {total_detections}")
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"📊 Detection Summary:")
    print(f"  Total Frames Processed: {frame_count}")
    print(f"  Frames with Ambulances: {len(detection_frames)}")
    print(f"  Total Ambulances Detected: {total_detections}")
    print(f"  Average per Frame: {total_detections/frame_count:.2f}")
    if max_confidence > 0:
        print(f"  Max Confidence: {max_confidence:.2%}")
    if detection_frames:
        print(f"  Detection Frames: {detection_frames[:10]}" + ("..." if len(detection_frames) > 10 else ""))
    if output_path:
        print(f"\n✅ Output saved to: {output_path}")
    print(f"{'='*60}\n")
    
    return True

def find_sample_video():
    """Find available sample videos in common locations"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm']
    
    search_dirs = [
        'uploads',
        'static/results/videos',
        'detections',
        '.'
    ]
    
    print("🔍 Searching for sample videos...")
    videos = []
    
    for search_dir in search_dirs:
        if os.path.isdir(search_dir):
            for file in os.listdir(search_dir):
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    full_path = os.path.join(search_dir, file)
                    videos.append(full_path)
                    print(f"  Found: {full_path}")
    
    return videos

def main():
    parser = argparse.ArgumentParser(
        description="Real-time Ambulance Detection Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py --video traffic.mp4
  python demo.py --video traffic.mp4 --output result.mp4
  python demo.py --webcam
        """
    )
    
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--output', type=str, help='Path to save output video')
    parser.add_argument('--webcam', action='store_true', help='Use webcam for real-time detection')
    parser.add_argument('--conf', type=float, default=0.60, help='Confidence threshold (0-1)')
    
    args = parser.parse_args()
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Update confidence threshold
    global CONFIDENCE_THRESHOLD
    CONFIDENCE_THRESHOLD = args.conf
    
    # Webcam mode
    if args.webcam:
        print("\n⚠️  Note: GUI display not available in this environment")
        print("📹 Webcam detection requires real-time display")
        print("💡 Tip: Use --video mode to process videos and save results\n")
        print("If running on a system with display support, use:")
        print("  python demo.py --webcam\n")
        return
    
    # Video file mode
    if args.video:
        video_path = args.video
    else:
        # Try to find a sample video
        videos = find_sample_video()
        
        if not videos:
            print("\n❌ No video file specified and no sample videos found")
            print("\nUsage:")
            print("  python demo.py --video <path_to_video>")
            print("  python demo.py --webcam")
            return
        
        if len(videos) == 1:
            video_path = videos[0]
            print(f"\n✅ Using video: {video_path}\n")
        else:
            print("\nMultiple videos found. Please choose one:")
            for i, v in enumerate(videos, 1):
                print(f"  {i}. {v}")
            choice = input(f"\nEnter number (1-{len(videos)}): ").strip()
            try:
                video_path = videos[int(choice) - 1]
            except (ValueError, IndexError):
                print("❌ Invalid choice")
                return
    
    # Play video with detection
    play_video_with_detection(video_path, model, args.output)

if __name__ == "__main__":
    main()
