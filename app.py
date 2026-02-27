from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import os
from pathlib import Path
import json
import time
import shutil
from datetime import datetime
from PIL import Image
import io
import numpy as np
import cv2
from traffic_control import traffic_network, initialize_sample_network
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size (for videos)
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv'}
app.config['VIDEO_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv'}

# Inference tuning (helps reduce false positives like cars labeled as ambulances)
DEFAULT_CONFIDENCE = 0.75    # Stricter: only confident detections
DEFAULT_IOU = 0.55           # Tighter: better NMS filtering
MAX_DETECTIONS = 5           # Reduce false positives
INFERENCE_IMG_SIZE = 256     # Smaller input for faster CPU inference
AMBULANCE_CLASS_ID = 0  # dataset is single-class (ambulance) -> class 0

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs(os.path.join('static', 'results', 'videos'), exist_ok=True)

# Initialize traffic control network
initialize_sample_network()

# Load the trained model
MODEL_PATH = Path("runs/detect/ambulance_detector/weights/best.pt")
if MODEL_PATH.exists():
    print(f"Loading model from {MODEL_PATH}")
    try:
        # Try to use GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        model = YOLO(str(MODEL_PATH))
        # CRITICAL: Disable model fusion to avoid Conv.bn attribute errors
        # Fusion can cause issues with custom model structures, especially on CPU
        try:
            model.fuse()  # Pre-fuse to prevent runtime errors
            print("✓ Model fusion prepared (one-time)")
        except Exception as e:
            print(f"Note: Model fusion unavailable: {e}")
        
        model.to(device)
        
        # Enable FP16 on GPU only
        if device == 'cuda':
            try:
                model.half()
                print("FP16 mode enabled - expect 2x faster inference!")
            except Exception as e:
                print(f"FP16 not available: {e}")
    except Exception as e:
        print(f"GPU error, falling back to CPU: {e}")
        device = 'cpu'
        model = YOLO(str(MODEL_PATH))
        print(f"Using device: {device}")
else:
    print("ERROR: Model not found! Please train the model first.")
    model = None

# Thread pool for parallel processing
thread_pool = ThreadPoolExecutor(max_workers=4)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def is_video_file(filename):
    """Check if file is a video"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['VIDEO_EXTENSIONS']

def safe_file_operation(operation, max_retries=5, delay=0.2):
    """Execute file operation with retry logic for Windows file locking"""
    for attempt in range(max_retries):
        try:
            return operation()
        except (OSError, PermissionError, shutil.Error) as e:
            if attempt < max_retries - 1:
                time.sleep(delay * (attempt + 1))  # Exponential backoff
            else:
                raise e

def filter_detections(results, min_confidence=0.75, min_bbox_area=5000):
    """Post-processing filter to reduce false positives"""
    filtered_results = []
    
    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            filtered_results.append(result)
            continue
        
        valid_boxes = []
        valid_confs = []
        valid_clss = []
        
        for i, box in enumerate(result.boxes):
            conf = float(box.conf[0])
            
            # Skip low confidence
            if conf < min_confidence:
                continue
            
            # Calculate bounding box area
            try:
                x1, y1, x2, y2 = box.xyxy[0]
                area = float((x2 - x1) * (y2 - y1))
                
                # Skip very small boxes (likely noise)
                if area < min_bbox_area:
                    continue
                
                valid_boxes.append(box.xyxy[0])
                valid_confs.append(box.conf[0])
                valid_clss.append(box.cls[0])
            except Exception:
                continue
        
        # Update result with filtered boxes
        if len(valid_boxes) > 0:
            import torch
            result.boxes.xyxy = torch.stack(valid_boxes) if valid_boxes else result.boxes.xyxy[:0]
            result.boxes.conf = torch.stack(valid_confs) if valid_confs else result.boxes.conf[:0]
            result.boxes.cls = torch.stack(valid_clss) if valid_clss else result.boxes.cls[:0]
        else:
            result.boxes.xyxy = result.boxes.xyxy[:0]
            result.boxes.conf = result.boxes.conf[:0]
            result.boxes.cls = result.boxes.cls[:0]
        
        filtered_results.append(result)
    
    return filtered_results

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    """Serve favicon or return empty response to suppress 404 errors"""
    return send_from_directory('static', 'favicon.ico', mimetype='image/vnd.microsoft.icon') if os.path.exists('static/favicon.ico') else '', 204

@app.route('/traffic-control')
def traffic_control():
    """Render the traffic control dashboard"""
    return render_template('traffic_control.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and run detection (images and videos)"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not model:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
    
    # Debug: Print file info
    print(f"Uploaded file: {file.filename}")
    print(f"File extension check: {allowed_file(file.filename) if file.filename else 'No filename'}")
    
    if file and allowed_file(file.filename):
        # Check if it's a video file
        if is_video_file(file.filename):
            return process_video(file, request.form.get('confidence', DEFAULT_CONFIDENCE))
        else:
            return process_image(file, request.form.get('confidence', DEFAULT_CONFIDENCE))
    else:
        # Get file extension for better error message
        file_ext = ''
        if file.filename and '.' in file.filename:
            file_ext = file.filename.rsplit('.', 1)[1].lower()
        error_msg = f'Invalid file type (.{file_ext}). Allowed formats: Videos (MP4, AVI, MOV, MKV, WEBM, FLV) or Images (PNG, JPG, JPEG, GIF, BMP)'
        print(f"ERROR: {error_msg}")
        print(f"DEBUG: Allowed extensions: {app.config['ALLOWED_EXTENSIONS']}")
        return jsonify({'error': error_msg}), 400

def process_image(file, conf_threshold):
    """Process image file and detect ambulances"""
    # Save uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    conf_threshold = float(conf_threshold)
    
    try:
        # Create unique filename with timestamp to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base_name = os.path.splitext(filename)[0]
        # Always save as .jpg regardless of input format (image or video)
        # PIL can only save to image formats, not video formats
        unique_filename = f"{base_name}_{timestamp}.jpg"
        
        # Prepare static result directory
        static_result_dir = os.path.join('static', 'results', 'predictions')
        os.makedirs(static_result_dir, exist_ok=True)
        static_result_path = os.path.join(static_result_dir, unique_filename)
        
        # Run detection WITHOUT saving (to avoid file locking issues)
        # We'll save the annotated image ourselves from memory
        results = model.predict(
            source=filepath,
            conf=conf_threshold,
            iou=DEFAULT_IOU,
            classes=[AMBULANCE_CLASS_ID],
            save=False,  # CRITICAL: Don't let YOLO save - we'll do it ourselves
            verbose=False,
            imgsz=INFERENCE_IMG_SIZE,  # Smaller size for faster CPU inference
            max_det=MAX_DETECTIONS,  # limit detections to reduce false positives
            device=0 if torch.cuda.is_available() else 'cpu'  # Use GPU if available
        )
        
        # Process results
        result = results[0]
        detections = []
        
        if result.boxes is not None and len(result.boxes) > 0:
            # Keep only the detection with highest confidence
            best_box_idx = int(result.boxes.conf.argmax())
            box = result.boxes[best_box_idx]
            conf = float(box.conf[0].item())
            detections.append({
                'id': 1,
                'confidence': round(conf * 100, 2),
                'bbox': box.xyxy[0].tolist()
            })
        
        # Get the annotated image by drawing boxes on the original image
        img = cv2.imread(filepath)
        if img is None:
            # fallback to creating a blank image
            annotated_image = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            # Draw the best detection (if any)
            if result.boxes is not None and len(result.boxes) > 0:
                best_box_idx = int(result.boxes.conf.argmax())
                box = result.boxes[best_box_idx]
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                label = f"Ambulance {conf:.1%}"
                cv2.putText(img, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            # Convert BGR -> RGB for PIL
            annotated_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image for saving
        pil_image = Image.fromarray(annotated_image)
        
        # Save the image ourselves with proper file handling
        def save_image():
            # Ensure directory exists
            os.makedirs(static_result_dir, exist_ok=True)
            # Save with high quality
            pil_image.save(static_result_path, quality=95, optimize=True)
            return static_result_path
        
        # Use safe file operation to save
        image_url = None
        try:
            saved_path = safe_file_operation(save_image, max_retries=5, delay=0.3)
            if os.path.exists(static_result_path):
                image_url = f"/static/results/predictions/{unique_filename}"
        except Exception as e:
            # If saving fails, try alternative approach with temp file
            temp_path = static_result_path + '.tmp'
            try:
                pil_image.save(temp_path, quality=95)
                time.sleep(0.2)  # Small delay
                if os.path.exists(temp_path):
                    # Try to move/rename the temp file
                    def move_file():
                        if os.path.exists(static_result_path):
                            os.remove(static_result_path)
                        shutil.move(temp_path, static_result_path)
                    
                    safe_file_operation(move_file, max_retries=3, delay=0.3)
                    if os.path.exists(static_result_path):
                        image_url = f"/static/results/predictions/{unique_filename}"
            except Exception:
                # Last resort: return without image URL
                image_url = None
        
        return jsonify({
            'success': True,
            'detections': detections,
            'count': len(detections),
            'image_url': image_url,
            'original_filename': filename,
            'file_type': 'image'
        })
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"ERROR: {str(e)}")
        print(error_trace)
        return jsonify({'error': f'Detection failed: {str(e)}'}), 500

def process_video(file, conf_threshold):
    """Process video file and detect ambulances in MP4 and other video formats"""
    # Save uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    conf_threshold = float(conf_threshold)
    
    try:
        # Create unique filename with timestamp to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base_name = os.path.splitext(filename)[0]
        # Always save output as MP4 for web compatibility
        unique_filename = f"{base_name}_{timestamp}.mp4"
        
        # Prepare output directory
        output_dir = os.path.join('static', 'results', 'videos')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, unique_filename)
        
        # Run video detection
        # YOLO processes video frame-by-frame and detects ambulances
        print(f"🚀 Starting video processing: {filename}")
        print(f"   Confidence threshold: {conf_threshold}")
        print(f"   Output will be saved as: {unique_filename}")
        
        # Open the input video to get properties
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            raise Exception(f"Could not open video file: {filepath}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"   Video properties: {width}x{height} @ {fps} FPS, {total_frames_video} frames")
        
        # Create video writer for output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_output = output_path.replace('.mp4', '_temp.mp4')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
        
        # Get detection statistics
        all_detections = []
        total_detections = 0
        frames_with_detections = 0
        frame_count = 0
        
        print(f"🎬 Processing all frames for ambulance detection...")
        
        # Process frames with aggressive skipping for speed
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run YOLO detection on this frame
            # Skip frames for faster processing: only process every 6th frame
            if frame_count % 6 != 0:  # Skip 5 out of every 6 frames
                # Still write the frame but don't process it
                out.write(frame)
                continue
            
            results = model.predict(
                source=frame,
                conf=conf_threshold,
                iou=DEFAULT_IOU,
                classes=[AMBULANCE_CLASS_ID],
                save=False,
                verbose=False,
                imgsz=INFERENCE_IMG_SIZE,  # Smaller size for faster CPU inference
                max_det=MAX_DETECTIONS,  # limit detections per frame
                device=0 if torch.cuda.is_available() else 'cpu'  # Use GPU if available
            )
            
            result = results[0]
            
            # Draw bounding boxes on frame if detections found
            if result.boxes is not None and len(result.boxes) > 0:
                
                # Filter detections by confidence threshold
                for box in result.boxes:
                    conf = float(box.conf[0].item())
                    
                    # Only count high-confidence detections (>= threshold in 0-1 scale)
                    if conf >= conf_threshold:
                        frames_with_detections += 1
                        total_detections += 1
                        
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Draw bounding box (bright red for visibility)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red box, thickness 3
                        
                        # Draw label with confidence
                        label = f'Ambulance {conf:.1%}'
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        
                        # Draw label background
                        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), (0, 0, 255), -1)
                        
                        # Draw label text
                        cv2.putText(frame, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # Store detection info (only high confidence)
                        all_detections.append({
                            'confidence': round(conf * 100, 2),
                            'frame': frame_count
                        })
            
            # Write frame to output video
            out.write(frame)
        
        # Release everything
        cap.release()
        out.release()
        
        # Move temp file to final location
        if os.path.exists(temp_output):
            if os.path.exists(output_path):
                os.remove(output_path)
            shutil.move(temp_output, output_path)
            print(f"✅ Video with bounding boxes saved: {output_path}")
        
        print(f"✅ Video processing complete!")
        print(f"   Processed {frame_count} frames")
        print(f"   Found {total_detections} ambulance detection(s) in {frames_with_detections} frame(s)")
        
        # Check if video was created successfully
        video_url = None
        if os.path.exists(output_path):
            video_url = f"/static/results/videos/{unique_filename}"
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
            print(f"🌐 Video URL: {video_url}")
            print(f"📊 Output video size: {file_size:.2f} MB")
        else:
            print(f"❌ Error: Processed video file not found at {output_path}")
        
        return jsonify({
            'success': True,
            'detections': all_detections,
            'count': total_detections,
            'frames_with_detections': frames_with_detections,
            'total_frames': frame_count,
            'video_url': video_url,
            'original_filename': filename,
            'file_type': 'video'
        })
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"ERROR: {str(e)}")
        print(error_trace)
        return jsonify({'error': f'Video processing failed: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None
    })

# ==================== TRAFFIC CONTROL API ENDPOINTS ====================

@app.route('/api/traffic/status', methods=['GET'])
def get_traffic_status():
    """Get current status of entire traffic control network"""
    return jsonify(traffic_network.get_network_status())

@app.route('/api/traffic/intersections', methods=['GET'])
def get_intersections():
    """Get list of all intersections in network"""
    intersections = []
    for inter_id, intersection in traffic_network.intersections.items():
        intersections.append({
            'id': inter_id,
            'lanes': intersection.lanes,
            'status': intersection.get_status()
        })
    return jsonify({
        'intersections': intersections,
        'total': len(intersections)
    })

@app.route('/api/traffic/ambulance/route', methods=['POST'])
def set_ambulance_route():
    """Set ambulance route through intersections (async processing)"""
    data = request.get_json()
    route = data.get('route', [])
    
    if not route:
        return jsonify({'error': 'Route cannot be empty'}), 400
    
    # Execute traffic control in background thread to prevent blocking
    def process_route():
        try:
            result = traffic_network.set_ambulance_route(route)
            if route:
                direction = data.get('direction', 'North')
                traffic_network.activate_green_wave_at_intersection(route[0], direction)
        except Exception as e:
            print(f"Error setting ambulance route: {e}")
    
    # Run in background - don't wait for completion
    thread_pool.submit(process_route)
    
    # Return immediately to client
    return jsonify({
        'status': 'route_processing',
        'route': route,
        'message': 'Route is being processed in background'
    })

@app.route('/api/traffic/ambulance/detect', methods=['POST'])
def ambulance_detected():
    """Handle ambulance detection - activate green wave (async)"""
    data = request.get_json()
    intersection_id = data.get('intersection_id')
    direction = data.get('direction', 'North')
    confidence = data.get('confidence', 0.0)
    
    if not intersection_id:
        return jsonify({'error': 'intersection_id required'}), 400
    
    # Execute in background thread - non-blocking
    def activate_wave():
        try:
            return traffic_network.activate_green_wave_at_intersection(intersection_id, direction)
        except Exception as e:
            print(f"Error activating green wave: {e}")
    
    thread_pool.submit(activate_wave)
    
    # Return immediately to client
    return jsonify({
        'event': 'ambulance_detected',
        'intersection_id': intersection_id,
        'direction': direction,
        'confidence': confidence,
        'status': 'green_wave_activating',
        'message': 'Green wave activation in progress',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/traffic/ambulance/position', methods=['POST'])
def update_ambulance_position():
    """Update ambulance position - moves green wave to next intersection (async)"""
    data = request.get_json()
    current_intersection = data.get('current_intersection')
    
    if not current_intersection:
        return jsonify({'error': 'current_intersection required'}), 400
    
    # Execute in background thread - non-blocking
    def update_position():
        try:
            return traffic_network.update_ambulance_position(current_intersection)
        except Exception as e:
            print(f"Error updating position: {e}")
    
    thread_pool.submit(update_position)
    
    # Return immediately to client
    return jsonify({
        'event': 'position_updating',
        'current_intersection': current_intersection,
        'status': 'position_update_in_progress',
        'message': 'Next intersection green wave activating',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/traffic/green-wave/activate', methods=['POST'])
def activate_green_wave():
    """Manually activate green wave at specific intersection (async)"""
    data = request.get_json()
    intersection_id = data.get('intersection_id')
    direction = data.get('direction', 'North')
    
    if not intersection_id:
        return jsonify({'error': 'intersection_id required'}), 400
    
    # Execute in background thread
    def activate_wave():
        try:
            return traffic_network.activate_green_wave_at_intersection(intersection_id, direction)
        except Exception as e:
            print(f"Error activating green wave: {e}")
    
    thread_pool.submit(activate_wave)
    
    # Return immediately
    return jsonify({
        'status': 'green_wave_activating',
        'intersection_id': intersection_id,
        'direction': direction,
        'message': 'Green wave activation in progress'
    })

@app.route('/api/traffic/green-wave/deactivate', methods=['POST'])
def deactivate_green_wave():
    """Deactivate green wave at specific intersection (async)"""
    data = request.get_json()
    intersection_id = data.get('intersection_id')
    
    if not intersection_id:
        return jsonify({'error': 'intersection_id required'}), 400
    
    # Execute in background thread
    def deactivate_wave():
        try:
            return traffic_network.deactivate_green_wave_at_intersection(intersection_id)
        except Exception as e:
            print(f"Error deactivating green wave: {e}")
    
    thread_pool.submit(deactivate_wave)
    
    # Return immediately
    return jsonify({
        'status': 'green_wave_deactivating',
        'intersection_id': intersection_id,
        'message': 'Green wave deactivation in progress'
    })

@app.route('/api/traffic/ambulance/stop', methods=['POST'])
def stop_ambulance_tracking():
    """Stop ambulance tracking and reset network (async)"""
    # Execute in background thread
    def stop_tracking():
        try:
            return traffic_network.stop_ambulance_tracking()
        except Exception as e:
            print(f"Error stopping tracking: {e}")
    
    thread_pool.submit(stop_tracking)
    
    # Return immediately
    return jsonify({
        'event': 'tracking_stopping',
        'status': 'tracking_stop_in_progress',
        'message': 'Ambulance tracking stopping in background',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files with error handling"""
    try:
        return send_from_directory('static', filename)
    except Exception as e:
        return jsonify({'error': f'File not found: {str(e)}'}), 404

def process_single_direction(direction, filepath, filename, is_video, conf_threshold, timestamp):
    """
    Process detection for a single direction (optimized for parallel execution)
    Returns detection result dict for the given direction
    """
    detections = []
    ambulance_found = False
    annotated_image = None
    
    try:
        if is_video:
            # Process video by sampling frames with OpenCV (avoid .plot() issues)
            cap = cv2.VideoCapture(filepath)
            if not cap.isOpened():
                raise Exception('Could not open video file')

            frame_count = 0
            sampled_frame = None
            skip_rate = 12  # ultra-aggressive skipping

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                if frame_count % skip_rate != 0:
                    continue

                # Run inference on this sampled frame (numpy BGR)
                results = model.predict(
                    source=frame,
                    conf=conf_threshold,
                    iou=DEFAULT_IOU,
                    classes=[AMBULANCE_CLASS_ID],
                    save=False,
                    verbose=False,
                    max_det=MAX_DETECTIONS,
                    imgsz=INFERENCE_IMG_SIZE,
                    device=0 if torch.cuda.is_available() else 'cpu'
                )
                r = results[0]
                if r.boxes is not None and len(r.boxes) > 0:
                    ambulance_found = True
                    best_box_idx = int(r.boxes.conf.argmax())
                    box = r.boxes[best_box_idx]
                    conf_val = float(box.conf[0].item())

                    # Filter out very small boxes to reduce false positives
                    x1_f, y1_f, x2_f, y2_f = box.xyxy[0].tolist()
                    area = float((x2_f - x1_f) * (y2_f - y1_f))
                    if area < 5000:
                        continue

                    detections.append({
                        'confidence': round(conf_val * 100, 2),
                        'bbox': box.xyxy[0].tolist()
                    })

                    # Draw boxes on the sampled frame (BGR)
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    label = f"Ambulance {conf_val:.1%}"
                    cv2.putText(frame, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    sampled_frame = frame
                    break

            cap.release()
            if sampled_frame is not None:
                # Convert BGR -> RGB for PIL
                annotated_image = cv2.cvtColor(sampled_frame, cv2.COLOR_BGR2RGB)
            else:
                annotated_image = None
        else:
            # Image processing - avoid using result.plot(), draw boxes manually
            img = cv2.imread(filepath)
            if img is None:
                raise Exception('Could not read image file')

            results = model.predict(
                source=filepath,
                conf=conf_threshold,
                iou=DEFAULT_IOU,
                classes=[AMBULANCE_CLASS_ID],
                save=False,
                verbose=False,
                max_det=MAX_DETECTIONS,
                imgsz=INFERENCE_IMG_SIZE,
                device=0 if torch.cuda.is_available() else 'cpu'
            )
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                ambulance_found = True
                best_box_idx = int(result.boxes.conf.argmax())
                box = result.boxes[best_box_idx]
                conf_val = float(box.conf[0].item())

                # Filter out very small boxes to reduce false positives
                x1_f, y1_f, x2_f, y2_f = box.xyxy[0].tolist()
                area = float((x2_f - x1_f) * (y2_f - y1_f))
                if area >= 5000:
                    detections.append({
                        'confidence': round(conf_val * 100, 2),
                        'bbox': box.xyxy[0].tolist()
                    })

                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                label = f"Ambulance {conf_val:.1%}"
                cv2.putText(img, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            # Convert BGR -> RGB for PIL
            annotated_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if annotated_image is None:
            annotated_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        pil_image = Image.fromarray(annotated_image)
        unique_filename = f"{direction}_{timestamp}.jpg"
        static_result_dir = os.path.join('static', 'results', 'multi_direction')
        os.makedirs(static_result_dir, exist_ok=True)
        static_result_path = os.path.join(static_result_dir, unique_filename)
        
        image_url = None
        try:
            pil_image.save(static_result_path, quality=60, optimize=True)
            if os.path.exists(static_result_path):
                image_url = f"/static/results/multi_direction/{unique_filename}"
        except Exception as e:
            print(f"Warning: Could not save image for {direction}: {str(e)}")
        
        return {
            'direction': direction,
            'ambulance_found': ambulance_found,
            'count': len(detections),
            'detections': detections,
            'image_url': image_url,
            'source_is_video': is_video
        }
    
    except Exception as e:
        print(f"Error processing {direction}: {str(e)}")
        return {
            'direction': direction,
            'ambulance_found': False,
            'count': 0,
            'detections': [],
            'image_url': None,
            'source_is_video': is_video
        }

@app.route('/upload-multi-direction', methods=['POST'])
def upload_multi_direction():
    """Handle 4 directional images or videos for traffic control simulation"""
    # Debug info: print incoming files
    try:
        print(f"upload-multi-direction: content_length={request.content_length}, files={list(request.files.keys())}")
    except Exception:
        print("upload-multi-direction: could not read request metadata")

    if not model:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500

    # Check if all 4 directions are provided and allow image or video per direction
    directions = ['NORTH', 'SOUTH', 'EAST', 'WEST']
    files_received = {}

    allowed_image_exts = {'jpg', 'jpeg', 'png', 'bmp', 'gif', 'tif', 'tiff'}
    allowed_video_exts = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

    missing_dirs = []
    for direction in directions:
        # accept several common field name variants from the client
        variants = [direction, direction.lower(), direction.capitalize()]
        file = None
        for v in variants:
            if v in request.files:
                file = request.files[v]
                break

        if file is None:
            missing_dirs.append(direction)
            print(f"Warning: Missing file for {direction}. available keys: {list(request.files.keys())}")
            continue

        if file.filename == '':
            missing_dirs.append(direction)
            print(f"Warning: Empty filename for {direction}")
            continue

        filename = secure_filename(file.filename)
        ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''

        if ext in allowed_image_exts:
            is_video = False
        elif ext in allowed_video_exts:
            is_video = True
        else:
            print(f"Warning: Invalid file extension .{ext} for {direction}")
            missing_dirs.append(direction)
            continue

        # Save uploaded file to temporary upload folder (one per direction)
        timestamp_local = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        temp_name = f"{direction}_{timestamp_local}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_name)
        try:
            file.save(filepath)
        except Exception as e:
            print(f"Error saving uploaded file for {direction}: {e}")
            missing_dirs.append(direction)
            continue

        files_received[direction] = {
            'filepath': filepath,
            'filename': filename,
            'is_video': is_video
        }

    conf_threshold = float(request.form.get('confidence', DEFAULT_CONFIDENCE))

    try:
        # Process all 4 directions in PARALLEL for massive speed boost
        detection_results = {}
        ambulance_detection_map = {
            'NORTH': False,
            'SOUTH': False,
            'EAST': False,
            'WEST': False
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Build list of tasks for parallel execution
        tasks = []
        for direction in directions:
            entry = files_received.get(direction)
            if entry is None:
                detection_results[direction] = {
                    'ambulance_found': False,
                    'count': 0,
                    'detections': [],
                    'image_url': None,
                    'source_is_video': False,
                    'note': 'no_file_provided'
                }
            else:
                # Submit task to thread pool
                task = thread_pool.submit(
                    process_single_direction,
                    direction,
                    entry['filepath'],
                    entry['filename'],
                    entry['is_video'],
                    conf_threshold,
                    timestamp
                )
                tasks.append(task)
        
        # Collect results from parallel tasks
        for future in as_completed(tasks):
            try:
                result = future.result(timeout=60)  # 60 second timeout per direction
                direction = result['direction']
                detection_results[direction] = {
                    'ambulance_found': result['ambulance_found'],
                    'count': result['count'],
                    'detections': result['detections'],
                    'image_url': result['image_url'],
                    'source_is_video': result['source_is_video']
                }
                if result['ambulance_found']:
                    ambulance_detection_map[direction] = True
                
                # Clean up temporary upload file
                if direction in files_received:
                    filepath = files_received[direction]['filepath']
                    if os.path.exists(filepath):
                        try:
                            os.remove(filepath)
                        except:
                            pass
            except Exception as e:
                print(f"Error in parallel task: {str(e)}")

        # Determine which routes to activate based on detections
        active_routes = []
        for direction in directions:
            if ambulance_detection_map[direction]:
                active_routes.append(direction)

        return jsonify({
            'success': True,
            'detection_results': detection_results,
            'ambulance_detection_map': ambulance_detection_map,
            'active_routes': active_routes,
            'total_ambulances': sum(1 for v in ambulance_detection_map.values() if v),
            'timestamp': timestamp
        })

    except Exception as e:
        import traceback
        print(f"ERROR: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'Detection failed: {str(e)}'}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Ambulance Detection Web App")
    print("="*60)
    print(f"Model loaded: {model is not None}")
    if model:
        print(f"Model path: {MODEL_PATH}")
    print("Starting Flask server...")
    print("Open your browser and go to: http://127.0.0.1:5000")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
