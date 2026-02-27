"""
Web Viewer for Detection Results
Displays the processed video with detections in a web interface
Supports video upload and real-time ambulance detection
"""
from flask import Flask, render_template, send_file, jsonify, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from pathlib import Path
import json
from datetime import datetime
import cv2
from ultralytics import YOLO
import threading
import time

app = Flask(__name__)
app.config['RESULTS_FOLDER'] = 'static/results'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024  # 1GB max file size
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv'}

# Create necessary folders
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['RESULTS_FOLDER'], 'videos'), exist_ok=True)

# Load model
MODEL_PATH = Path("runs/detect/ambulance_detector/weights/best.pt")
CONFIDENCE_THRESHOLD = 0.60
model = None

# Processing status
processing_status = {
    'status': 'idle',
    'current_file': '',
    'progress': 0,
    'total_frames': 0,
    'detections': 0
}

def load_model():
    global model
    if model is None:
        if MODEL_PATH.exists():
            model = YOLO(str(MODEL_PATH))
        else:
            print(f"❌ Model not found at {MODEL_PATH}")
    return model

load_model()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_video_detection(input_path, output_path):
    """Process video and detect ambulances"""
    global processing_status
    
    try:
        processing_status['status'] = 'processing'
        processing_status['current_file'] = os.path.basename(input_path)
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            processing_status['status'] = 'error'
            processing_status['message'] = 'Could not open video file'
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        processing_status['total_frames'] = total_frames
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_detections = 0
        max_confidence = 0
        detection_frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
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
            processing_status['progress'] = int((frame_count / total_frames) * 100)
            processing_status['detections'] = total_detections
            
            # Get max confidence
            if len(detections.boxes) > 0:
                confidences = detections.boxes.conf
                frame_max_conf = max(confidences).item()
                if frame_max_conf > max_confidence:
                    max_confidence = frame_max_conf
            
            # Add statistics to frame
            overlay = annotated_frame.copy()
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
            
            if frame_detections > 0:
                cv2.putText(
                    annotated_frame,
                    f"🚑 AMBULANCE: {frame_detections}",
                    (15, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    3
                )
                cv2.putText(
                    annotated_frame,
                    f"Total: {total_detections}",
                    (15, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )
                
                for i, box in enumerate(detections.boxes):
                    conf = box.conf[0].item()
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    color = (0, 255, 0) if conf > 0.80 else (0, 165, 255)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                    
                    label = f"Ambulance {i+1}"
                    conf_text = f"Conf: {conf:.1%}"
                    
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1 - 60),
                        (x1 + text_size[0] + 10, y1 - 10),
                        (0, 0, 0),
                        -1
                    )
                    
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
                cv2.putText(
                    annotated_frame,
                    "No Ambulances Detected",
                    (15, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 165, 255),
                    2
                )
            
            writer.write(annotated_frame)
        
        cap.release()
        writer.release()
        
        processing_status['status'] = 'completed'
        processing_status['progress'] = 100
        processing_status['result'] = {
            'total_frames': frame_count,
            'total_detections': total_detections,
            'detection_rate': f"{(len(detection_frames)/frame_count*100):.1f}%" if frame_count > 0 else "0%",
            'max_confidence': f"{max_confidence:.2%}",
            'output_file': output_path
        }
        
        return True
        
    except Exception as e:
        processing_status['status'] = 'error'
        processing_status['message'] = str(e)
        return False

def get_video_info():
    """Get information about available videos"""
    videos = []
    
    # Check results folder for processed videos
    results_folder = os.path.join(app.config['RESULTS_FOLDER'], 'videos')
    if os.path.exists(results_folder):
        for file in os.listdir(results_folder):
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                file_path = os.path.join(results_folder, file)
                file_size = os.path.getsize(file_path) / (1024 * 1024)
                videos.append({
                    'name': file,
                    'file': f'/results/{file}',
                    'size': f"{file_size:.2f} MB",
                    'type': 'processed',
                    'full_path': file_path
                })
    
    # Check for result video in root
    if os.path.exists('result_with_detections.mp4'):
        file_size = os.path.getsize('result_with_detections.mp4') / (1024 * 1024)
        videos.insert(0, {
            'name': 'result_with_detections.mp4',
            'file': '/video/result_with_detections.mp4',
            'size': f"{file_size:.2f} MB",
            'type': 'processed',
            'full_path': 'result_with_detections.mp4'
        })
    
    return videos

@app.route('/')
def index():
    """Main page"""
    videos = get_video_info()
    return render_template('view_results.html', videos=videos)

@app.route('/api/videos')
def api_videos():
    """API endpoint for videos"""
    videos = get_video_info()
    return jsonify(videos)

@app.route('/api/status')
def api_status():
    """Get processing status"""
    return jsonify(processing_status)

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload and process"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: ' + ', '.join(app.config['ALLOWED_EXTENSIONS'])}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        
        # Generate output path
        output_filename = f"detected_{timestamp}_{Path(file.filename).stem}.mp4"
        output_path = os.path.join(app.config['RESULTS_FOLDER'], 'videos', output_filename)
        
        # Process video in background
        thread = threading.Thread(
            target=process_video_detection,
            args=(upload_path, output_path)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Video upload successful. Processing started...',
            'filename': filename,
            'output_filename': output_filename
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/video/<path:filename>')
def serve_video(filename):
    """Serve video file from root directory"""
    if '..' in filename or filename.startswith('/'):
        return "Invalid file path", 400
    
    if os.path.exists(filename):
        return send_file(filename, mimetype='video/mp4')
    return "File not found", 404

@app.route('/results/<path:filename>')
def serve_result(filename):
    """Serve result files"""
    if '..' in filename or filename.startswith('/'):
        return "Invalid file path", 400
    
    file_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='video/mp4')
    return "File not found", 404

@app.route('/api/refresh')
def refresh():
    """Refresh video list"""
    videos = get_video_info()
    return jsonify(videos)

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎥 Detection Results Viewer")
    print("="*60)
    print("\n✅ Starting web server...")
    print("🌐 Open your browser: http://127.0.0.1:5001")
    print("\nAvailable Videos:")
    
    videos = get_video_info()
    for i, video in enumerate(videos, 1):
        status = "✅ PROCESSED" if video['type'] == 'processed' else "⏳ UNPROCESSED"
        print(f"  {i}. {video['name']} ({video['size']}) {status}")
    
    print("\n" + "="*60 + "\n")
    
    app.run(debug=False, port=5001, host='127.0.0.1')
