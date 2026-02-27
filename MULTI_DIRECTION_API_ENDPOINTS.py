"""
Flask API Endpoint for Multi-Directional Ambulance Detection
Handles 4 images (one per direction: North, South, East, West)
"""

import numpy as np

# Add these new endpoints to app.py

@app.route('/upload-multi-direction', methods=['POST'])
def upload_multi_direction():
    """Handle 4 directional images for traffic control simulation"""
    # Debug info: print incoming files and size
    try:
        print(f"upload-multi-direction: content_length={request.content_length}, files={list(request.files.keys())}")
    except Exception:
        print("upload-multi-direction: could not read request metadata")

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
    
    if not model:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
    
    conf_threshold = float(request.form.get('confidence', DEFAULT_CONFIDENCE))
    
    try:
        # Process all 4 images
        detection_results = {}
        ambulance_detection_map = {
            'NORTH': False,
            'SOUTH': False,
            'EAST': False,
            'WEST': False
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
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
                continue
            filepath = entry['filepath']
            filename = entry['filename']
            is_video = entry['is_video']

            detections = []
            ambulance_found = False
            annotated_image = None

            if is_video:
                # Stream video frames until an ambulance is found (or exhaust)
                last_result = None
                stream = model.predict(
                    source=filepath,
                    conf=conf_threshold,
                    iou=DEFAULT_IOU,
                    classes=[AMBULANCE_CLASS_ID],
                    stream=True,
                    verbose=False,
                    max_det=MAX_DETECTIONS
                )
                try:
                    for r in stream:
                        last_result = r
                        if r.boxes is not None and len(r.boxes) > 0:
                            ambulance_found = True
                            ambulance_detection_map[direction] = True
                            best_box_idx = int(r.boxes.conf.argmax())
                            box = r.boxes[best_box_idx]
                            conf_val = float(box.conf[0].item())
                            detections.append({
                                'confidence': round(conf_val * 100, 2),
                                'bbox': box.xyxy[0].tolist()
                            })
                            annotated_image = r.plot()
                            break
                except Exception:
                    # If streaming fails, fall back to single-pass prediction
                    last_result = None

                if annotated_image is None and last_result is not None:
                    annotated_image = last_result.plot()

            else:
                results = model.predict(
                    source=filepath,
                    conf=conf_threshold,
                    iou=DEFAULT_IOU,
                    classes=[AMBULANCE_CLASS_ID],
                    save=False,
                    verbose=False,
                    max_det=MAX_DETECTIONS
                )
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    ambulance_found = True
                    ambulance_detection_map[direction] = True
                    best_box_idx = int(result.boxes.conf.argmax())
                    box = result.boxes[best_box_idx]
                    conf_val = float(box.conf[0].item())
                    detections.append({
                        'confidence': round(conf_val * 100, 2),
                        'bbox': box.xyxy[0].tolist()
                    })
                annotated_image = result.plot()

            if annotated_image is None:
                # fallback: create a blank image informing no output
                annotated_image = np.zeros((480, 640, 3), dtype=np.uint8)

            pil_image = Image.fromarray(annotated_image)

            file_ext = os.path.splitext(filename)[1] or '.jpg'
            unique_filename = f"{direction}_{timestamp}{file_ext}"

            static_result_dir = os.path.join('static', 'results', 'multi_direction')
            os.makedirs(static_result_dir, exist_ok=True)
            static_result_path = os.path.join(static_result_dir, unique_filename)

            def save_image():
                pil_image.save(static_result_path, quality=95, optimize=True)
                return static_result_path

            image_url = None
            try:
                safe_file_operation(save_image, max_retries=5, delay=0.3)
                if os.path.exists(static_result_path):
                    image_url = f"/static/results/multi_direction/{unique_filename}"
            except Exception as e:
                print(f"Warning: Could not save image for {direction}: {str(e)}")

            detection_results[direction] = {
                'ambulance_found': ambulance_found,
                'count': len(detections),
                'detections': detections,
                'image_url': image_url,
                'source_is_video': is_video
            }

            # Clean up temporary upload file
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass
        
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


@app.route('/api/traffic-multi-direction', methods=['GET'])
def get_traffic_multi_direction():
    """Get current traffic status for multi-directional system"""
    try:
        # Get last detection results from session or cache
        return jsonify({
            'status': 'ready',
            'directions': ['NORTH', 'SOUTH', 'EAST', 'WEST'],
            'message': 'Multi-directional traffic control ready'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
