from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import cv2
import pickle
import os
import numpy as np
from datetime import datetime
from deepface import DeepFace

app = Flask(__name__)
CORS(app)

class CameraDetectionSystem:
    def __init__(self):
        self.camera = None
        self.is_running = False
        self.known_faces_db = []
        self.database_file = 'face_database.pkl'
        self.model_name = 'VGG-Face'
        self.detection_threshold = 0.3  # Increased for more matches
        
    def load_database(self):
        if os.path.exists(self.database_file):
            with open(self.database_file, 'rb') as f:
                self.known_faces_db = pickle.load(f)
            return True
        return False
    
    def cosine_distance(self, embedding1, embedding2):
        embedding1 = np.array(embedding1)
        embedding2 = np.array(embedding2)
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 1.0
        similarity = dot_product / (norm1 * norm2)
        return 1 - similarity
    
    def find_best_match(self, embedding):
        if not self.known_faces_db:
            return None
        min_distance = float('inf')
        best_match = None
        for criminal in self.known_faces_db:
            distance = self.cosine_distance(embedding, criminal['embedding'])
            if distance < min_distance and distance < self.detection_threshold:
                min_distance = distance
                best_match = {
                    'name': criminal['name'],
                    'confidence': (1 - distance) * 100
                }
        return best_match
    
    def detect_and_recognize(self, frame):
        try:
            # More sensitive face detection
            face_objs = DeepFace.extract_faces(
                img_path=frame,
                detector_backend='opencv',
                enforce_detection=False,
                align=False  # Faster processing
            )
            
            for face_obj in face_objs:
                # Lower confidence threshold for better detection
                if face_obj['confidence'] > 0.5:
                    facial_area = face_obj['facial_area']
                    x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                    
                    # Add padding for better recognition
                    padding = 10
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(frame.shape[1] - x, w + 2*padding)
                    h = min(frame.shape[0] - y, h + 2*padding)
                    
                    face_img = frame[y:y+h, x:x+w]
                    
                    if face_img.size > 0 and w > 30 and h > 30:  # Minimum face size
                        try:
                            embedding = DeepFace.represent(
                                img_path=face_img,
                                model_name=self.model_name,
                                enforce_detection=False,
                                detector_backend='skip'
                            )
                            
                            if embedding:
                                match = self.find_best_match(embedding[0]['embedding'])
                                if match:
                                    # Criminal detected - thick red box
                                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
                                    label = f"{match['name'].split('_')[1]}"  # Clean name
                                    conf = f"{match['confidence']:.1f}%"
                                    cv2.putText(frame, label, (x+5, y-25), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                    cv2.putText(frame, conf, (x+5, y-5), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                else:
                                    # Unknown person - green box
                                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        except:
                            # Still draw box even if recognition fails
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        except:
            pass
        return frame
    
    def start_camera(self, camera_index=0):
        if self.is_running:
            return False
        self.camera = cv2.VideoCapture(camera_index)
        if not self.camera.isOpened():
            return False
        # Better camera settings
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        self.is_running = True
        return True
    
    def stop_camera(self):
        self.is_running = False
        if self.camera:
            self.camera.release()
    
    def generate_frames(self):
        frame_count = 0
        while self.is_running:
            ret, frame = self.camera.read()
            if not ret:
                break
            frame_count += 1
            # Process every 10th frame for faster response
            if frame_count % 10 == 0:
                frame = self.detect_and_recognize(frame)
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

detection_system = CameraDetectionSystem()

@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    if not detection_system.known_faces_db:
        if not detection_system.load_database():
            return jsonify({'error': 'Failed to load database', 'database_size': 0}), 500
    if detection_system.start_camera(0):
        return jsonify({
            'message': 'Camera started',
            'database_size': len(detection_system.known_faces_db)
        })
    return jsonify({'error': 'Failed to start camera'}), 500

@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    detection_system.stop_camera()
    return jsonify({'message': 'Camera stopped'})

@app.route('/api/camera/feed')
def video_feed():
    if not detection_system.is_running:
        return jsonify({'error': 'Camera not running'}), 400
    return Response(detection_system.generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
