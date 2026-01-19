from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from deepface import DeepFace
import cv2
import os
import pickle
import numpy as np
from datetime import datetime
import threading

app = Flask(__name__)
CORS(app)

class CriminalFaceDetector:
    def __init__(self, known_faces_dir='scraped_data/images'):
        self.known_faces_dir = known_faces_dir
        self.known_faces_db = []
        self.database_file = 'face_database.pkl'
        self.model_name = 'VGG-Face'
        self.detection_threshold = 0.7
        self.camera = None
        self.is_running = False
        self.latest_alerts = []
        self.alert_lock = threading.Lock()
        
    def load_database(self):
        if os.path.exists(self.database_file):
            with open(self.database_file, 'rb') as f:
                self.known_faces_db = pickle.load(f)
            print(f"✓ Loaded {len(self.known_faces_db)} faces from database")
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
                    'confidence': (1 - distance) * 100,
                    'timestamp': datetime.now().isoformat()
                }
        return best_match
    
    def detect_and_recognize(self, frame):
        try:
            face_objs = DeepFace.extract_faces(
                img_path=frame,
                detector_backend='opencv',
                enforce_detection=False,
                align=False
            )
            
            for face_obj in face_objs:
                if face_obj['confidence'] > 0.5:
                    facial_area = face_obj['facial_area']
                    x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                    
                    padding = 10
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(frame.shape[1] - x, w + 2*padding)
                    h = min(frame.shape[0] - y, h + 2*padding)
                    
                    face_img = frame[y:y+h, x:x+w]
                    
                    if face_img.size > 0 and w > 30 and h > 30:
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
                                    # Criminal detected
                                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
                                    label = f"{match['name']}"
                                    conf = f"{match['confidence']:.1f}%"
                                    cv2.putText(frame, label, (x+5, y-25), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                    cv2.putText(frame, conf, (x+5, y-5), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                    
                                    # Add to alerts
                                    with self.alert_lock:
                                        self.latest_alerts.insert(0, match)
                                        if len(self.latest_alerts) > 10:
                                            self.latest_alerts = self.latest_alerts[:10]
                                    
                                    print(f"⚠️  CRIMINAL DETECTED: {match['name']} - {match['confidence']:.1f}%")
                                else:
                                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        except:
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        except:
            pass
        return frame
    
    def start_camera(self):
        if self.is_running:
            return False
        
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            return False
        
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        self.is_running = True
        return True
    
    def stop_camera(self):
        self.is_running = False
        if self.camera:
            self.camera.release()
            self.camera = None
    
    def generate_frames(self):
        frame_count = 0
        while self.is_running:
            ret, frame = self.camera.read()
            if not ret:
                break
            
            frame_count += 1
            
            if frame_count % 10 == 0:
                frame = self.detect_and_recognize(frame)
            
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# Initialize detector
detector = CriminalFaceDetector()
detector.load_database()

# Flask Routes
@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    """Start camera monitoring"""
    if detector.is_running:
        return jsonify({'error': 'Camera already running'}), 400
    
    if not detector.known_faces_db:
        if not detector.load_database():
            return jsonify({'error': 'Failed to load face database'}), 500
    
    if detector.start_camera():
        return jsonify({
            'message': 'Camera started successfully',
            'database_size': len(detector.known_faces_db)
        })
    else:
        return jsonify({'error': 'Failed to start camera'}), 500

@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    """Stop camera monitoring"""
    detector.stop_camera()
    return jsonify({'message': 'Camera stopped successfully'})

@app.route('/api/camera/feed')
def video_feed():
    """Video streaming route"""
    if not detector.is_running:
        return jsonify({'error': 'Camera not running'}), 400
    
    return Response(detector.generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/alerts/latest', methods=['GET'])
def get_latest_alerts():
    """Get latest criminal alerts"""
    with detector.alert_lock:
        alerts = detector.latest_alerts.copy()
    
    return jsonify({
        'alerts': alerts,
        'count': len(alerts),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/alerts/clear', methods=['POST'])
def clear_alerts():
    """Clear all alerts"""
    with detector.alert_lock:
        detector.latest_alerts = []
    
    return jsonify({'message': 'Alerts cleared'})

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status"""
    return jsonify({
        'camera_running': detector.is_running,
        'database_loaded': len(detector.known_faces_db) > 0,
        'database_size': len(detector.known_faces_db),
        'alerts_count': len(detector.latest_alerts),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Criminal Detection Flask API")
    print("="*60)
    print("\nEndpoints:")
    print("  POST   /api/camera/start      - Start camera")
    print("  POST   /api/camera/stop       - Stop camera")
    print("  GET    /api/camera/feed       - Video stream")
    print("  GET    /api/alerts/latest     - Get latest alerts")
    print("  POST   /api/alerts/clear      - Clear alerts")
    print("  GET    /api/status            - System status")
    print("  GET    /api/health            - Health check")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)
