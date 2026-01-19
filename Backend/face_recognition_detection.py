from deepface import DeepFace
import cv2
import os
import pickle
import numpy as np
from datetime import datetime

class CriminalFaceTrainer:
    def __init__(self, known_faces_dir='scraped_data/images'):
        self.known_faces_dir = known_faces_dir
        self.known_faces_db = []
        self.database_file = 'face_database.pkl'
        self.model_name = 'VGG-Face'
        self.detection_threshold = 0.7
        
    def build_face_database(self):
        """Build face database from downloaded criminal images"""
        print("\n" + "="*60)
        print("Building Criminal Face Database...")
        print("="*60 + "\n")
        
        if not os.path.exists(self.known_faces_dir):
            print(f"✗ Directory {self.known_faces_dir} not found!")
            os.makedirs(self.known_faces_dir, exist_ok=True)
            print(f"✓ Created directory: {self.known_faces_dir}")
            return False
        
        image_files = [f for f in os.listdir(self.known_faces_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not image_files:
            print("✗ No images found in directory!")
            print(f"Please add criminal images to: {self.known_faces_dir}")
            return False
        
        print(f"Found {len(image_files)} images to process\n")
        
        self.known_faces_db = []
        success_count = 0
        
        for idx, image_file in enumerate(image_files):
            try:
                image_path = os.path.join(self.known_faces_dir, image_file)
                
                print(f"Processing [{idx+1}/{len(image_files)}]: {image_file}...", end=" ")
                
                # Extract face embedding
                embedding = DeepFace.represent(
                    img_path=image_path,
                    model_name=self.model_name,
                    enforce_detection=False,
                    detector_backend='opencv'
                )
                
                if embedding:
                    self.known_faces_db.append({
                        'name': image_file,
                        'path': image_path,
                        'embedding': embedding[0]['embedding']
                    })
                    success_count += 1
                    print("✓")
                else:
                    print("✗ No face detected")
                    
            except Exception as e:
                print(f"✗ Error: {str(e)}")
        
        print(f"\n{'='*60}")
        print(f"✓ Successfully processed {success_count}/{len(image_files)} faces")
        print(f"{'='*60}\n")
        
        if success_count > 0:
            self.save_database()
            return True
        return False
    
    def save_database(self):
        """Save face database to pickle file"""
        try:
            with open(self.database_file, 'wb') as f:
                pickle.dump(self.known_faces_db, f)
            print(f"✓ Saved database to {self.database_file}")
        except Exception as e:
            print(f"✗ Error saving database: {str(e)}")
    
    def load_database(self):
        """Load existing database"""
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
                    'confidence': (1 - distance) * 100
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
                                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
                                    label = f"{match['name']}"
                                    conf = f"{match['confidence']:.1f}%"
                                    cv2.putText(frame, label, (x+5, y-25), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                    cv2.putText(frame, conf, (x+5, y-5), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                    
                                    print(f"\n⚠️  CRIMINAL DETECTED: {match['name']} - {match['confidence']:.1f}%")
                                else:
                                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        except:
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        except:
            pass
        return frame
    
    def start_camera_detection(self):
        """Start camera and detect criminals"""
        print("\n" + "="*60)
        print("Starting Camera Detection...")
        print("="*60)
        print("Press 'q' to quit")
        print("="*60 + "\n")
        
        camera = cv2.VideoCapture(0)
        
        if not camera.isOpened():
            print("✗ Could not open camera!")
            return
        
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = camera.read()
                if not ret:
                    break
                
                frame_count += 1
                
                if frame_count % 10 == 0:
                    frame = self.detect_and_recognize(frame)
                
                cv2.imshow('Criminal Detection System - Press Q to Quit', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            camera.release()
            cv2.destroyAllWindows()
            print("\n✓ Camera stopped")

if __name__ == '__main__':
    trainer = CriminalFaceTrainer()
    
    # Build or load database
    if not trainer.load_database():
        print("No existing database found. Building new database...\n")
        if not trainer.build_face_database():
            print("\n✗ Failed to build database. Exiting...")
            exit(1)
    
    # Start camera detection automatically
    trainer.start_camera_detection()
