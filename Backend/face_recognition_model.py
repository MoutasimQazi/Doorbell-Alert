from deepface import DeepFace
import cv2
import os
import pickle
import numpy as np
from datetime import datetime
import pandas as pd

class CriminalFaceRecognizer:
    def __init__(self, known_faces_dir='scraped_data/images'):
        self.known_faces_dir = known_faces_dir
        self.known_faces_db = []
        self.encodings_file = 'face_database.pkl'
        self.model_name = 'VGG-Face'  # Options: VGG-Face, Facenet, OpenFace, DeepFace, DeepID, ArcFace
        
    def build_face_database(self):
        """Build face database from downloaded criminal images"""
        print("\n" + "="*60)
        print("Building criminal face database...")
        print("="*60 + "\n")
        
        if not os.path.exists(self.known_faces_dir):
            print(f"✗ Directory {self.known_faces_dir} not found!")
            return
        
        image_files = [f for f in os.listdir(self.known_faces_dir) 
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print("✗ No images found in directory!")
            return
        
        print(f"Found {len(image_files)} images to process\n")
        
        self.known_faces_db = []
        
        for idx, image_file in enumerate(image_files):
            try:
                image_path = os.path.join(self.known_faces_dir, image_file)
                
                # Extract face embedding using DeepFace
                embedding = DeepFace.represent(
                    img_path=image_path,
                    model_name=self.model_name,
                    enforce_detection=False
                )
                
                if embedding:
                    self.known_faces_db.append({
                        'name': image_file,
                        'path': image_path,
                        'embedding': embedding[0]['embedding']
                    })
                    print(f"✓ [{idx+1}/{len(image_files)}] Processed: {image_file}")
                else:
                    print(f"✗ [{idx+1}/{len(image_files)}] No face found in: {image_file}")
                    
            except Exception as e:
                print(f"✗ [{idx+1}/{len(image_files)}] Error processing {image_file}: {str(e)}")
        
        print(f"\n✓ Successfully processed {len(self.known_faces_db)} faces")
        self.save_database()
    
    def save_database(self):
        """Save face database to file"""
        with open(self.encodings_file, 'wb') as f:
            pickle.dump(self.known_faces_db, f)
        print(f"✓ Saved database to {self.encodings_file}")
    
    def load_database(self):
        """Load pre-computed face database"""
        if os.path.exists(self.encodings_file):
            with open(self.encodings_file, 'rb') as f:
                self.known_faces_db = pickle.load(f)
            print(f"✓ Loaded {len(self.known_faces_db)} faces from database")
            return True
        return False
    
    def detect_criminal_in_frame(self, frame):
        """Detect if any criminal face matches in the given frame"""
        matches = []
        
        try:
            # Detect and extract faces from frame
            face_objs = DeepFace.extract_faces(
                img_path=frame,
                detector_backend='opencv',
                enforce_detection=False
            )
            
            for face_obj in face_objs:
                if face_obj['confidence'] > 0.9:
                    # Get face region
                    facial_area = face_obj['facial_area']
                    x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                    
                    # Extract face for comparison
                    face_img = frame[y:y+h, x:x+w]
                    
                    # Get embedding for detected face
                    try:
                        detected_embedding = DeepFace.represent(
                            img_path=face_img,
                            model_name=self.model_name,
                            enforce_detection=False
                        )
                        
                        if detected_embedding:
                            # Compare with known criminals
                            best_match = self.find_best_match(detected_embedding[0]['embedding'])
                            
                            if best_match:
                                matches.append({
                                    'name': best_match['name'],
                                    'confidence': best_match['confidence'],
                                    'location': (y, x+w, y+h, x)
                                })
                    except:
                        pass
                        
        except Exception as e:
            pass
        
        return matches
    
    def find_best_match(self, embedding, threshold=0.6):
        """Find best matching criminal face"""
        if not self.known_faces_db:
            return None
        
        min_distance = float('inf')
        best_match = None
        
        for criminal in self.known_faces_db:
            # Calculate cosine distance
            distance = self.cosine_distance(embedding, criminal['embedding'])
            
            if distance < min_distance and distance < threshold:
                min_distance = distance
                best_match = {
                    'name': criminal['name'],
                    'confidence': 1 - distance,
                    'distance': distance
                }
        
        return best_match
    
    def cosine_distance(self, embedding1, embedding2):
        """Calculate cosine distance between two embeddings"""
        embedding1 = np.array(embedding1)
        embedding2 = np.array(embedding2)
        
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        similarity = dot_product / (norm1 * norm2)
        distance = 1 - similarity
        
        return distance
    
    def monitor_camera(self, camera_index=0, display=True):
        """Monitor camera feed for criminal faces"""
        if not self.known_faces_db:
            if not self.load_database():
                print("✗ No face database found. Run build_face_database() first!")
                return
        
        print("\n" + "="*60)
        print("Starting camera monitoring...")
        print("Press 'q' to quit")
        print("="*60 + "\n")
        
        video_capture = cv2.VideoCapture(camera_index)
        
        if not video_capture.isOpened():
            print("✗ Could not open camera!")
            return
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = video_capture.read()
                
                if not ret:
                    print("✗ Failed to grab frame")
                    break
                
                frame_count += 1
                
                # Process every 10th frame for performance (DeepFace is slower)
                if frame_count % 10 == 0:
                    matches = self.detect_criminal_in_frame(frame)
                    
                    if matches:
                        print(f"\n⚠️  ALERT! Criminal detected at {datetime.now().strftime('%H:%M:%S')}")
                        for match in matches:
                            print(f"   - {match['name']} (Confidence: {match['confidence']:.2%})")
                            
                            if display:
                                # Draw rectangle around face
                                top, right, bottom, left = match['location']
                                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                                
                                # Draw label
                                label = f"{match['name']} ({match['confidence']:.1%})"
                                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                                cv2.putText(frame, label, (left + 6, bottom - 6), 
                                          cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                
                if display:
                    cv2.imshow('Criminal Detection System', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
            print("\n✓ Camera monitoring stopped")

# Example usage
if __name__ == '__main__':
    recognizer = CriminalFaceRecognizer()
    
    # Step 1: Build face database (run once after downloading images)
    recognizer.build_face_database()
    
    # Step 2: Start monitoring camera
    # recognizer.monitor_camera(camera_index=0, display=True)
