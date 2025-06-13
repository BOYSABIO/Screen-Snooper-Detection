import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import torch

class GazeDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Constants for gaze detection
        self.EAR_THRESHOLD = 0.15
        self.LOOKING_DOWN_THRESHOLD = 0.2
        
    def detect_face(self, frame):
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0]
        return None
        
    def detect_gaze(self, face_landmarks):
        # Get eye landmarks
        left_eye = [face_landmarks.landmark[33], face_landmarks.landmark[160],
                   face_landmarks.landmark[158], face_landmarks.landmark[133],
                   face_landmarks.landmark[153], face_landmarks.landmark[144]]
        
        right_eye = [face_landmarks.landmark[362], face_landmarks.landmark[385],
                    face_landmarks.landmark[387], face_landmarks.landmark[263],
                    face_landmarks.landmark[373], face_landmarks.landmark[380]]
        
        # Calculate eye aspect ratio
        left_ear = self._calculate_ear(left_eye)
        right_ear = self._calculate_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0
        
        # Get iris position
        left_iris = face_landmarks.landmark[468]
        right_iris = face_landmarks.landmark[473]
        
        # Check if looking at screen
        is_eyes_open = ear > self.EAR_THRESHOLD
        is_looking_down = (left_iris.y + right_iris.y) / 2 > self.LOOKING_DOWN_THRESHOLD
        
        return is_eyes_open and not is_looking_down
        
    def _calculate_ear(self, eye):
        # Calculate the eye aspect ratio
        A = np.linalg.norm(np.array([eye[1].x - eye[5].x, eye[1].y - eye[5].y]))
        B = np.linalg.norm(np.array([eye[2].x - eye[4].x, eye[2].y - eye[4].y]))
        C = np.linalg.norm(np.array([eye[0].x - eye[3].x, eye[0].y - eye[3].y]))
        return (A + B) / (2.0 * C)

def main():
    # Detect and select the best available device
    if torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA for acceleration")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Apple Silicon) for acceleration")
    else:
        device = "cpu"
        print("Using CPU (no GPU acceleration available)")

    # Initialize models
    det_model = YOLO('yolov8n.pt')
    det_model.to(device)
    
    gaze_detector = GazeDetector()
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run YOLO detection
        results = det_model(frame)
        
        # Process each detection
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if int(box.cls[0]) == 0:  # Assuming class 0 is person
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    if conf > 0.5:  # Confidence threshold
                        # Get face landmarks
                        face_landmarks = gaze_detector.detect_face(frame)
                        
                        if face_landmarks is not None:
                            # Detect gaze
                            is_looking_at_screen = gaze_detector.detect_gaze(face_landmarks)
                            
                            # Draw bounding box
                            color = (0, 255, 0) if is_looking_at_screen else (0, 0, 255)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            
                            # Add label
                            label = f"{'Looking at screen' if is_looking_at_screen else 'Not looking at screen'}"
                            cv2.putText(frame, label, (x1, y1 - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Display the frame
        cv2.imshow('Gaze Detection', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 