import cv2
import numpy as np
import mediapipe as mp
from collections import deque

class GazeDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=4,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        self.LEFT_EYE_REF = [362, 263]
        self.RIGHT_EYE_REF = [33, 133]
        self.EAR_THRESHOLD = 0.15
        self.LOOKING_DOWN_THRESHOLD = 0.2
        
        # State smoothing
        self.state_buffer = deque(maxlen=5)  # Keep last 5 states
        self.state_buffer.append(("Unknown", False))  # Initial state
        
        # Distance threshold (person height as percentage of frame height)
        self.DISTANCE_THRESHOLD = 0.3  # If person height is less than 30% of frame height

    def get_eye_aspect_ratio(self, landmarks, eye_indices):
        points = np.array([landmarks[i] for i in eye_indices])
        vertical_dist = np.linalg.norm(points[1] - points[5]) + np.linalg.norm(points[2] - points[4])
        horizontal_dist = np.linalg.norm(points[0] - points[3])
        return vertical_dist / (2.0 * horizontal_dist)

    def get_gaze_ratio(self, landmarks, eye_indices, iris_indices, eye_ref_indices):
        eye_points = np.array([landmarks[i] for i in eye_indices])
        iris_points = np.array([landmarks[i] for i in iris_indices])
        ref_points = np.array([landmarks[i] for i in eye_ref_indices])
        eye_center = np.mean(eye_points, axis=0)
        iris_center = np.mean(iris_points, axis=0)
        eye_width = np.linalg.norm(ref_points[0] - ref_points[1])
        if eye_width > 0:
            gaze_vector = (iris_center - eye_center) / eye_width
        else:
            gaze_vector = iris_center - eye_center
        return gaze_vector, eye_center

    def calculate_head_pose(self, landmarks):
        # Get key facial landmarks for head pose
        nose = landmarks[1]  # Nose tip
        left_eye = landmarks[33]  # Left eye corner
        right_eye = landmarks[263]  # Right eye corner
        
        # Calculate head direction
        eye_center = (left_eye + right_eye) / 2
        head_direction = eye_center - nose
        
        # Calculate head tilt
        eye_vector = right_eye - left_eye
        head_tilt = np.arctan2(eye_vector[1], eye_vector[0])
        
        return head_direction, head_tilt

    def is_looking_at_screen(self, head_direction, head_tilt):
        norm = np.linalg.norm(head_direction)
        if norm == 0:
            return False
        
        normalized_direction = head_direction / norm
        is_facing_forward = abs(normalized_direction[0]) < 0.4 and normalized_direction[1] > -0.2
        is_not_tilted = abs(head_tilt) < np.pi/3  # 60 degrees
        
        return is_facing_forward and is_not_tilted

    def get_smoothed_state(self, new_state, new_eyes_closed):
        self.state_buffer.append((new_state, new_eyes_closed))
        
        # Count occurrences of each state
        state_counts = {}
        for state, _ in self.state_buffer:
            state_counts[state] = state_counts.get(state, 0) + 1
        
        # Get the most common state
        most_common_state = max(state_counts.items(), key=lambda x: x[1])[0]
        
        # If we have enough confidence in the state, return it
        if state_counts[most_common_state] >= 3:  # At least 3 out of 5 frames
            return most_common_state, new_eyes_closed
        
        return "Unknown", new_eyes_closed

    def is_person_far(self, person_height, frame_height):
        return person_height / frame_height < self.DISTANCE_THRESHOLD

    def detect_gaze(self, frame, person_bbox=None):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_h, frame_w = frame.shape[:2]
        results = self.face_mesh.process(frame_rgb)
        detections = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = np.array([[lm.x * frame_w, lm.y * frame_h] for lm in face_landmarks.landmark])
                
                # Check if we should use pose detection (if person is far)
                use_pose = False
                if person_bbox is not None:
                    x1, y1, x2, y2 = person_bbox
                    person_height = y2 - y1
                    use_pose = self.is_person_far(person_height, frame_h)

                # Gaze detection
                left_ear = self.get_eye_aspect_ratio(landmarks, self.LEFT_EYE)
                right_ear = self.get_eye_aspect_ratio(landmarks, self.RIGHT_EYE)
                avg_ear = (left_ear + right_ear) / 2

                left_gaze, left_center = self.get_gaze_ratio(landmarks, self.LEFT_EYE, self.LEFT_IRIS, self.LEFT_EYE_REF)
                right_gaze, right_center = self.get_gaze_ratio(landmarks, self.RIGHT_EYE, self.RIGHT_IRIS, self.RIGHT_EYE_REF)
                gaze_vector = (left_gaze + right_gaze) / 2
                face_center = (left_center + right_center) / 2

                eyes_closed = avg_ear < self.EAR_THRESHOLD
                looking_down = gaze_vector[1] > self.LOOKING_DOWN_THRESHOLD
                gaze_looking = abs(gaze_vector[0]) < 0.15 and abs(gaze_vector[1]) < 0.08 and not eyes_closed and not looking_down

                # Only use pose detection if person is far
                looking = gaze_looking
                if use_pose:
                    head_direction, head_tilt = self.calculate_head_pose(landmarks)
                    pose_looking = self.is_looking_at_screen(head_direction, head_tilt)
                    looking = (gaze_looking or pose_looking) and not eyes_closed
                
                # Get smoothed state
                state = "Watcher" if looking else "Non-Watcher"
                smoothed_state, smoothed_eyes_closed = self.get_smoothed_state(state, eyes_closed)
                
                detections.append((smoothed_state == "Watcher", gaze_vector, face_center, smoothed_eyes_closed))
        
        return detections 