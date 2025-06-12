import cv2
import numpy as np
import mediapipe as mp

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

    def detect_gaze(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_h, frame_w = frame.shape[:2]
        results = self.face_mesh.process(frame_rgb)
        detections = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = np.array([[lm.x * frame_w, lm.y * frame_h] for lm in face_landmarks.landmark])
                left_ear = self.get_eye_aspect_ratio(landmarks, self.LEFT_EYE)
                right_ear = self.get_eye_aspect_ratio(landmarks, self.RIGHT_EYE)
                avg_ear = (left_ear + right_ear) / 2

                left_gaze, left_center = self.get_gaze_ratio(landmarks, self.LEFT_EYE, self.LEFT_IRIS, self.LEFT_EYE_REF)
                right_gaze, right_center = self.get_gaze_ratio(landmarks, self.RIGHT_EYE, self.RIGHT_IRIS, self.RIGHT_EYE_REF)
                gaze_vector = (left_gaze + right_gaze) / 2
                face_center = (left_center + right_center) / 2

                eyes_closed = avg_ear < self.EAR_THRESHOLD
                looking_down = gaze_vector[1] > self.LOOKING_DOWN_THRESHOLD
                looking = abs(gaze_vector[0]) < 0.15 and abs(gaze_vector[1]) < 0.08 and not eyes_closed and not looking_down

                detections.append((looking, gaze_vector, face_center, eyes_closed))
        return detections 