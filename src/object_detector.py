import cv2
import torch
from ultralytics import YOLO


class ObjectDetector:
    def __init__(self, model_path='models/pretrained/best.pt'):
        self.device = self._get_device()
        self.model = YOLO(model_path)
        self.model.to(self.device)

    def _get_device(self):
        if torch.cuda.is_available():
            device = "cuda"
            print(
                f"Using CUDA for acceleration on GPU: {torch.cuda.get_device_name(0)}"
            )
        elif torch.backends.mps.is_available():
            device = "mps"
            print("Using MPS (Apple Silicon) for acceleration")
        else:
            device = "cpu"
            print("WARNING: Running on CPU. This will be slow!")
        return device

    def detect_people(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(frame_rgb)[0]
        detections = []
        
        for box in results.boxes:
            if int(box.cls[0]) == 0:  # Class 0 is person in COCO dataset
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append((x1, y1, x2, y2))
        
        return detections 