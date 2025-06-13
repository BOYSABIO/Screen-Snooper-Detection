import cv2
import numpy as np
import platform
import time
import os
import logging
import threading
from datetime import datetime
from queue import Queue
from gaze_detector import GazeDetector
from object_detector import ObjectDetector

# Import appropriate notification library based on OS
if platform.system() == "Windows":
    import winsound
elif platform.system() == "Darwin":  # macOS
    import subprocess
    from pygame import mixer
    mixer.init()

# Set up logging
logging.basicConfig(
    filename='screen_watcher.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

class AdvancedFeatures:
    def __init__(self):
        self.last_notification_time = 0
        self.notification_cooldown = 5  # seconds between notifications
        self.last_screenshot_time = 0
        self.screenshot_cooldown = 10  # seconds between screenshots
        self.screenshot_dir = "screenshots"
        
        # Create screenshots directory if it doesn't exist
        if not os.path.exists(self.screenshot_dir):
            os.makedirs(self.screenshot_dir)
        
        # Initialize overlay window
        self.overlay = None
        self.overlay_visible = False
        
        # Load sound file if on macOS
        if platform.system() == "Darwin":
            self.sound_file = "alert.wav"  # You'll need to provide this file
            if os.path.exists(self.sound_file):
                mixer.music.load(self.sound_file)
        
        # Create queues for threaded operations
        self.screenshot_queue = Queue()
        
        # Start worker threads
        self.start_worker_threads()

    def start_worker_threads(self):
        """Start worker threads for non-critical operations."""
        self.screenshot_thread = threading.Thread(
            target=self._screenshot_worker,
            daemon=True
        )
        self.screenshot_thread.start()

    def _screenshot_worker(self):
        """Worker thread for handling screenshots."""
        while True:
            if not self.screenshot_queue.empty():
                frame = self.screenshot_queue.get()
                self._take_screenshot_internal(frame)
            time.sleep(0.1)

    def play_alert_sound(self):
        """Play an alert sound based on the operating system."""
        if platform.system() == "Windows":
            # Use a separate thread for sound to prevent blocking
            threading.Thread(
                target=lambda: winsound.Beep(1000, 500),
                daemon=True
            ).start()
        elif platform.system() == "Darwin":
            if os.path.exists(self.sound_file):
                threading.Thread(
                    target=lambda: mixer.music.play(),
                    daemon=True
                ).start()

    def show_overlay(self, frame):
        """Show a fullscreen overlay when someone is looking."""
        if self.overlay is None:
            # Get screen dimensions
            screen_width = 1920  # Default to common resolution
            screen_height = 1080
            
            # Create a fullscreen overlay
            self.overlay = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
            self.overlay[:] = (0, 0, 255)  # Red background
            
            # Add text to overlay
            text = "SCREEN WATCHER DETECTED!"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2
            thickness = 3
            
            # Get text size
            (text_width, text_height), _ = cv2.getTextSize(
                text, font, font_scale, thickness
            )
            
            # Calculate text position to center it
            text_x = (screen_width - text_width) // 2
            text_y = (screen_height + text_height) // 2
            
            # Add text to overlay
            cv2.putText(
                self.overlay, text, (text_x, text_y),
                font, font_scale, (255, 255, 255), thickness
            )
        
        # Create a named window for the overlay
        cv2.namedWindow('Alert Overlay', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Alert Overlay', cv2.WND_PROP_FULLSCREEN, 1)
        cv2.setWindowProperty('Alert Overlay', cv2.WND_PROP_TOPMOST, 1)
        
        # Show overlay
        cv2.imshow('Alert Overlay', self.overlay)
        self.overlay_visible = True

    def hide_overlay(self):
        """Hide the overlay window."""
        if self.overlay_visible:
            cv2.destroyWindow('Alert Overlay')
            self.overlay_visible = False

    def _take_screenshot_internal(self, frame):
        """Internal method to take screenshot (called by worker thread)."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.screenshot_dir}/screenshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        logging.info(f"Screenshot saved: {filename}")

    def take_screenshot(self, frame):
        """Queue a screenshot to be taken by the worker thread."""
        current_time = time.time()
        if (current_time - self.last_screenshot_time) > self.screenshot_cooldown:
            self.screenshot_queue.put(frame.copy())  # Use copy to prevent frame modification
            self.last_screenshot_time = current_time

    def log_detection(self, num_watchers):
        """Log when someone is detected looking at the screen."""
        logging.info(f"Detected {num_watchers} people looking at screen")

def test_advanced_features():
    # Initialize detectors and features
    object_detector = ObjectDetector()
    gaze_detector = GazeDetector()
    features = AdvancedFeatures()

    # Initialize camera
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not camera.isOpened():
        raise IOError("Cannot open camera")

    print("Press 'q' to quit the application")
    
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        
        # Detect people in the frame
        person_detections = object_detector.detect_people(frame)
        
        # Track if anyone is looking
        anyone_looking = False
        num_watchers = 0
        
        # Process each detected person
        for x1, y1, x2, y2 in person_detections:
            person_crop = frame[y1:y2, x1:x2]
            gaze_detections = gaze_detector.detect_gaze(person_crop, (x1, y1, x2, y2))
            
            label = "Unknown"
            if gaze_detections:
                looking, _, _, eyes_closed = gaze_detections[0]
                if eyes_closed:
                    label = "Eyes Closed"
                else:
                    label = "Watcher" if looking else "Non-Watcher"
                    if looking:
                        anyone_looking = True
                        num_watchers += 1
            
            color = (0, 255, 0) if label == "Watcher" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Handle advanced features
        current_time = time.time()
        if anyone_looking:
            # Show notification if cooldown has passed
            if (current_time - features.last_notification_time) > features.notification_cooldown:
                features.last_notification_time = current_time
                features.play_alert_sound()
                features.log_detection(num_watchers)
                features.take_screenshot(frame)
                features.show_overlay(frame)
        else:
            features.hide_overlay()

        cv2.imshow('Screen Watching Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_advanced_features() 