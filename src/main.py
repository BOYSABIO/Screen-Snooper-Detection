import cv2
from gaze_detector import GazeDetector
from object_detector import ObjectDetector
import platform
import time

# Import appropriate notification library based on OS
if platform.system() == "Windows":
    from win10toast import ToastNotifier
    toaster = ToastNotifier()
elif platform.system() == "Darwin":  # macOS
    import subprocess
elif platform.system() == "Linux":
    import subprocess

def show_notification(title, message):
    """Show a system notification based on the operating system."""
    if platform.system() == "Windows":
        toaster.show_toast(title, message, duration=3)
    elif platform.system() == "Darwin":  # macOS
        subprocess.run(['osascript', '-e', f'display notification "{message}" with title "{title}"'])
    elif platform.system() == "Linux":
        subprocess.run(['notify-send', title, message])

def main():
    # Initialize detectors
    object_detector = ObjectDetector()
    gaze_detector = GazeDetector()

    # Initialize camera
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not camera.isOpened():
        raise IOError("Cannot open camera")

    print("Press 'q' to quit the application")
    
    # Variables for notification cooldown
    last_notification_time = 0
    notification_cooldown = 5  # seconds between notifications
    
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        
        # Detect people in the frame
        person_detections = object_detector.detect_people(frame)
        
        # Track if anyone is looking
        anyone_looking = False
        
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
            
            color = (0, 255, 0) if label == "Watcher" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Show notification if someone is looking and cooldown has passed
        current_time = time.time()
        if anyone_looking and (current_time - last_notification_time) > notification_cooldown:
            show_notification(
                "Screen Watcher Detected!",
                "Someone is looking at your screen!"
            )
            last_notification_time = current_time

        cv2.imshow('Screen Watching Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 