import cv2
import argparse
import logging
from object_detector import ObjectDetector
from gaze_detector import GazeDetector
from advanced_features import AdvancedFeatures

def setup_logging():
    logging.basicConfig(
        filename='screen_watcher.log',
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Screen Snooper Detection with optional advanced features'
    )
    parser.add_argument(
        '--advanced', 
        action='store_true',
        help='Enable advanced features (overlay, screenshots, sound alerts)'
    )
    parser.add_argument(
        '--model', 
        choices=['basic', 'trained'],
        default='trained',
        help='Choose between basic YOLO (basic) or transfer-learned model (trained)'
    )
    parser.add_argument(
        '--camera', 
        type=int, 
        default=0,
        help='Camera index to use (default: 0)'
    )
    parser.add_argument(
        '--width', 
        type=int, 
        default=1280,
        help='Camera width (default: 1280)'
    )
    parser.add_argument(
        '--height', 
        type=int, 
        default=720,
        help='Camera height (default: 720)'
    )
    args = parser.parse_args()

    # Set up logging
    setup_logging()

    # Initialize detectors with selected model
    model_path = 'models/pretrained/yolov11n.pt' if args.model == 'basic' else 'models/pretrained/best.pt'
    object_detector = ObjectDetector(model_path=model_path)
    gaze_detector = GazeDetector()
    
    # Initialize advanced features if requested
    features = AdvancedFeatures() if args.advanced else None

    # Initialize camera
    camera = cv2.VideoCapture(args.camera)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    if not camera.isOpened():
        raise IOError("Cannot open camera")

    print("Press 'q' to quit the application")
    print(f"Advanced features: {'Enabled' if args.advanced else 'Disabled'}")
    print(f"Using model: {args.model}")
    
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
            cv2.putText(
                frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
            )

        # Handle advanced features if enabled
        if args.advanced and features:
            if anyone_looking:
                features.show_overlay(frame)
                features.play_alert_sound()
                features.take_screenshot(frame)
                features.log_detection(num_watchers)
            else:
                features.hide_overlay()

        # Show the frame
        cv2.imshow('Screen Snooper Detection', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 