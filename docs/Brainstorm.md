# Screen Snooper Detector

A real-time computer vision system that detects when unauthorized individuals are looking at your screen—protecting your privacy in public or shared spaces using object detection, head pose estimation, and gaze tracking.

---

## Project Overview

This project uses a webcam to monitor the user's environment and detect if anyone other than the primary user is looking at their screen. The system triggers alerts when it detects suspicious gaze behavior from others in the vicinity.

---

## Objectives

- Detect people in the webcam's field of view.
- Estimate head and gaze direction for each person.
- Distinguish between the **primary user** and potential **snoopers**.
- Trigger an alert when unauthorized gaze is directed toward the screen.

---

## 🛠️ Features

- ✅ Real-time webcam integration
- ✅ Object and face detection using YOLOv8 / Mediapipe
- ✅ Head pose and gaze estimation
- ✅ Snoop detection logic with thresholding
- ✅ Visual overlay and alert system

---

## System Architecture

```plaintext
[Webcam Frame]
     ↓
[YOLOv8 / Face Detection]
     ↓
[Face Landmarks & Pose Estimation]
     ↓
[Gaze Direction Estimation]
     ↓
[Snooper Logic]
     ↓
[Real-Time Alerts / Overlays]
```

---

## Technologies & Tools

| Component             | Tool/Library               |
|----------------------|----------------------------|
| Object Detection      | YOLOv8 (Ultralytics)       |
| Face Landmarks        | Mediapipe / dlib           |
| Gaze Estimation       | Gaze360 / OpenGaze         |
| Real-Time Video       | OpenCV                     |
| Tracking (optional)   | DeepSORT / IOU-based       |
| Alerts / UI           | OpenCV overlay / sound     |
| Deployment (optional) | Streamlit / Gradio         |

---

## Project Structure

```
screen-snooper-detector/
│
├── models/                # YOLOv8 weights / Gaze models
├── notebooks/             # Prototyping and experiments
├── src/                   # Main source code
│   ├── detection.py       # Object & face detection logic
│   ├── pose.py            # Head pose estimation
│   ├── gaze.py            # Gaze estimation
│   ├── snooper_logic.py   # Decision logic for snooping
│   └── main.py            # Real-time video processing loop
│
├── assets/                # Demo images/videos
├── README.md              # This file
└── requirements.txt       # Python dependencies
```

---

## Key Functional Logic

### Primary User Identification
- Largest/closest face = assumed to be the user
- Or use 1-time registration at launch

### Gaze Estimation
- Calculate yaw/pitch using facial landmarks
- Use gaze vector to check if looking at screen area

### Alert Trigger
- Trigger if `n > 1` people are looking at the screen
- Ignore registered user face
- Optional: add delay or time threshold

---

## Dataset Resources

| Task            | Dataset                        |
|-----------------|---------------------------------|
| Gaze Estimation | [Gaze360](https://vcai.mpi-inf.mpg.de/projects/Gaze360/) |
| Face Landmarks  | Included in Mediapipe/dlib      |
| Pose Estimation | Use pretrained + webcam data    |
| Custom Training | Capture webcam footage if needed |

---

## Suggested Timeline

### Week 1:
- Set up repo, webcam feed, YOLOv8 or Mediapipe detection
- Extract faces and run pose estimation

### Week 2:
- Integrate gaze estimation model (e.g., Gaze360)
- Calculate gaze vectors in screen space

### Week 3:
- Implement snooper detection logic
- Filter out primary user
- Build live alerts/visual overlays

### Week 4:
- Test in different lighting and scenes
- Record demo video
- Finalize report and presentation

---

## Evaluation Metrics

- 🔍 Detection accuracy (YOLO precision/recall)
- 👁️ Gaze estimation correctness
- 🚨 False positive/negative rate for snooper alerts
- ⏱️ Latency: Real-time performance at ≥15 FPS

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run real-time detector
python src/main.py
```

---

## Demo (To be added)

- [ ] Live screen demo GIF or YouTube link
- [ ] Example of false vs. true positives

---

## Future Improvements

- Train a lightweight custom gaze model
- Add audio alerts or screen dimming
- Track snooping over time (privacy analytics)
- Deploy as desktop app or Chrome extension

---
