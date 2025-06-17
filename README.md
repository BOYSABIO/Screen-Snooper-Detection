# Screen Snooper Detection

A computer vision system that detects when someone is looking at your screen, with support for transfer learning and advanced features.

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation and Model Training

### 1. Dataset Labeling with Grounding DINO

1. Open `notebooks/Data Labeling with Grounding DINO.ipynb`
2. Follow the notebook to:
   - Install Grounding DINO
   - Set up your dataset directory structure
   - Label your images using Grounding DINO's interactive interface
   - Export labeled data in YOLO format

### 2. Transfer Learning

1. Open `notebooks/People Training.ipynb`
2. The notebook will:
   - Load the YOLOv8 base model
   - Configure training parameters
   - Train on your labeled dataset
   - Save the trained model to `models/pretrained/best.pt`

## Running the Detection System

### Basic Usage

Run the program with default settings:
```bash
python src/main.py
```

### Command Line Options

- `--advanced`: Enable advanced features (overlay, screenshots, sound alerts)
- `--model`: Choose between models
  - `basic`: Uses YOLOv8n base model
  - `trained`: Uses transfer-learned model (default) yolov11n
- `--camera`: Specify camera index (default: 0)
- `--width`: Set camera width (default: 1280)
- `--height`: Set camera height (default: 720)

Example with advanced features:
```bash
python src/main.py --advanced --model trained --camera 0
```

### Advanced Features

When running with `--advanced`:
- Visual overlay appears when someone is detected looking at the screen
- Screenshots are automatically saved to the `screenshots` directory
- Sound alerts play when someone is detected
- Detection events are logged to `screen_watcher.log`

## Project Structure

```
.
├── models/
│   ├── datasets/          # Training datasets
│   ├── pretrained/        # Model weights
│   └── training_results/  # Training outputs
├── notebooks/
│   ├── Data Labeling with Grounding DINO.ipynb
│   └── People Training.ipynb
├── src/
│   ├── main.py           # Main program
│   ├── gaze_detector.py  # Gaze detection logic
│   ├── object_detector.py # Person detection
│   └── advanced_features.py # Additional features
├── screenshots/          # Captured screenshots
└── requirements.txt      # Project dependencies
```

## Technical Details

The system uses:
- Trained YOLOv11n for object detection
- MediaPipe for facial landmarks and gaze detection
- OpenCV for camera handling and visualization
- PyTorch for model training and inference

## Logging

The program logs detection events to `screen_watcher.log` with timestamps and detection details.
