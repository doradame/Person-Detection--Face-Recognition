
<p align="center">
  <img src="https://mojalab.com/content/images/size/w1440/2025/07/Person_and_face_Detection_2.webp" alt="AI Security Mascot" width="500"/>
</p>



# Real-Time Face Recognition and Liveness Detection

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)


This project is a real-time video processing pipeline that performs face detection, recognition, and liveness detection on multiple video streams. It's designed to be robust and scalable, utilizing a multi-threaded architecture to handle I/O operations and stream processing concurrently.

👉 **Read the full article on MojaLab:**  
[https://mojalab.com/person-detection-and-face-recognition-with-liveness-detection-a-python-project-using-powerful-ai-models/](https://mojalab.com/person-detection-and-face-recognition-with-liveness-detection-a-python-project-using-powerful-ai-models/)

## Features

*   **Multi-Stream Processing**: Can process multiple video streams (e.g., from RTSP cameras or local files) simultaneously.
*   **Face Detection**: Utilizes the YOLO (You Only Look Once) model to detect individuals in the video frames.
*   **Face Recognition**: Employs the InsightFace library for accurate face recognition, comparing detected faces against a database of known profiles.
*   **Liveness Detection**: Implements a blink-based liveness detection mechanism to prevent spoofing attacks.
*   **Automatic Profile Creation**: Automatically creates new profiles for unknown individuals who pass the liveness check.
*   **Dynamic Profile Management**: Continuously updates and prunes profiles with new, high-quality face samples.
*   **Configuration via Environment Variables**: Easily configurable through a `.env` file.
*   **State Management**: Maintains and updates the state of detected individuals in a JSON file.

## How It Works

The application follows these main steps:

1.  **Stream Capturing**: It captures video frames from the specified sources (e.g., RTSP URLs).
2.  **Person Detection**: The YOLO model processes each frame to detect people.
3.  **Face Analysis**: For each detected person, the system crops the corresponding area and uses InsightFace to find faces, extract facial embeddings, and locate facial landmarks.
4.  **Face Recognition**: The extracted facial embedding is compared against the embeddings of known profiles.
    *   If a match is found with a high enough similarity score, the person is identified.
    *   If the face is recognized, and the face quality is high, the new face sample is added to the existing profile to improve it.
5.  **Liveness Detection for Unknown Faces**: If the face is not recognized, the system initiates a liveness check.
    *   It tracks the person's eyes and counts the number of blinks.
    *   If the person performs the required number of blinks, they are considered "live."
6.  **New Profile Promotion**: After a successful liveness check, the system starts collecting high-quality face samples of the new individual.
    *   Once enough samples are collected, a new profile is created and saved.
7.  **State Updates**: The application continuously updates a JSON file with the current state of all detected individuals (both known and unknown).

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/doradame/Person-Detection--Face-Recognition
    cd Person-Detection--Face-Recognition
    ```

2.  **Create and activate a virtual environment (recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the YOLO model:**

    The default model is `yolov8n.pt`. If you don't have it, it will be downloaded automatically on the first run.

## Configuration

The project is configured using a `.env` file. Create a file named `.env` in the root of the project directory and add the following variables:

```
# --- Models and Source ---
# Path to the YOLO model file
MODEL_PATH=yolov8n.pt
# Comma-separated list of RTSP URLs or video file paths (use '0' for webcam)
RTSP_URLS=0
# Directory to store known face profiles
PROFILES_DIR=known_faces
# Path to the log file
LOG_PATH=logs/detections.log

# --- Detection Parameters ---
# Class ID for person detection in the YOLO model (0 is typically 'person')
DETECT_CLASS=0
# Minimum confidence score for YOLO detections
MIN_YOLO_CONFIDENCE=0.75

# --- Recognition and Liveness Parameters ---
# Minimum similarity score to consider a face as recognized
RECOGNITION_THRESHOLD=0.4
# Eye Aspect Ratio (EAR) threshold for blink detection
EAR_THRESHOLD=0.57
# Minimum quality score for a face to be added as a sample
FACE_QUALITY_THRESHOLD=8.0
# Number of blinks required for a successful liveness check
LIVENESS_MIN_BLINKS=2

# --- Promotion Parameters ---
# Number of high-quality samples required to create a new profile
PROMOTION_MIN_SAMPLES=5

# --- Tracking Parameters ---
# Number of frames to skip between processing
FRAME_SKIP=0
# Maximum number of frames a tracker can go without seeing the person
TRACKER_MAX_UNSEEN=10
# Maximum distance (in pixels) to match a face to an existing tracker
TRACKER_MAX_DISTANCE=75

# --- Face Recognition Backend ---
# The face recognition library to use (currently only 'insightface' is supported)
FACE_RECOGNITION_BACKEND=insightface

# --- State Management Parameters ---
# Path to the file that stores the current state
STATE_FILE_PATH=current_state.json
# Path to the file that stores the history of states
HISTORY_FILE_PATH=state_history.jsonl
# Interval in seconds to update the state file
STATE_UPDATE_INTERVAL_SECONDS=5

# --- Profile Management ---
# Interval in seconds for periodic pruning of profiles
PRUNING_INTERVAL_SECONDS=60
# Maximum number of samples to keep per profile
MAX_SAMPLES_PER_PROFILE=20
```

### Renaming Known Faces

When a new, unknown person is detected and passes the liveness check, a new profile is created for them in the `known_faces` directory (or the directory you specified in `PROFILES_DIR`). The folder will be named with a unique identifier (UUID).

To associate a name with a profile, you need to rename the subfolder within `known_faces`. For example, if a new profile is created with the folder name `e3f5b5f5-4f8d-4b1e-8b0a-9e8f6a7b1c2d`, and you know this person is "John Doe", you should rename the folder to `John_Doe`.

The `rename_profile.py` script is provided to facilitate this. You can run it as follows:

```bash
python rename_profile.py <old_name> <new_name>
```

For example:

```bash
python rename_profile.py e3f5b5f5-4f8d-4b1e-8b0a-9e8f6a7b1c2d John_Doe
```

This will rename the folder and update the profile information accordingly.

## Usage

1.  **Ensure your `.env` file is configured correctly.**
2.  **Run the main application:**

    ```bash
    python main.py
    ```

3.  **View the streams:**

    The application will open a window for each video stream, showing the processed video with bounding boxes and labels.

4.  **Stop the application:**

    Press 'q' in one of the stream windows or use Ctrl+C in the terminal to gracefully shut down the application.

## Project Structure

```
.
├── .env                  # Configuration file (you need to create this)
├── .gitignore            # Git ignore file
├── README.md             # This file
├── config.py             # Loads and provides configuration variables
├── face_analyzer.py      # Handles face analysis using InsightFace
├── inspect_model.py      # (Utility script, needs further investigation)
├── known_faces/          # Directory to store profiles of known individuals
├── logger_config.py      # Configures the logging for the application
├── main.py               # Main entry point of the application
├── profile_manager.py    # Manages the creation, loading, and recognition of profiles
├── rename_profile.py     # Script to rename a profile
├── requirements.txt      # Python dependencies
├── state_manager.py      # Manages the state of detected individuals
└── utils.py              # Utility functions (e.g., EAR calculation)
```
