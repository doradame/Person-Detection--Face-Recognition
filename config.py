import os
from dotenv import load_dotenv

load_dotenv(override=True)

# --- Models and Source ---
MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n.pt")
RTSP_URLS_STR = os.getenv("RTSP_URLS", "0")
RTSP_URLS = [int(url) if url.strip().isdigit() else url.strip() for url in RTSP_URLS_STR.split(',')]
PROFILES_DIR = os.getenv("PROFILES_DIR", "known_faces")
LOG_PATH = os.getenv("LOG_PATH", "logs/detections.log")

# --- Detection Parameters ---
DETECT_CLASS = int(os.getenv("DETECT_CLASS", 0))
MIN_YOLO_CONFIDENCE = float(os.getenv("MIN_YOLO_CONFIDENCE", 0.75))

# --- Recognition and Liveness Parameters ---
RECOGNITION_THRESHOLD = float(os.getenv("RECOGNITION_THRESHOLD", 0.4))
EAR_THRESHOLD = float(os.getenv("EAR_THRESHOLD", 0.57))
QUALITY_THRESHOLD = float(os.getenv("FACE_QUALITY_THRESHOLD", 8.0))
LIVENESS_MIN_BLINKS = int(os.getenv("LIVENESS_MIN_BLINKS", 2))

# --- Promotion Parameters ---
PROMOTION_MIN_SAMPLES = int(os.getenv("PROMOTION_MIN_SAMPLES", 5))

# --- Tracking Parameters ---
FRAME_SKIP = int(os.getenv("FRAME_SKIP", 0))
TRACKER_MAX_UNSEEN = int(os.getenv("TRACKER_MAX_UNSEEN", 10))
TRACKER_MAX_DISTANCE = int(os.getenv("TRACKER_MAX_DISTANCE", 75))

# --- Face Recognition Backend ---
FACE_RECOGNITION_BACKEND = os.getenv("FACE_RECOGNITION_BACKEND", "insightface")

# --- State Management Parameters ---
STATE_FILE_PATH = os.getenv("STATE_FILE_PATH", "current_state.json")
HISTORY_FILE_PATH = os.getenv("HISTORY_FILE_PATH", "state_history.jsonl")
STATE_UPDATE_INTERVAL_SECONDS = int(os.getenv("STATE_UPDATE_INTERVAL_SECONDS", 5))

# --- Profile Management ---
PRUNING_INTERVAL_SECONDS = int(os.getenv("PRUNING_INTERVAL_SECONDS", 60))
MAX_SAMPLES_PER_PROFILE = int(os.getenv("MAX_SAMPLES_PER_PROFILE", 20))