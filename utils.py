import cv2
import numpy as np
from scipy.spatial import distance as dist

def calculate_ear(eye):
    """
    Calculates the Eye Aspect Ratio (EAR) for an eye.
    Args:
        eye: List of eye landmark points.
    Returns:
        EAR (float)
    """
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    if C == 0:
        return 0.0
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_face_quality(face_crop, landmarks):
    """
    Calculates a quality score for a face based on sharpness and pose.
    """
    if face_crop is None or face_crop.size == 0:
        return 0, "No face crop"
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = min(sharpness / 20, 100)
    pose_score = 0
    if landmarks is not None and len(landmarks) == 106:
        left_eye_center = np.mean(landmarks[35:41], axis=0)
        right_eye_center = np.mean(landmarks[89:95], axis=0)
        eyes_center = (left_eye_center + right_eye_center) / 2
        nose_tip = landmarks[49]
        horizontal_offset = abs(nose_tip[0] - eyes_center[0])
        interocular_distance = np.linalg.norm(left_eye_center - right_eye_center)
        pose_score = max(0, 100 - (horizontal_offset / interocular_distance) * 150)
    quality = 0.7 * sharpness_score + 0.3 * pose_score
    return quality, f"Sharpness: {sharpness_score:.1f}, Pose: {pose_score:.1f}"