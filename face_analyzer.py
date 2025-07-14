import cv2
import numpy as np
import logging
from insightface.app import FaceAnalysis

class FaceAnalyzer:
    def __init__(self, config):
        """
        Initializes the FaceAnalyzer with InsightFace.

        Args:
            config: The configuration object.
        """
        logging.info("Initializing FaceAnalyzer with InsightFace...")
        self.config = config
        self.app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'landmark_2d_106', 'recognition'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        logging.info("FaceAnalyzer (InsightFace) initialized successfully.")

    def get_faces(self, frame):
        """
        Detects faces in a frame using InsightFace.

        Args:
            frame (np.ndarray): The image in which to detect faces.

        Returns:
            list: A list of InsightFace 'Face' objects (containing bbox, landmark, embedding, etc.).
        """
        try:
            faces = self.app.get(frame)
            return faces
        except Exception as e:
            logging.error(f"Error detecting faces with InsightFace: {e}", exc_info=True)
            return []

