import logging
from insightface.app import FaceAnalysis
import os

# Ensures that the models directory exists
os.makedirs(os.path.expanduser('~/.insightface/models'), exist_ok=True)

logging.basicConfig(level=logging.INFO)
logging.info("--- Start of insightface initialization test ---")
logging.info("This script will download the necessary models. The operation may take a few minutes...")

try:
    # Initialize FaceAnalysis, which will trigger the download of the models
    app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'landmark_2d_106'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    logging.info("--- Insightface initialization test completed successfully! ---")
    logging.info("The models have been downloaded and loaded correctly.")
except Exception as e:
    logging.error(f"An error occurred during the test: {e}", exc_info=True)

