import logging
import os
import sys
from logging.handlers import RotatingFileHandler
import config

def setup_logging():
    """
    Sets up centralized logging for the application.
    """
    # Remove any pre-configured handlers from the root logger to avoid duplication.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # If the log file exists, delete it for a clean log at each startup.
    if os.path.exists(config.LOG_PATH):
        try:
            os.remove(config.LOG_PATH)
        except OSError as e:
            print(f"Error deleting log file: {e}")

    # Create log directory if it doesn't exist, as per the configuration file.
    log_directory = os.path.dirname(config.LOG_PATH)
    os.makedirs(log_directory, exist_ok=True)

    # Define a standard formatter for all handlers.
    # We add the module name and line number for easier debugging.
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Configure the root logger.
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG) # Set to DEBUG to capture more details.

    # 1. Handler to write to a file with automatic rotation.
    # Rotates the log file when it reaches 2MB and keeps up to 5 backup files.
    file_handler = RotatingFileHandler(
        config.LOG_PATH, maxBytes=2*1024*1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 2. Handler to write to the console (stdout).
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logging.info("--- Centralized logging configuration complete. ---")
