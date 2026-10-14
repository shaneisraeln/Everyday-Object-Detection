"""OpenCV Camera capture service."""

import logging
from typing import Generator
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraService:
    """Manages OpenCV VideoCapture lifecycle safely."""

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self._cap: cv2.VideoCapture | None = None
        self._is_running = False

    @property
    def is_running(self) -> bool:
        return self._is_running and self._cap is not None and self._cap.isOpened()

    def start(self) -> bool:
        """Open camera stream."""
        if self._cap is not None:
            self.stop()

        logger.info(f"Attempting to open camera at index {self.camera_index}...")
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            logger.warning(f"Could not open camera index {self.camera_index}.")
            self._cap = None
            self._is_running = False
            return False

        self._is_running = True
        logger.info("Camera successfully started.")
        return True

    def read_frame(self) -> np.ndarray | None:
        """Capture single frame in RGB format."""
        if not self.is_running or self._cap is None:
            return None

        ret, bgr_frame = self._cap.read()
        if not ret or bgr_frame is None:
            logger.warning("Failed to read frame from camera.")
            return None

        # Convert to RGB for Gradio and standard web display
        return cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

    def stop(self) -> None:
        """Release camera resource."""
        if self._cap is not None:
            logger.info("Releasing camera capture resource...")
            try:
                self._cap.release()
            except Exception as e:
                logger.warning(f"Error releasing camera: {e}")
            self._cap = None
        self._is_running = False

    def __del__(self):
        self.stop()
