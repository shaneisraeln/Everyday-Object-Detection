"""Ultralytics YOLO object detector implementation with Everyday Object filtering."""

import logging
from typing import Any
import numpy as np
from src.schemas import Detection, ObjectDetector

logger = logging.getLogger(__name__)

# Allowed indoor and everyday classes (filters out wild animals, outdoor vehicles, etc.)
EVERYDAY_CLASSES = {
    "bottle", "water bottle", "cup", "coffee cup", "mug", "wine glass",
    "scissors", "pair of scissors",
    "cell phone", "phone", "smartphone", "laptop", "mouse", "computer mouse", "keyboard",
    "book", "pen", "pencil", "backpack", "handbag", "bag", "suitcase", "umbrella",
    "remote", "remote control", "clock", "watch", "wristwatch",
    "glasses", "eyeglasses", "sunglasses", "headphones",
    "chair", "couch", "potted plant", "bowl", "fork", "knife", "spoon",
    "apple", "banana", "orange", "sandwich", "toothbrush",
    "tv", "monitor", "microwave", "oven", "toaster", "sink", "refrigerator", "vase",
    "person",
}

YOLO_WORLD_CLASSES = [
    "scissors", "pair of scissors", "water bottle", "bottle", "coffee cup", "cup",
    "cell phone", "smartphone", "laptop", "computer mouse", "keyboard", "book",
    "backpack", "handbag", "umbrella", "remote control", "clock", "watch", "wristwatch",
    "eyeglasses", "glasses", "headphones", "pen", "chair", "potted plant", "apple",
]


class YOLODetector(ObjectDetector):
    """Encapsulates an Ultralytics YOLO model with everyday object filtering and GPU acceleration."""

    def __init__(
        self,
        model_name: str = "yolo11m.pt",
        confidence_threshold: float = 0.30,
        max_detections: int = 10,
        device: str | None = None,
        filter_everyday: bool = True,
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.max_detections = max_detections
        self.device = device
        self.filter_everyday = filter_everyday
        self._model: Any = None
        self._class_names: dict[int, str] = {}
        self._load_model()

    def _load_model(self) -> None:
        """Load model weights, configure device, and extract class vocabulary."""
        try:
            from ultralytics import YOLO

            logger.info(f"Loading YOLO model weights from '{self.model_name}' on device '{self.device}'...")
            self._model = YOLO(self.model_name)

            if "world" in self.model_name.lower():
                try:
                    self._model.set_classes(YOLO_WORLD_CLASSES)
                    logger.info("Configured YOLO-World with tailored everyday vocabulary.")
                except Exception as ex:
                    logger.warning(f"Could not set custom classes for YOLO-World: {ex}")

            self._class_names = self._model.names or {}
            logger.info(f"Successfully loaded YOLO model with {len(self._class_names)} classes.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}", exc_info=True)
            raise RuntimeError(f"Detector initialization failed for model '{self.model_name}': {e}") from e

    def switch_model(self, model_name: str) -> None:
        """Switch the underlying model weights dynamically."""
        if model_name == self.model_name and self._model is not None:
            return
        self.model_name = model_name
        self._load_model()

    @property
    def class_names(self) -> dict[int, str]:
        """Class id to class name dictionary from model vocabulary."""
        return self._class_names

    def detect(self, frame: np.ndarray, confidence: float | None = None) -> list[Detection]:
        """Run object detection inference on a single frame.
        
        Args:
            frame: Image array in BGR or RGB format.
            confidence: Optional confidence threshold override.
            
        Returns:
            List of Detection objects sorted by descending confidence.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return []

        active_conf = confidence if confidence is not None else self.confidence_threshold

        try:
            results = self._model.predict(
                source=frame,
                conf=active_conf,
                max_det=self.max_detections * 2,  # allow extra before everyday filtering
                verbose=False,
                device=self.device,
            )
        except Exception as e:
            logger.error(f"Error during YOLO prediction: {e}")
            return []

        detections: list[Detection] = []
        if not results:
            return detections

        first_res = results[0]
        boxes = first_res.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            if conf < active_conf:
                continue

            label = self._class_names.get(cls_id, f"class_{cls_id}").strip()

            # Filter out non-everyday classes (e.g. wild animals like 'bird', vehicles like 'airplane')
            if self.filter_everyday:
                clean_lbl = label.lower()
                is_everyday = any(
                    clean_lbl == ec or ec in clean_lbl or clean_lbl in ec
                    for ec in EVERYDAY_CLASSES
                )
                if not is_everyday:
                    logger.debug(f"Filtered out non-everyday class: {label} ({conf:.2f})")
                    continue

            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

            detections.append(
                Detection(
                    class_id=cls_id,
                    label=label,
                    confidence=conf,
                    bbox_xyxy=(x1, y1, x2, y2),
                )
            )

        # Sort detections by descending confidence
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections[: self.max_detections]
