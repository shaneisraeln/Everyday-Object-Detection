"""YOLO-World open-vocabulary object detector backend.

Provides zero-shot text-prompted detection using Ultralytics YOLO-World models.
Implements the ObjectDetector protocol and allows dynamic prompt vocabulary updates.
"""

import logging
import re
from pathlib import Path
from typing import Any
import numpy as np

from src.schemas import Detection, ObjectDetector

logger = logging.getLogger(__name__)

# Configurable default prompt vocabulary for everyday desk and household objects
DEFAULT_YOLO_WORLD_CLASSES: list[str] = [
    "pen",
    "pencil",
    "keys",
    "wallet",
    "charger",
    "earbuds",
    "calculator",
    "glasses",
    "notebook",
    "mug",
    "book",
    "bottle",
    "phone",
    "laptop",
    "scissors",
    "backpack",
    "handbag",
    "watch",
    "remote control",
    "cup",
    "chair",
]


class YOLOWorldDetector(ObjectDetector):
    """Open-vocabulary object detector backend backed by YOLO-World checkpoints."""

    def __init__(
        self,
        model_path: str = "yolov8s-worldv2.pt",
        classes: list[str] | str | None = None,
        confidence_threshold: float = 0.25,
        max_detections: int = 10,
        device: str | None = None,
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.max_detections = max_detections
        self.device = device
        self._model: Any = None
        self._classes: list[str] = []
        self._class_names: dict[int, str] = {}

        # Validate initial classes
        initial_classes = classes if classes is not None else DEFAULT_YOLO_WORLD_CLASSES
        self._classes = self.validate_prompts(initial_classes)

        self._load_model()

    @staticmethod
    def validate_prompts(prompts: list[str] | str) -> list[str]:
        """Validate, clean, and deduplicate candidate text prompts.
        
        Args:
            prompts: List of prompt strings or a single comma/newline-separated string.
            
        Returns:
            Deduplicated list of cleaned non-empty prompt strings.
            
        Raises:
            ValueError: If no valid non-empty prompts are provided.
        """
        if isinstance(prompts, str):
            raw_items = re.split(r"[,;\n]+", prompts)
        elif isinstance(prompts, (list, tuple, set)):
            raw_items = list(prompts)
        else:
            raise ValueError(f"Prompts must be a list of strings or comma-separated string, got {type(prompts)}")

        cleaned: list[str] = []
        seen_lower: set[str] = set()

        for item in raw_items:
            if not isinstance(item, str):
                continue
            token = item.strip()
            if not token:
                continue
            token_lower = token.lower()
            if token_lower not in seen_lower:
                seen_lower.add(token_lower)
                cleaned.append(token)

        if not cleaned:
            raise ValueError("Class prompt vocabulary cannot be empty. Please provide at least one valid object label.")

        return cleaned

    def _load_model(self) -> None:
        """Load YOLO-World weights and initialize prompt vocabulary."""
        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"YOLO-World checkpoint not found at '{self.model_path}'. "
                f"Please ensure the weights file exists in the project root."
            )

        try:
            from ultralytics import YOLO

            logger.info(f"Loading YOLO-World from '{self.model_path}' on device '{self.device}'...")
            self._model = YOLO(str(path))

            # Apply candidate text prompts
            self._apply_classes(self._classes)
            logger.info(f"YOLO-World initialized with {len(self._classes)} candidate prompts.")
        except Exception as e:
            logger.error(f"Failed to initialize YOLO-World detector: {e}", exc_info=True)
            raise RuntimeError(f"YOLO-World initialization failed for '{self.model_path}': {e}") from e

    def _apply_classes(self, validated_classes: list[str]) -> None:
        """Internal helper to push classes to the underlying model and update class maps."""
        self._model.set_classes(validated_classes)
        names = getattr(self._model, "names", None)
        if isinstance(names, dict) and names:
            self._class_names = {int(k): str(v) for k, v in names.items()}
        else:
            self._class_names = {idx: name for idx, name in enumerate(validated_classes)}

    def set_classes(self, classes: list[str] | str) -> list[str]:
        """Dynamically update candidate text prompts without restarting the model.
        
        Args:
            classes: New candidate prompts as a list or comma-separated string.
            
        Returns:
            The newly active list of classes.
        """
        validated = self.validate_prompts(classes)
        self._apply_classes(validated)
        self._classes = validated
        logger.info(f"Updated YOLO-World prompts to: {self._classes}")
        return list(self._classes)

    def get_classes(self) -> list[str]:
        """Return a copy of currently active candidate prompts."""
        return list(self._classes)

    def get_classes_str(self) -> str:
        """Return currently active candidate prompts as a comma-separated string."""
        return ", ".join(self._classes)

    def reset_classes(self) -> list[str]:
        """Reset candidate prompts back to default everyday objects."""
        return self.set_classes(DEFAULT_YOLO_WORLD_CLASSES)

    @property
    def class_names(self) -> dict[int, str]:
        """Class ID to class name mapping for active vocabulary."""
        return self._class_names

    def detect(self, frame: np.ndarray, confidence: float | None = None) -> list[Detection]:
        """Run open-vocabulary inference on a frame.
        
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
                max_det=self.max_detections,
                verbose=False,
                device=self.device,
            )
        except Exception as e:
            logger.error(f"Error during YOLO-World prediction: {e}")
            return []

        detections: list[Detection] = []
        if not results:
            return detections

        first_res = results[0]
        boxes = first_res.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            conf = float(box.conf[0].item())
            if conf < active_conf:
                continue

            cls_id = int(box.cls[0].item())
            label = self._class_names.get(cls_id, f"object_{cls_id}").strip()

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

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections[: self.max_detections]
