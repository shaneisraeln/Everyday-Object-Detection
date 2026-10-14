"""Manages active object selection, locking, and visibility state across frames."""

import time
import logging
from src.schemas import Detection, ObjectContext

logger = logging.getLogger(__name__)

BACKGROUND_LABELS = {"person"}


class ActiveObjectManager:
    """Tracks and manages the active object context for question answering."""

    def __init__(self):
        self._active_context: ObjectContext | None = None
        self._is_frozen: bool = False
        self._freeze_timestamp: float | None = None

    @property
    def current_context(self) -> ObjectContext | None:
        """Return the current active object context."""
        return self._active_context

    @property
    def is_frozen(self) -> bool:
        """Check if context is currently frozen for question answering."""
        return self._is_frozen

    def update_from_detections(
        self,
        detections: list[Detection],
        user_selected_index: int | None = None,
    ) -> tuple[ObjectContext | None, str]:
        """Update active context based on current detections and optional manual selection.
        
        Returns:
            Tuple of (ObjectContext or None, status_message)
        """
        # If frozen (mid-turn Q&A), do not update context from new camera frames
        if self._is_frozen:
            label = self._active_context.label if self._active_context else "None"
            return self._active_context, f"Context frozen on '{label}' during question processing"

        if not detections:
            if self._active_context:
                return (
                    self._active_context,
                    f"Selected object '{self._active_context.label}' is no longer visible in camera view",
                )
            return None, "No objects detected in camera feed"

        # Explicit user selection by index
        if user_selected_index is not None and 0 <= user_selected_index < len(detections):
            det = detections[user_selected_index]
            self._active_context = ObjectContext(
                label=det.label,
                confidence=det.confidence,
                bbox_xyxy=det.bbox_xyxy,
                knowledge_key=det.label.lower(),
                observed_at=time.time(),
            )
            return self._active_context, f"Selected object: {det.label} ({det.confidence:.0%})"

        # If exactly one detection is present, auto-select it
        if len(detections) == 1:
            det = detections[0]
            self._active_context = ObjectContext(
                label=det.label,
                confidence=det.confidence,
                bbox_xyxy=det.bbox_xyxy,
                knowledge_key=det.label.lower(),
                observed_at=time.time(),
            )
            return self._active_context, f"Active object: {det.label} ({det.confidence:.0%})"

        # If multiple detections exist:
        # 1. Check if the currently active object is still among them and is not just a person when other objects are present
        non_person_dets = [d for d in detections if d.label.lower() not in BACKGROUND_LABELS]

        if self._active_context:
            matching = [d for d in detections if d.label.lower() == self._active_context.label.lower()]
            # If current active was a person, but now a non-person object is held up, prefer the object
            if self._active_context.label.lower() in BACKGROUND_LABELS and non_person_dets:
                matching = []

            if matching:
                best_match = max(matching, key=lambda d: d.confidence)
                self._active_context = ObjectContext(
                    label=best_match.label,
                    confidence=best_match.confidence,
                    bbox_xyxy=best_match.bbox_xyxy,
                    knowledge_key=best_match.label.lower(),
                    observed_at=time.time(),
                )
                return (
                    self._active_context,
                    f"Multiple objects visible. Kept '{self._active_context.label}' ({self._active_context.confidence:.0%})",
                )

        # 2. If exactly one non-person everyday object exists alongside person/background, auto-focus it
        if len(non_person_dets) == 1:
            det = non_person_dets[0]
            self._active_context = ObjectContext(
                label=det.label,
                confidence=det.confidence,
                bbox_xyxy=det.bbox_xyxy,
                knowledge_key=det.label.lower(),
                observed_at=time.time(),
            )
            return self._active_context, f"Active object: {det.label} ({det.confidence:.0%})"

        # 3. Multiple everyday objects detected (e.g. bottle and cup) - require disambiguation
        labels = [f"{d.label} ({d.confidence:.0%})" for d in detections]
        return None, f"Multiple objects detected ({', '.join(labels)}). Please select one."

    def select_explicit(self, detection: Detection) -> ObjectContext:
        """Explicitly set active object context from a detection."""
        self._active_context = ObjectContext(
            label=detection.label,
            confidence=detection.confidence,
            bbox_xyxy=detection.bbox_xyxy,
            knowledge_key=detection.label.lower(),
            observed_at=time.time(),
        )
        return self._active_context

    def freeze_for_qa(self) -> ObjectContext | None:
        """Freeze the current object context so camera frames do not replace it during Q&A."""
        self._is_frozen = True
        self._freeze_timestamp = time.time()
        logger.info(f"Freezing object context: {self._active_context}")
        return self._active_context

    def unfreeze_qa(self) -> None:
        """Unfreeze object context after Q&A completes."""
        self._is_frozen = False
        self._freeze_timestamp = None
        logger.info("Unfrozen object context")

    def clear(self) -> None:
        """Clear the active object context and unfreeze."""
        self._active_context = None
        self._is_frozen = False
        self._freeze_timestamp = None
