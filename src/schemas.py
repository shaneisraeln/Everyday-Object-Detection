"""Data schemas, domain models, and error definitions for the application."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable
import time


class ErrorCode(str, Enum):
    CAMERA_UNAVAILABLE = "CAMERA_UNAVAILABLE"
    DETECTOR_INITIALIZATION_FAILED = "DETECTOR_INITIALIZATION_FAILED"
    NO_OBJECT_DETECTED = "NO_OBJECT_DETECTED"
    LOW_CONFIDENCE_DETECTION = "LOW_CONFIDENCE_DETECTION"
    OBJECT_SELECTION_REQUIRED = "OBJECT_SELECTION_REQUIRED"
    SPEECH_UNSUPPORTED = "SPEECH_UNSUPPORTED"
    SPEECH_PERMISSION_DENIED = "SPEECH_PERMISSION_DENIED"
    EMPTY_TRANSCRIPTION = "EMPTY_TRANSCRIPTION"
    KNOWLEDGE_ENTRY_MISSING = "KNOWLEDGE_ENTRY_MISSING"
    ANSWER_PROVIDER_UNAVAILABLE = "ANSWER_PROVIDER_UNAVAILABLE"
    BEDROCK_ACCESS_DENIED = "BEDROCK_ACCESS_DENIED"
    BEDROCK_THROTTLED = "BEDROCK_THROTTLED"
    ANSWER_TIMEOUT = "ANSWER_TIMEOUT"
    TTS_UNAVAILABLE = "TTS_UNAVAILABLE"


ERROR_MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.CAMERA_UNAVAILABLE: "Camera is currently unavailable or permissions are not granted. Please check your camera connection and browser permissions.",
    ErrorCode.DETECTOR_INITIALIZATION_FAILED: "Object detector failed to initialize. Please check model weights and system configuration.",
    ErrorCode.NO_OBJECT_DETECTED: "No object detected in the camera view. Please hold an object clearly in front of the lens.",
    ErrorCode.LOW_CONFIDENCE_DETECTION: "Detection confidence is below the minimum threshold. Please adjust lighting or bring the object closer.",
    ErrorCode.OBJECT_SELECTION_REQUIRED: "Multiple objects detected. Please select which object you would like to ask about.",
    ErrorCode.SPEECH_UNSUPPORTED: "Voice input is not supported in this browser. You can type your question in the text box.",
    ErrorCode.SPEECH_PERMISSION_DENIED: "Microphone permission was denied. Please allow microphone access in your browser settings.",
    ErrorCode.EMPTY_TRANSCRIPTION: "No speech recognized. Please speak clearly into the microphone or type your question.",
    ErrorCode.KNOWLEDGE_ENTRY_MISSING: "No knowledge record found for this object class.",
    ErrorCode.ANSWER_PROVIDER_UNAVAILABLE: "The selected answer provider is not available. Falling back to local knowledge base.",
    ErrorCode.BEDROCK_ACCESS_DENIED: "Amazon Bedrock access denied. Check AWS credentials and model access permissions.",
    ErrorCode.BEDROCK_THROTTLED: "Amazon Bedrock request rate exceeded. Please try again in a few moments.",
    ErrorCode.ANSWER_TIMEOUT: "Answer generation timed out. Please try again.",
    ErrorCode.TTS_UNAVAILABLE: "Text-to-speech is unavailable in this browser. Answers will be displayed on screen.",
}


@dataclass(frozen=True)
class Detection:
    """A single object detection result."""
    class_id: int
    label: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "bbox_xyxy": self.bbox_xyxy,
        }


@dataclass(frozen=True)
class ObjectContext:
    """Context of the currently selected or active object."""
    label: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int] | None = None
    knowledge_key: str | None = None
    observed_at: float = field(default_factory=time.time)

    def is_visible(self, current_detections: list[Detection]) -> bool:
        """Check if this object is still visible in current detections."""
        return any(d.label.lower() == self.label.lower() for d in current_detections)


@dataclass(frozen=True)
class KnowledgeRecord:
    """Structured knowledge entry for an everyday object."""
    canonical_name: str
    description: str
    purpose: str
    common_uses: list[str] = field(default_factory=list)
    components: list[str] = field(default_factory=list)
    care_notes: list[str] = field(default_factory=list)
    safety_notes: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "canonical_name": self.canonical_name,
            "description": self.description,
            "purpose": self.purpose,
            "common_uses": self.common_uses,
            "components": self.components,
            "care_notes": self.care_notes,
            "safety_notes": self.safety_notes,
            "limitations": self.limitations,
        }


@dataclass(frozen=True)
class AnswerResult:
    """Structured response from the question answering engine."""
    answer: str
    provider: str
    grounded: bool
    warning: str | None = None


@runtime_checkable
class ObjectDetector(Protocol):
    """Protocol for object detector implementations."""
    def detect(self, frame) -> list[Detection]:
        ...


@runtime_checkable
class AnswerProvider(Protocol):
    """Protocol for question answering providers."""
    def answer(
        self,
        question: str,
        object_context: ObjectContext,
        knowledge: KnowledgeRecord | None,
    ) -> AnswerResult:
        ...
