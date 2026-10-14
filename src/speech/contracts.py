"""Contracts and data transfer models for speech input and speech synthesis."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class TranscriptionResult:
    """Result of speech recognition attempt."""
    text: str
    confidence: float = 1.0
    is_final: bool = True
    error: str | None = None


class SpeechInputProvider(Protocol):
    """Protocol for speech-to-text providers."""
    def transcribe(self, audio_data: bytes | None) -> TranscriptionResult:
        ...


class SpeechOutputProvider(Protocol):
    """Protocol for text-to-speech providers."""
    def speak(self, text: str) -> bool:
        ...
    def stop(self) -> None:
        ...
