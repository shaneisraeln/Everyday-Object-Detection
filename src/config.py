"""Application configuration management with environment variable loading."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load .env file from project root or current working directory
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class AppConfig:
    app_env: str = field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    detection_model: str = field(default_factory=lambda: os.getenv("DETECTION_MODEL", "yolo11m.pt"))
    detection_confidence: float = field(
        default_factory=lambda: float(os.getenv("DETECTION_CONFIDENCE", "0.30"))
    )
    detection_interval_ms: int = field(
        default_factory=lambda: int(os.getenv("DETECTION_INTERVAL_MS", "250"))
    )
    max_detections: int = field(
        default_factory=lambda: int(os.getenv("MAX_DETECTIONS", "10"))
    )
    device: str = field(
        default_factory=lambda: os.getenv(
            "DEVICE",
            "cuda:0" if __import__("torch").cuda.is_available() else "cpu"
        )
    )

    answer_provider: str = field(
        default_factory=lambda: os.getenv("ANSWER_PROVIDER", "local").lower()
    )
    bedrock_region: str | None = field(
        default_factory=lambda: os.getenv("BEDROCK_REGION", "").strip() or None
    )
    bedrock_model_id: str | None = field(
        default_factory=lambda: os.getenv("BEDROCK_MODEL_ID", "").strip() or None
    )
    bedrock_timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("BEDROCK_TIMEOUT_SECONDS", "20"))
    )
    bedrock_max_output_tokens: int = field(
        default_factory=lambda: int(os.getenv("BEDROCK_MAX_OUTPUT_TOKENS", "250"))
    )

    knowledge_base_path: Path = field(
        default_factory=lambda: BASE_DIR / os.getenv("KNOWLEDGE_BASE_PATH", "src/knowledge/data/objects.json")
    )
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper())

    def validate(self) -> list[str]:
        """Validate config values, returning warnings/errors."""
        issues = []
        if not (0.0 <= self.detection_confidence <= 1.0):
            issues.append(f"DETECTION_CONFIDENCE must be between 0.0 and 1.0, got {self.detection_confidence}")
        if self.detection_interval_ms < 50:
            issues.append(f"DETECTION_INTERVAL_MS should be at least 50ms, got {self.detection_interval_ms}")
        if self.max_detections <= 0:
            issues.append(f"MAX_DETECTIONS must be positive, got {self.max_detections}")
        if self.answer_provider not in ("local", "bedrock"):
            issues.append(f"ANSWER_PROVIDER must be 'local' or 'bedrock', got {self.answer_provider}")
        if self.answer_provider == "bedrock" and not self.bedrock_model_id:
            issues.append("BEDROCK_MODEL_ID is not configured for Bedrock answer provider; fallback to local will be used.")
        return issues


def get_config() -> AppConfig:
    """Retrieve and validate the application configuration."""
    cfg = AppConfig()
    return cfg
