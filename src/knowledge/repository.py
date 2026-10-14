"""Repository for managing, loading, and validating structured object knowledge."""

import json
import logging
from pathlib import Path
from src.schemas import KnowledgeRecord

logger = logging.getLogger(__name__)

# Canonical aliases map common variations to base knowledge keys
CANONICAL_ALIASES: dict[str, str] = {
    "bottle": "bottle",
    "water bottle": "bottle",
    "cup": "cup",
    "mug": "cup",
    "coffee cup": "cup",
    "cell phone": "cell phone",
    "cellphone": "cell phone",
    "phone": "cell phone",
    "mobile phone": "cell phone",
    "smartphone": "cell phone",
    "laptop": "laptop",
    "computer": "laptop",
    "pc": "laptop",
    "mouse": "mouse",
    "computer mouse": "mouse",
    "keyboard": "keyboard",
    "computer keyboard": "keyboard",
    "book": "book",
    "notebook": "book",
    "backpack": "backpack",
    "bag": "backpack",
    "chair": "chair",
    "office chair": "chair",
    "remote": "remote",
    "remote control": "remote",
    "scissors": "scissors",
    "shears": "scissors",
    "clock": "clock",
    "wall clock": "clock",
    "watch": "clock",
    "umbrella": "umbrella",
    "parasol": "umbrella",
    "handbag": "handbag",
    "purse": "handbag",
    "fork": "fork",
    "knife": "knife",
    "spoon": "spoon",
    "bowl": "bowl",
    "banana": "banana",
    "apple": "apple",
    "potted plant": "potted plant",
    "plant": "potted plant",
    "pottedplant": "potted plant",
    "pen": "pen",
    "ballpoint pen": "pen",
    "pencil": "pencil",
    "key": "keys",
    "keys": "keys",
    "keychain": "keys",
    "wallet": "wallet",
    "charger": "charger",
    "phone charger": "charger",
    "power adapter": "charger",
    "earbuds": "headphones",
    "earphones": "headphones",
    "airpods": "headphones",
    "headphones": "headphones",
    "calculator": "calculator",
    "glasses": "glasses",
    "eyeglasses": "glasses",
    "spectacles": "glasses",
    "sunglasses": "glasses",
}


class KnowledgeRepository:
    """Manages retrieval and validation of everyday object knowledge records."""

    def __init__(self, json_path: Path | str):
        self.path = Path(json_path)
        self._records: dict[str, KnowledgeRecord] = {}
        self.load()

    def load(self) -> None:
        """Load and validate JSON knowledge base from disk."""
        if not self.path.exists():
            raise FileNotFoundError(f"Knowledge database file not found at: {self.path}")

        with open(self.path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        loaded: dict[str, KnowledgeRecord] = {}
        for key, item in raw_data.items():
            normalized_key = key.strip().lower()
            record = KnowledgeRecord(
                canonical_name=item.get("canonical_name", key.capitalize()),
                description=item.get("description", ""),
                purpose=item.get("purpose", ""),
                common_uses=item.get("common_uses", []),
                components=item.get("components", []),
                care_notes=item.get("care_notes", []),
                safety_notes=item.get("safety_notes", []),
                limitations=item.get("limitations", []),
            )
            loaded[normalized_key] = record

        self._records = loaded
        logger.info(f"Loaded {len(self._records)} knowledge records from {self.path}")

    def normalize_key(self, label: str) -> str:
        """Resolve a label or variation to a canonical knowledge key."""
        clean = label.strip().lower()
        return CANONICAL_ALIASES.get(clean, clean)

    def get(self, label: str) -> KnowledgeRecord | None:
        """Retrieve the knowledge record for a given object label, or None if not found."""
        key = self.normalize_key(label)
        return self._records.get(key)

    def contains(self, label: str) -> bool:
        """Check if an object label exists in the repository."""
        key = self.normalize_key(label)
        return key in self._records

    def list_supported_classes(self) -> list[str]:
        """List all canonical object classes present in the repository."""
        return sorted(list(self._records.keys()))
