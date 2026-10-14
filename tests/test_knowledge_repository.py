"""Tests for KnowledgeRepository loading, alias resolution, and retrieval."""

from pathlib import Path
import pytest
from src.knowledge.repository import KnowledgeRepository


@pytest.fixture
def repo():
    data_path = Path("src/knowledge/data/objects.json")
    return KnowledgeRepository(data_path)


def test_knowledge_repo_load_and_classes(repo):
    classes = repo.list_supported_classes()
    assert "bottle" in classes
    assert "cup" in classes
    assert "laptop" in classes
    assert len(classes) >= 10


def test_knowledge_repo_retrieval(repo):
    record = repo.get("bottle")
    assert record is not None
    assert record.canonical_name == "Bottle"
    assert len(record.common_uses) > 0
    assert len(record.components) > 0
    assert len(record.care_notes) > 0
    assert len(record.safety_notes) > 0
    assert len(record.limitations) > 0


def test_knowledge_repo_aliases(repo):
    # Test alias mappings
    assert repo.get("water bottle") is not None
    assert repo.get("water bottle").canonical_name == "Bottle"

    assert repo.get("mug") is not None
    assert repo.get("mug").canonical_name == "Cup"

    assert repo.get("smartphone") is not None
    assert repo.get("smartphone").canonical_name == "Cell Phone / Smartphone"


def test_knowledge_repo_missing_key(repo):
    res = repo.get("non_existent_object_xyz")
    assert res is None
    assert repo.contains("non_existent_object_xyz") is False
