"""Tests for QuestionAnsweringService and LocalKnowledgeAnswerProvider."""

from pathlib import Path
import pytest
from src.schemas import ObjectContext
from src.config import AppConfig
from src.knowledge.repository import KnowledgeRepository
from src.qa.service import QuestionAnsweringService
from src.qa.providers.local_provider import LocalKnowledgeAnswerProvider


@pytest.fixture
def qa_service():
    repo = KnowledgeRepository("src/knowledge/data/objects.json")
    cfg = AppConfig(answer_provider="local")
    return QuestionAnsweringService(cfg, repo)


def test_empty_question(qa_service):
    ctx = ObjectContext(label="bottle", confidence=0.9)
    res = qa_service.answer_question("", ctx)
    assert res.grounded is False
    assert "No speech recognized" in res.answer or "type your question" in res.answer


def test_no_object_selected(qa_service):
    res = qa_service.answer_question("What is this?", None)
    assert res.grounded is False
    assert "No object is currently selected" in res.answer


def test_answer_identity(qa_service):
    ctx = ObjectContext(label="bottle", confidence=0.88)
    res = qa_service.answer_question("What is this?", ctx)
    assert res.grounded is True
    assert "Bottle" in res.answer
    assert "88%" in res.answer
    assert res.provider == "Local Knowledge Base"


def test_answer_purpose(qa_service):
    ctx = ObjectContext(label="cup", confidence=0.92)
    res = qa_service.answer_question("What is this used for?", ctx)
    assert res.grounded is True
    assert "Cup" in res.answer
    assert "drinking" in res.answer.lower() or "liquids" in res.answer.lower()


def test_answer_care(qa_service):
    ctx = ObjectContext(label="laptop", confidence=0.95)
    res = qa_service.answer_question("How should I clean and take care of this?", ctx)
    assert res.grounded is True
    assert "clean" in res.answer.lower() or "care" in res.answer.lower()


def test_answer_safety(qa_service):
    ctx = ObjectContext(label="scissors", confidence=0.89)
    res = qa_service.answer_question("What are the safety precautions?", ctx)
    assert res.grounded is True
    assert "safety" in res.answer.lower() or "blades" in res.answer.lower() or "caution" in res.answer.lower()


def test_acknowledge_limitations_for_brand_or_contents(qa_service):
    ctx = ObjectContext(label="bottle", confidence=0.85)
    res = qa_service.answer_question("What brand is this bottle and what is inside it?", ctx)
    assert res.grounded is True
    assert "cannot determine" in res.answer.lower() or "brand" in res.answer.lower()


def test_unsupported_knowledge_class(qa_service):
    ctx = ObjectContext(label="unknown_gadget_3000", confidence=0.7)
    res = qa_service.answer_question("What is this?", ctx)
    assert res.grounded is False
    assert "no verified knowledge entry" in res.answer
