"""Tests for Bedrock provider mocked execution and safe fallback to local knowledge."""

from unittest.mock import MagicMock, patch
import pytest
from src.schemas import ObjectContext, KnowledgeRecord
from src.qa.providers.bedrock_provider import BedrockAnswerProvider
from src.qa.providers.local_provider import LocalKnowledgeAnswerProvider


@pytest.fixture
def sample_context():
    return ObjectContext(label="bottle", confidence=0.91)


@pytest.fixture
def sample_knowledge():
    return KnowledgeRecord(
        canonical_name="Bottle",
        description="A container for liquids.",
        purpose="Holding liquids.",
        common_uses=["Drinking water"],
        components=["Cap", "Body"],
        care_notes=["Wash with soap"],
        safety_notes=["Avoid boiling water in thin plastics"],
        limitations=["Cannot identify contents"],
    )


def test_bedrock_success(sample_context, sample_knowledge):
    provider = BedrockAnswerProvider(model_id="anthropic.claude-3-haiku-20240307-v1:0")

    mock_client = MagicMock()
    mock_client.converse.return_value = {
        "output": {
            "message": {
                "content": [{"text": "This is a bottle, primarily used for holding drinking water."}]
            }
        }
    }

    with patch.object(provider, "_get_client", return_value=mock_client):
        result = provider.answer("What is this used for?", sample_context, sample_knowledge)

        assert result.grounded is True
        assert "This is a bottle" in result.answer
        assert result.warning is None


def test_bedrock_access_denied_fallback(sample_context, sample_knowledge):
    provider = BedrockAnswerProvider(model_id="anthropic.claude-3-haiku-20240307-v1:0")

    mock_client = MagicMock()
    mock_client.converse.side_effect = Exception("AccessDeniedException: User is not authorized")

    with patch.object(provider, "_get_client", return_value=mock_client):
        result = provider.answer("What is this used for?", sample_context, sample_knowledge)

        # Should fall back cleanly without raising exception
        assert result.grounded is True
        assert "Bottle" in result.answer
        assert "AWS credentials / model access unauthorized" in result.warning
        assert "Bedrock Fallback" in result.provider


def test_bedrock_timeout_fallback(sample_context, sample_knowledge):
    provider = BedrockAnswerProvider(model_id="anthropic.claude-3-haiku-20240307-v1:0")

    mock_client = MagicMock()
    mock_client.converse.side_effect = Exception("Read timeout on endpoint URL")

    with patch.object(provider, "_get_client", return_value=mock_client):
        result = provider.answer("How to clean it?", sample_context, sample_knowledge)

        assert result.grounded is True
        assert "clean" in result.answer.lower() or "wash" in result.answer.lower()
        assert "timed out" in result.warning.lower()


def test_bedrock_empty_response_fallback(sample_context, sample_knowledge):
    provider = BedrockAnswerProvider(model_id="anthropic.claude-3-haiku-20240307-v1:0")

    mock_client = MagicMock()
    mock_client.converse.return_value = {"output": {"message": {"content": []}}}

    with patch.object(provider, "_get_client", return_value=mock_client):
        result = provider.answer("What is this?", sample_context, sample_knowledge)

        assert result.grounded is True
        assert "Bottle" in result.answer
        assert "fell back" in result.warning.lower()
