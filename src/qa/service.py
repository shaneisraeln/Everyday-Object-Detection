"""High-level Question Answering service orchestrator."""

import logging
from src.schemas import ObjectContext, AnswerResult, ErrorCode, ERROR_MESSAGES
from src.knowledge.repository import KnowledgeRepository
from src.qa.providers.base import BaseAnswerProvider
from src.qa.providers.local_provider import LocalKnowledgeAnswerProvider
from src.qa.providers.bedrock_provider import BedrockAnswerProvider
from src.config import AppConfig

logger = logging.getLogger(__name__)


class QuestionAnsweringService:
    """Coordinates validation, knowledge retrieval, and provider execution."""

    def __init__(self, config: AppConfig, knowledge_repo: KnowledgeRepository):
        self.config = config
        self.repo = knowledge_repo
        self.local_provider = LocalKnowledgeAnswerProvider()
        
        # Configure active provider based on settings
        if config.answer_provider == "bedrock" and config.bedrock_model_id:
            self.provider: BaseAnswerProvider = BedrockAnswerProvider(
                region_name=config.bedrock_region,
                model_id=config.bedrock_model_id,
                timeout_seconds=config.bedrock_timeout_seconds,
                max_tokens=config.bedrock_max_output_tokens,
                fallback_provider=self.local_provider,
            )
        else:
            self.provider = self.local_provider

    def answer_question(
        self,
        question: str,
        object_context: ObjectContext | None,
    ) -> AnswerResult:
        """Validate input and generate a knowledge-grounded answer."""
        # 1. Validate question text
        clean_q = question.strip() if question else ""
        if not clean_q:
            return AnswerResult(
                answer=ERROR_MESSAGES[ErrorCode.EMPTY_TRANSCRIPTION],
                provider=self.provider.name,
                grounded=False,
                warning="Empty question submitted",
            )

        # 2. Validate object context
        if not object_context:
            return AnswerResult(
                answer="No object is currently selected. Please point the camera at an everyday object and select it before asking a question.",
                provider=self.provider.name,
                grounded=False,
                warning="No active object context",
            )

        # 3. Retrieve knowledge record
        knowledge = self.repo.get(object_context.label)

        # 4. Generate answer
        try:
            return self.provider.answer(clean_q, object_context, knowledge)
        except Exception as e:
            logger.error(f"Unexpected error in QA service: {e}", exc_info=True)
            # Safe user-facing fallback
            fallback_res = self.local_provider.answer(clean_q, object_context, knowledge)
            return AnswerResult(
                answer=fallback_res.answer,
                provider=self.local_provider.name,
                grounded=fallback_res.grounded,
                warning="An error occurred; generated answer using local fallback.",
            )
