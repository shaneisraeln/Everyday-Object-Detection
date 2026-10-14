"""Amazon Bedrock answer provider with robust error handling and local fallback."""

import json
import logging
from typing import Any
from src.schemas import ObjectContext, KnowledgeRecord, AnswerResult
from src.qa.providers.base import BaseAnswerProvider
from src.qa.providers.local_provider import LocalKnowledgeAnswerProvider
from src.qa.prompt_builder import build_grounded_prompt

logger = logging.getLogger(__name__)


class BedrockAnswerProvider(BaseAnswerProvider):
    """Answers questions using Amazon Bedrock Runtime with automatic fallback."""

    def __init__(
        self,
        region_name: str | None = None,
        model_id: str | None = None,
        timeout_seconds: int = 20,
        max_tokens: int = 250,
        fallback_provider: BaseAnswerProvider | None = None,
    ):
        self.region_name = region_name or "us-east-1"
        self.model_id = model_id or "anthropic.claude-3-haiku-20240307-v1:0"
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.fallback = fallback_provider or LocalKnowledgeAnswerProvider()
        self._client: Any = None

    @property
    def name(self) -> str:
        return f"Amazon Bedrock ({self.model_id})"

    def _get_client(self):
        """Lazily initialize boto3 bedrock-runtime client."""
        if self._client is None:
            import boto3
            from botocore.config import Config

            cfg = Config(
                connect_timeout=5,
                read_timeout=self.timeout_seconds,
                retries={"max_attempts": 2, "mode": "standard"},
            )
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self.region_name,
                config=cfg,
            )
        return self._client

    def answer(
        self,
        question: str,
        object_context: ObjectContext,
        knowledge: KnowledgeRecord | None,
    ) -> AnswerResult:
        """Query Amazon Bedrock with strictly grounded prompt; falls back to local provider on error."""
        try:
            client = self._get_client()
            system_prompt, user_prompt = build_grounded_prompt(
                question=question,
                object_context=object_context,
                knowledge=knowledge,
            )

            # Use the universal Bedrock Converse API
            response = client.converse(
                modelId=self.model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": user_prompt}],
                    }
                ],
                system=[{"text": system_prompt}],
                inferenceConfig={
                    "maxTokens": self.max_tokens,
                    "temperature": 0.3,
                    "topP": 0.9,
                },
            )

            # Parse response text
            output_msg = response.get("output", {}).get("message", {})
            content_blocks = output_msg.get("content", [])
            text_parts = [b.get("text", "") for b in content_blocks if "text" in b]
            answer_text = "\n".join(text_parts).strip()

            if not answer_text:
                raise ValueError("Empty response received from Bedrock model.")

            return AnswerResult(
                answer=answer_text,
                provider=self.name,
                grounded=True,
            )

        except Exception as e:
            err_msg = str(e)
            logger.warning(f"Bedrock invocation failed: {err_msg}. Triggering local fallback.")
            
            # Identify friendly warning reason
            warning = "Amazon Bedrock error, fell back to Local Knowledge Base."
            if "AccessDenied" in err_msg or "UnrecognizedClientException" in err_msg:
                warning = "AWS credentials / model access unauthorized; using Local Knowledge Base."
            elif "Throttling" in err_msg:
                warning = "Bedrock rate limit reached; using Local Knowledge Base."
            elif "timeout" in err_msg.lower():
                warning = "Bedrock request timed out; using Local Knowledge Base."

            # Fall back safely
            fallback_res = self.fallback.answer(question, object_context, knowledge)
            return AnswerResult(
                answer=fallback_res.answer,
                provider=f"{self.fallback.name} (Bedrock Fallback)",
                grounded=fallback_res.grounded,
                warning=warning,
            )
