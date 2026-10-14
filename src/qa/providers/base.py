"""Base protocol and interface for question answering providers."""

from typing import Protocol
from src.schemas import ObjectContext, KnowledgeRecord, AnswerResult


class BaseAnswerProvider(Protocol):
    """Protocol that all QA providers must implement."""

    @property
    def name(self) -> str:
        """Display name of the provider."""
        ...

    def answer(
        self,
        question: str,
        object_context: ObjectContext,
        knowledge: KnowledgeRecord | None,
    ) -> AnswerResult:
        """Generate an answer given question, detected context, and knowledge."""
        ...
