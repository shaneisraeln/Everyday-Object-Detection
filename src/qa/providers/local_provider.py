"""Local deterministic answer provider grounded in structured knowledge."""

import re
import logging
from src.schemas import ObjectContext, KnowledgeRecord, AnswerResult
from src.qa.providers.base import BaseAnswerProvider

logger = logging.getLogger(__name__)


class LocalKnowledgeAnswerProvider(BaseAnswerProvider):
    """Deterministic, offline answer provider grounded strictly in KnowledgeRecords."""

    @property
    def name(self) -> str:
        return "Local Knowledge Base"

    def answer(
        self,
        question: str,
        object_context: ObjectContext,
        knowledge: KnowledgeRecord | None,
    ) -> AnswerResult:
        """Deterministically answer questions based on question intent and knowledge record."""
        q = question.strip().lower()
        label = object_context.label

        if not knowledge:
            return AnswerResult(
                answer=(
                    f"I detected a {label} with {object_context.confidence:.0%} confidence, "
                    f"but there is currently no verified knowledge entry for this object category in the local database."
                ),
                provider=self.name,
                grounded=False,
                warning="Missing knowledge record for detected class",
            )

        name = knowledge.canonical_name

        # 1. Identity intent: "what is this", "what is that", "what is it", "who made this"
        if any(p in q for p in ["what is this", "what is that", "what is it", "what's this", "what's that", "identify"]):
            ans = f"This appears to be a {name} (detected with {object_context.confidence:.0%} confidence). {knowledge.description}"
            return AnswerResult(answer=ans, provider=self.name, grounded=True)

        # 2. Purpose / What is it used for: "used for", "purpose", "why do we use", "what does it do"
        if any(p in q for p in ["used for", "purpose", "why do we use", "what does it do", "why is it used", "function"]):
            uses = "; ".join(knowledge.common_uses[:2]) if knowledge.common_uses else knowledge.purpose
            ans = f"{name} is {knowledge.purpose.lower()} Common uses include: {uses}."
            return AnswerResult(answer=ans, provider=self.name, grounded=True)

        # 3. Common uses: "how is it used", "how do i use", "uses", "what can i do"
        if any(p in q for p in ["how is it used", "how to use", "how do i use", "common uses", "what can i do"]):
            if knowledge.common_uses:
                uses_str = ", ".join(knowledge.common_uses)
                ans = f"Common uses for a {name} include: {uses_str}."
            else:
                ans = f"{name} is primarily used to {knowledge.purpose.lower()}."
            return AnswerResult(answer=ans, provider=self.name, grounded=True)

        # 4. Components / Parts: "made of", "parts", "components", "pieces", "what does it contain"
        if any(p in q for p in ["parts", "components", "made of", "pieces", "what does it consist of"]):
            if knowledge.components:
                comps = ", ".join(knowledge.components)
                ans = f"The main components of a typical {name} are: {comps}."
            else:
                ans = f"I do not have specific component breakdowns recorded for {name}."
            return AnswerResult(answer=ans, provider=self.name, grounded=True)

        # 5. Care / Cleaning / Maintenance: "clean", "care", "wash", "maintain", "storage", "store"
        if any(p in q for p in ["clean", "wash", "care", "maintain", "maintenance", "store", "storage"]):
            if knowledge.care_notes:
                notes = " ".join(knowledge.care_notes)
                ans = f"Care guidance for a {name}: {notes}"
            else:
                ans = f"No specific cleaning or care instructions are recorded for {name}."
            return AnswerResult(answer=ans, provider=self.name, grounded=True)

        # 6. Safety / Hazards: "safe", "safety", "danger", "hazard", "precaution", "harm", "toxic"
        if any(p in q for p in ["safe", "safety", "danger", "hazard", "precaution", "harm", "toxic", "burn", "hot"]):
            if knowledge.safety_notes:
                notes = " ".join(knowledge.safety_notes)
                ans = f"Safety notes for a {name}: {notes}"
            else:
                ans = f"Exercise general common sense and caution when handling a {name}."
            return AnswerResult(answer=ans, provider=self.name, grounded=True)

        # 7. Limitations / Brand / Model / Contents: "what brand", "what model", "what is inside", "what type"
        if any(p in q for p in ["brand", "model", "inside", "type of", "who made", "price", "how much"]):
            limitations = " ".join(knowledge.limitations)
            ans = (
                f"I cannot determine specific details like brand, exact model, or internal contents. "
                f"{limitations}"
            )
            return AnswerResult(answer=ans, provider=self.name, grounded=True)

        # Default fallback summarizing the object
        ans = (
            f"Regarding the detected {name}: {knowledge.description} "
            f"Its primary purpose is {knowledge.purpose.lower()} "
            f"If you want to know about components, uses, care, or safety, feel free to ask!"
        )
        return AnswerResult(answer=ans, provider=self.name, grounded=True)
