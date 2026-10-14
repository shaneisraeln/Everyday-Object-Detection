"""Builds strictly grounded, concise prompts for LLM question answering."""

from src.schemas import ObjectContext, KnowledgeRecord


def build_grounded_prompt(
    question: str,
    object_context: ObjectContext,
    knowledge: KnowledgeRecord | None,
) -> tuple[str, str]:
    """Constructs system prompt and user prompt strictly grounded in knowledge.
    
    Returns:
        (system_prompt, user_prompt)
    """
    system_prompt = (
        "You are an everyday object knowledge assistant. Your role is to answer questions about objects "
        "detected through a camera. Follow these rules strictly:\n"
        "1. Treat the object label as a detector prediction with the specified confidence score, not unquestionable truth.\n"
        "2. Answer ONLY using the supplied verified object knowledge and low-risk, everyday general facts.\n"
        "3. Do NOT invent, assume, or hallucinate specific brands, product models, materials, hidden contents, or historical origins that are not in the knowledge.\n"
        "4. If the question asks for information not covered in the knowledge or visual label, state clearly and honestly that this cannot be determined.\n"
        "5. Keep your answer concise (2 to 4 sentences), spoken-friendly, clear, and direct.\n"
        "6. Do not provide hazardous advice or instructions."
    )

    if knowledge:
        knowledge_block = (
            f"Object: {knowledge.canonical_name}\n"
            f"Description: {knowledge.description}\n"
            f"Purpose: {knowledge.purpose}\n"
            f"Common Uses: {', '.join(knowledge.common_uses)}\n"
            f"Components: {', '.join(knowledge.components)}\n"
            f"Care Instructions: {'; '.join(knowledge.care_notes)}\n"
            f"Safety & Precautions: {'; '.join(knowledge.safety_notes)}\n"
            f"Known Limitations: {'; '.join(knowledge.limitations)}"
        )
    else:
        knowledge_block = (
            f"Object Label: {object_context.label}\n"
            "Note: Detailed knowledge record is unavailable for this object."
        )

    user_prompt = (
        f"[Detected Object Information]\n"
        f"Predicted Class: {object_context.label}\n"
        f"Detection Confidence: {object_context.confidence:.1%}\n\n"
        f"[Verified Knowledge Record]\n"
        f"{knowledge_block}\n\n"
        f"[User Question]\n"
        f"{question.strip()}\n\n"
        "Please provide a concise, natural, and grounded answer suitable to be read aloud:"
    )

    return system_prompt, user_prompt
