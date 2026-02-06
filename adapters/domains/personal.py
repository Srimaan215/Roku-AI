"""
Personal domain adapter for Roku AI.

Handles personal queries:
- Preferences and hobbies
- Friends and relationships
- Personal reminders
- General conversation
"""

from typing import Dict, Any
from adapters.domains.base import DomainAdapter


class PersonalAdapter(DomainAdapter):
    """Personal domain adapter."""
    
    def __init__(self):
        super().__init__(
            name="personal",
            keywords=[
                "remind", "remember", "preference", "favorite", "hobby",
                "friend", "family", "birthday", "anniversary", "like",
                "enjoy", "love", "music", "movie", "book", "game"
            ]
        )
    
    def prepare_context(
        self,
        message: str,
        context: Dict[str, Any],
        reasoning_layer: Any,
    ) -> str:
        """Prepare personal context."""
        lines = ["=== PERSONAL CONTEXT ==="]
        
        # Personal preferences and identity
        personal_chunks = reasoning_layer.store.retrieve(
            message,
            top_k=5,
            source_filter=["profile"]
        )
        personal_profile = [
            chunk for chunk, score in personal_chunks
            if chunk.metadata.get("section") in ["identity", "preferences", "goals"]
        ]
        if personal_profile:
            lines.append("Personal Profile:")
            for chunk in personal_profile:
                lines.append(f"  - {chunk.text}")
        
        lines.append("=== END PERSONAL CONTEXT ===")
        return "\n".join(lines)
    
    def get_domain_prompt_addition(self) -> str:
        """Add personal-specific instructions."""
        return (
            "When discussing personal matters:\n"
            "- Be warm, friendly, and conversational\n"
            "- Remember preferences and past conversations\n"
            "- Help with reminders and personal organization\n"
            "- Show interest in hobbies and interests"
        )
