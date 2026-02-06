"""
Home domain adapter for Roku AI.

Handles home-related queries:
- Smart home device control
- Home routines and preferences
- Family matters
- Home maintenance
"""

from typing import Dict, Any
from adapters.domains.base import DomainAdapter


class HomeAdapter(DomainAdapter):
    """Home-specific domain adapter."""
    
    def __init__(self):
        super().__init__(
            name="home",
            keywords=[
                "lights", "temperature", "thermostat", "lock", "door",
                "home", "house", "room", "bed", "security", "alarm",
                "garage", "kitchen", "living room", "bedroom",
                "family", "dinner", "cooking", "cleaning"
            ]
        )
    
    def prepare_context(
        self,
        message: str,
        context: Dict[str, Any],
        reasoning_layer: Any,
    ) -> str:
        """Prepare home-specific context."""
        lines = ["=== HOME CONTEXT ==="]
        
        # Smart home context is always relevant
        smart_home_chunks = reasoning_layer.store.retrieve(
            message,
            top_k=3,
            source_filter=["smart_home"]
        )
        if smart_home_chunks:
            lines.append("Smart Home Status:")
            for chunk, score in smart_home_chunks:
                lines.append(f"  {chunk.text}")
        
        # Home-related profile chunks
        home_chunks = reasoning_layer.store.retrieve(
            message,
            top_k=5,
            source_filter=["profile"]
        )
        home_profile = [
            chunk for chunk, score in home_chunks
            if chunk.metadata.get("section") in ["location", "preferences"]
        ]
        if home_profile:
            lines.append("\nHome Profile:")
            for chunk in home_profile:
                lines.append(f"  - {chunk.text}")
        
        lines.append("=== END HOME CONTEXT ===")
        return "\n".join(lines)
    
    def get_domain_prompt_addition(self) -> str:
        """Add home-specific instructions."""
        return (
            "When discussing home matters:\n"
            "- Be warm and practical\n"
            "- Reference smart home devices when relevant\n"
            "- Help with routines and preferences\n"
            "- Suggest home improvements or maintenance when appropriate"
        )
