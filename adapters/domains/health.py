"""
Health domain adapter for Roku AI.

Handles health-related queries:
- Fitness and exercise
- Sleep tracking
- Nutrition and diet
- Wellness and self-care
"""

from typing import Dict, Any
from adapters.domains.base import DomainAdapter


class HealthAdapter(DomainAdapter):
    """Health-specific domain adapter."""
    
    def __init__(self):
        super().__init__(
            name="health",
            keywords=[
                "workout", "exercise", "sleep", "steps", "calories",
                "health", "fitness", "medication", "vitamin", "run",
                "walk", "heart", "weight", "diet", "doctor",
                "gym", "yoga", "meditation", "wellness", "hydration"
            ]
        )
    
    def prepare_context(
        self,
        message: str,
        context: Dict[str, Any],
        reasoning_layer: Any,
    ) -> str:
        """Prepare health-specific context."""
        lines = ["=== HEALTH CONTEXT ==="]
        
        # Health-related profile chunks
        health_chunks = reasoning_layer.store.retrieve(
            message,
            top_k=5,
            source_filter=["profile"]
        )
        health_profile = [
            chunk for chunk, score in health_chunks
            if chunk.metadata.get("section") in ["goals", "preferences"]
        ]
        if health_profile:
            lines.append("Health Profile:")
            for chunk in health_profile:
                lines.append(f"  - {chunk.text}")
        
        # Time context is important for health (meal timing, sleep schedule)
        time_chunks = reasoning_layer.store.retrieve(
            message,
            top_k=2,
            source_filter=["time"]
        )
        if time_chunks:
            lines.append("\nTime Context:")
            for chunk, score in time_chunks:
                lines.append(f"  {chunk.text}")
        
        lines.append("=== END HEALTH CONTEXT ===")
        return "\n".join(lines)
    
    def post_process(self, response: str, context: Dict[str, Any]) -> str:
        """Add health disclaimers when appropriate."""
        # Add disclaimer for medical advice
        if any(word in response.lower() for word in ["diagnose", "treatment", "prescription", "medical condition"]):
            return response + "\n\n(Note: This is not medical advice. Consult a healthcare professional for medical concerns.)"
        return response
    
    def get_domain_prompt_addition(self) -> str:
        """Add health-specific instructions."""
        return (
            "When discussing health matters:\n"
            "- Be supportive and encouraging\n"
            "- Reference fitness goals and routines when relevant\n"
            "- Help track progress and suggest improvements\n"
            "- Remind about hydration, sleep, and self-care\n"
            "- NEVER provide medical diagnoses or treatment advice"
        )
