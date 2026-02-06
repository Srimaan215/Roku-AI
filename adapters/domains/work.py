"""
Work domain adapter for Roku AI.

Handles work-related queries:
- Meetings and scheduling
- Projects and deadlines
- Professional communication
- Work preferences and context
"""

from typing import Dict, Any
from adapters.domains.base import DomainAdapter


class WorkAdapter(DomainAdapter):
    """Work-specific domain adapter."""
    
    def __init__(self):
        super().__init__(
            name="work",
            keywords=[
                "meeting", "email", "project", "deadline", "schedule",
                "calendar", "client", "presentation", "report",
                "colleague", "boss", "office", "work", "task", "todo",
                "conference", "call", "zoom", "team", "manager"
            ]
        )
    
    def prepare_context(
        self,
        message: str,
        context: Dict[str, Any],
        reasoning_layer: Any,
    ) -> str:
        """Prepare work-specific context."""
        lines = ["=== WORK CONTEXT ==="]
        
        # Retrieve work-related profile chunks
        work_chunks = reasoning_layer.store.retrieve(
            message,
            top_k=5,
            source_filter=["profile"]
        )
        
        # Filter for work-related chunks
        work_profile = [
            chunk for chunk, score in work_chunks
            if chunk.metadata.get("section") in ["work", "schedule", "goals"]
        ]
        
        if work_profile:
            lines.append("Work Profile:")
            for chunk in work_profile:
                lines.append(f"  - {chunk.text}")
        
        # Calendar context is always relevant for work
        calendar_chunks = reasoning_layer.store.retrieve(
            message,
            top_k=3,
            source_filter=["calendar"]
        )
        if calendar_chunks:
            lines.append("\nCalendar:")
            for chunk, score in calendar_chunks:
                lines.append(f"  {chunk.text}")
        
        lines.append("=== END WORK CONTEXT ===")
        return "\n".join(lines)
    
    def get_domain_prompt_addition(self) -> str:
        """Add work-specific instructions."""
        return (
            "When discussing work matters:\n"
            "- Be professional but friendly\n"
            "- Reference specific projects, deadlines, or meetings when relevant\n"
            "- Help prioritize tasks based on deadlines\n"
            "- Suggest scheduling breaks if workload is heavy"
        )
