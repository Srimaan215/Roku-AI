"""
Base domain adapter interface for Roku AI.

Domain adapters provide:
- Domain-specific context preparation
- Response post-processing
- Domain keyword detection
- Custom prompt formatting
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class DomainAdapter(ABC):
    """Base class for domain-specific adapters."""
    
    def __init__(self, name: str, keywords: List[str]):
        """
        Initialize domain adapter.
        
        Args:
            name: Domain name (e.g., "work", "home", "health")
            keywords: Keywords that indicate this domain
        """
        self.name = name
        self.keywords = keywords
    
    def matches(self, message: str) -> bool:
        """
        Check if message matches this domain.
        
        Args:
            message: User's message
            
        Returns:
            True if message is relevant to this domain
        """
        message_lower = message.lower()
        return any(kw.lower() in message_lower for kw in self.keywords)
    
    @abstractmethod
    def prepare_context(
        self,
        message: str,
        context: Dict[str, Any],
        reasoning_layer: Any,  # ReasoningLayer
    ) -> str:
        """
        Prepare domain-specific context for the prompt.
        
        Args:
            message: User's message
            context: General context dict
            reasoning_layer: ReasoningLayer instance for retrieval
            
        Returns:
            Formatted context string to inject into prompt
        """
        pass
    
    def post_process(self, response: str, context: Dict[str, Any]) -> str:
        """
        Post-process the model's response.
        
        Override this to add domain-specific formatting, disclaimers, etc.
        
        Args:
            response: Model's raw response
            context: Context dict
            
        Returns:
            Processed response
        """
        return response
    
    def get_domain_prompt_addition(self) -> str:
        """
        Get additional system prompt text for this domain.
        
        Override to add domain-specific instructions.
        
        Returns:
            Additional prompt text
        """
        return ""
