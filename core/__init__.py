"""
Roku Core Module
================
Core functionality for the Roku AI assistant.
"""

from .llm import LocalLLM
from .context import ContextManager
from .router import QueryRouter

# Optional: Voice interface (has heavy dependencies)
try:
    from .voice import VoiceInterface
except (ImportError, OSError):
    VoiceInterface = None

__all__ = ["LocalLLM", "VoiceInterface", "ContextManager", "QueryRouter"]
