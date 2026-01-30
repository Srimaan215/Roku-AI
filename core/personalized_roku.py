"""
Personalized Roku - Combines multi-LoRA with context injection.

Architecture:
- Facts → System prompt injection (zero hallucination)
- Style → Personality adapter (trained behavior)
- The personal adapter is now optional/deprecated for facts
"""

from typing import Optional, List, Dict
from pathlib import Path

from .multi_lora import MultiLoRALlama
from .context_manager import ContextManager


class PersonalizedRoku:
    """
    Roku AI with personalization via context injection.
    
    Facts are injected via system prompt (accurate, no training needed).
    Personality comes from the personality adapter (trained style).
    """
    
    def __init__(
        self,
        username: Optional[str] = None,
        use_personality_adapter: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize personalized Roku.
        
        Args:
            username: User to load profile for (None = generic mode)
            use_personality_adapter: Load Roku personality adapter
            verbose: Debug output
        """
        self.verbose = verbose
        self.username = username
        
        # Initialize context manager
        self.context = ContextManager()
        if username:
            if not self.context.load_profile(username):
                print(f"Warning: Profile for '{username}' not found. Running in generic mode.")
                self.username = None
        
        # Initialize LLM with adapters
        self.llm = MultiLoRALlama(verbose=verbose)
        
        if use_personality_adapter:
            personality_path = Path.home() / "Roku/roku-ai/models/adapters/personality.gguf"
            if personality_path.exists():
                self.llm.add_adapter("personality", str(personality_path), scale=1.0)
                if verbose:
                    print("✓ Loaded personality adapter")
            else:
                print("Warning: Personality adapter not found")
        
        if verbose and username:
            tokens = self.context.get_context_tokens_estimate()
            print(f"✓ Loaded profile for {username} (~{tokens} tokens)")
    
    def chat(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 256,
    ) -> str:
        """
        Chat with personalized context.
        
        Args:
            message: User message
            history: Previous conversation (optional)
            max_tokens: Max response length
            
        Returns:
            Assistant response
        """
        # Build messages with system prompt
        system_prompt = self.context.build_system_prompt()
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add history if provided
        if history:
            messages.extend(history)
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        # Generate response
        response = self.llm.chat(messages, max_tokens=max_tokens)
        
        return response
    
    def quick_ask(self, question: str, max_tokens: int = 150) -> str:
        """Quick single-turn question with full context."""
        return self.chat(question, max_tokens=max_tokens)
    
    @property
    def profile_summary(self) -> str:
        """Get the current profile summary."""
        return self.context.get_profile_summary()


def test_personalized_roku():
    """Test the personalized Roku."""
    print("="*60)
    print("Testing Personalized Roku with Context Injection")
    print("="*60)
    
    roku = PersonalizedRoku(username="Srimaan", verbose=True)
    
    print("\n" + "-"*60)
    print("TESTING FACTUAL RECALL (should be accurate)")
    print("-"*60)
    
    test_questions = [
        "What's my name?",
        "What am I studying?",
        "What universities do I work at?",
        "What should you remind me about?",
        "What's my current research project about?",
        "What time do I usually wake up?",
    ]
    
    for q in test_questions:
        print(f"\nQ: {q}")
        response = roku.quick_ask(q, max_tokens=100)
        print(f"A: {response}")


if __name__ == "__main__":
    test_personalized_roku()
