"""
PersonalizedRoku with RAG-CoT

Combines:
- Vector retrieval for relevant context
- Chain-of-Thought prompting for explicit reasoning
- Multi-LoRA for personality/style

Architecture:
1. User query → Embed → Retrieve relevant chunks
2. Build CoT prompt with retrieved context
3. Model reasons through context → Answer
"""

import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from core.reasoning import ReasoningLayer
from core.multi_lora import MultiLoRALlama

# Optional integrations
try:
    from core.integrations.calendar_provider import CalendarProvider
    CALENDAR_AVAILABLE = True
except ImportError:
    CALENDAR_AVAILABLE = False

try:
    from core.integrations.weather_provider import WeatherProvider
    WEATHER_AVAILABLE = True
except ImportError:
    WEATHER_AVAILABLE = False


class PersonalizedRokuCoT:
    """
    RAG-CoT enhanced Roku AI.
    
    Uses vector retrieval + chain-of-thought for intelligent reasoning.
    """
    
    DEFAULT_PROFILES_DIR = Path.home() / "Roku/roku-ai/data/profiles"
    
    def __init__(
        self,
        username: str,
        model_path: Optional[str] = None,
        enable_calendar: bool = True,
        enable_weather: bool = True,
        enable_personality: bool = True,
        verbose: bool = False,
    ):
        self.username = username
        self.verbose = verbose
        
        # Initialize reasoning layer
        if self.verbose:
            print("Initializing reasoning layer...")
        self.reasoning = ReasoningLayer()
        
        # Load profile
        self._load_profile()
        
        # Initialize calendar
        self.calendar: Optional[CalendarProvider] = None
        if enable_calendar and CALENDAR_AVAILABLE:
            self._init_calendar()
        
        # Initialize weather
        self.weather: Optional[WeatherProvider] = None
        if enable_weather and WEATHER_AVAILABLE:
            self._init_weather()
        
        # Initialize LLM with optional personality adapter
        if self.verbose:
            print("Loading LLM...")
        self.llm = MultiLoRALlama(
            model_path=model_path,
            verbose=verbose
        )
        
        if enable_personality:
            personality_path = Path.home() / "Roku/roku-ai/models/adapters/personality.gguf"
            if personality_path.exists():
                self.llm.add_adapter("personality", scale=0.8)
                if self.verbose:
                    print("✓ Personality adapter loaded")
    
    def _load_profile(self) -> None:
        """Load user profile into reasoning layer."""
        profile_path = self.DEFAULT_PROFILES_DIR / f"{self.username}.json"
        
        if not profile_path.exists():
            raise FileNotFoundError(f"Profile not found: {profile_path}")
        
        with open(profile_path) as f:
            data = json.load(f)
        
        self.profile = data.get('profile', data)
        self.reasoning.load_profile_chunks(self.profile, self.username)
        
        if self.verbose:
            print(f"✓ Profile loaded: {self.username}")
    
    def _init_calendar(self) -> None:
        """Initialize calendar if credentials exist."""
        try:
            self.calendar = CalendarProvider()
            if self.calendar.token_path.exists():
                if self.calendar.authenticate():
                    self._refresh_calendar_context()
                    if self.verbose:
                        print("✓ Calendar connected")
        except Exception as e:
            if self.verbose:
                print(f"Calendar init warning: {e}")
    
    def _init_weather(self) -> None:
        """Initialize weather if API key exists."""
        try:
            self.weather = WeatherProvider()
            if self.weather.is_configured():
                self._refresh_weather_context()
                if self.verbose:
                    print("✓ Weather connected")
            elif self.verbose:
                print("⚠ Weather API key not configured")
        except Exception as e:
            if self.verbose:
                print(f"Weather init warning: {e}")
    
    def _refresh_calendar_context(self) -> None:
        """Update calendar context in reasoning layer."""
        if self.calendar and self.calendar.is_authenticated():
            context = self.calendar.get_calendar_context()
            self.reasoning.update_calendar_context(context)
    
    def _refresh_weather_context(self) -> None:
        """Update weather context in reasoning layer."""
        if self.weather and self.weather.is_configured():
            context = self.weather.get_weather_context()
            self.reasoning.update_weather_context(context)
    
    def ask(
        self,
        query: str,
        max_tokens: int = 200,
        temperature: float = 0.7,
        show_reasoning: bool = False,
    ) -> str:
        """
        Ask a question with RAG-CoT reasoning.
        
        Args:
            query: User's question
            max_tokens: Maximum response length
            temperature: Response creativity
            show_reasoning: If True, include reasoning trace in output
            
        Returns:
            Model's response (optionally with reasoning trace)
        """
        # Refresh live context
        if self.calendar:
            self._refresh_calendar_context()
        if self.weather:
            self._refresh_weather_context()
        
        # Build CoT prompt with retrieved context
        prompt = self.reasoning.build_cot_prompt(
            query=query,
            username=self.username,
            include_reasoning_hint=True
        )
        
        if self.verbose:
            print(f"Retrieved from: {self.reasoning.get_retrieved_sources()}")
        
        # Generate response
        response = self.llm.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|eot_id|>"]
        )
        
        if show_reasoning:
            return response
        else:
            # Extract just the answer (after reasoning)
            # Look for common answer patterns
            answer = response
            for marker in ["\n\nAnswer:", "\n\nSo,", "\n\nTherefore,", "\n\nYes,", "\n\nNo,"]:
                if marker in response:
                    answer = response.split(marker, 1)[1] if marker != "\n\n" else response
                    if not marker.endswith(","):
                        break
            
            return answer.strip()
    
    def quick_ask(
        self,
        query: str,
        max_tokens: int = 150,
    ) -> str:
        """Quick question without reasoning trace."""
        return self.ask(query, max_tokens=max_tokens, show_reasoning=False)
    
    def debug_retrieval(self, query: str) -> None:
        """Show what context would be retrieved for a query."""
        self.reasoning.update_time_context()
        if self.calendar:
            self._refresh_calendar_context()
        
        print(f"Query: {query}")
        print("-" * 60)
        context = self.reasoning.retrieve_context(query)
        print(context)
        print(f"\nSources: {self.reasoning.get_retrieved_sources()}")


if __name__ == "__main__":
    print("Testing PersonalizedRoku with RAG-CoT...")
    print("=" * 60)
    
    roku = PersonalizedRokuCoT(username="Srimaan", verbose=True)
    
    print("\n" + "=" * 60)
    print("TESTING QUERIES")
    print("=" * 60)
    
    tests = [
        "Am I free tonight?",
        "What should I focus on this weekend?",
        "Where do I go to work out?",
    ]
    
    for q in tests:
        print(f"\nQ: {q}")
        print("-" * 40)
        response = roku.ask(q, show_reasoning=True)
        print(f"Full response:\n{response}")
