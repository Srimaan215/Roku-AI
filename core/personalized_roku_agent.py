"""
PersonalizedRoku Agent - Tool-Using AI Assistant

This is the agentic version of Roku that uses function calling (ReAct pattern):
1. Receive user query
2. Model reasons and may call tools
3. Execute tools, inject results
4. Model generates final answer

This replaces the static RAG-CoT approach with dynamic tool use.
"""

import json
import re
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from core.tools import (
    ToolRegistry, ToolCall, Tool,
    create_default_registry, parse_tool_call
)
from core.tool_executor import ToolExecutor, ToolResult
from core.multi_lora import MultiLoRALlama

# Optional integrations
try:
    from core.integrations.calendar_provider import CalendarProvider
    CALENDAR_AVAILABLE = True
except ImportError:
    CALENDAR_AVAILABLE = False

try:
    from core.integrations.ics_provider import ICSProvider
    ICS_AVAILABLE = True
except ImportError:
    ICS_AVAILABLE = False

try:
    from core.integrations.weather_provider import WeatherProvider
    WEATHER_AVAILABLE = True
except ImportError:
    WEATHER_AVAILABLE = False

try:
    from core.integrations.reminders_provider import RemindersProvider
    REMINDERS_AVAILABLE = True
except ImportError:
    REMINDERS_AVAILABLE = False


class PersonalizedRokuAgent:
    """
    Agentic AI assistant with tool-calling capabilities.
    
    Uses the ReAct pattern:
    - Reason about the query
    - Act by calling tools if needed
    - Observe tool results
    - Respond with grounded answer
    """
    
    DEFAULT_PROFILES_DIR = Path.home() / "Roku/roku-ai/data/profiles"
    MAX_TOOL_CALLS = 3  # Prevent infinite loops
    
    # Canvas ICS feed URL (direct from Canvas, always fresh)
    CANVAS_ICS_URL = "https://umamherst.instructure.com/feeds/calendars/user_48pPzatsivhapk5fhxm15bvqETCisgJs1kBqxnaj.ics"
    
    def __init__(
        self,
        username: str,
        model_path: Optional[str] = None,
        enable_calendar: bool = True,
        enable_weather: bool = True,
        enable_reminders: bool = True,
        enable_personality: bool = True,
        verbose: bool = False,
    ):
        self.username = username
        self.verbose = verbose
        
        # Load profile
        self._load_profile()
        
        # Initialize integrations
        self.calendar: Optional[CalendarProvider] = None
        self.ics: Optional[ICSProvider] = None
        self.weather: Optional[WeatherProvider] = None
        self.reminders: Optional[RemindersProvider] = None
        
        if enable_calendar and CALENDAR_AVAILABLE:
            self._init_calendar()
        
        # Always try to init ICS for Canvas
        if ICS_AVAILABLE:
            self._init_ics()
        
        if enable_weather and WEATHER_AVAILABLE:
            self._init_weather()
        
        if enable_reminders and REMINDERS_AVAILABLE:
            self._init_reminders()
        
        # Create tool registry and executor
        self.tools = create_default_registry()
        self.executor = ToolExecutor(
            calendar_provider=self.calendar,
            ics_provider=self.ics,
            weather_provider=self.weather,
            reminders_provider=self.reminders,
            profile=self.profile,
            username=self.username,
        )
        
        # Initialize LLM
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
        """Load user profile."""
        profile_path = self.DEFAULT_PROFILES_DIR / f"{self.username}.json"
        
        if not profile_path.exists():
            raise FileNotFoundError(f"Profile not found: {profile_path}")
        
        with open(profile_path) as f:
            data = json.load(f)
        
        self.profile = data.get('profile', data)
        
        if self.verbose:
            print(f"✓ Profile loaded: {self.username}")
    
    def _init_calendar(self) -> None:
        """Initialize calendar if credentials exist."""
        try:
            self.calendar = CalendarProvider()
            if self.calendar.token_path.exists():
                if self.calendar.authenticate():
                    if self.verbose:
                        print("✓ Calendar connected")
                else:
                    self.calendar = None
            else:
                self.calendar = None
        except Exception as e:
            if self.verbose:
                print(f"Calendar init warning: {e}")
            self.calendar = None
    
    def _init_ics(self) -> None:
        """Initialize ICS provider with Canvas feed."""
        try:
            self.ics = ICSProvider()
            self.ics.add_feed("canvas", self.CANVAS_ICS_URL)
            if self.verbose:
                print("✓ Canvas ICS feed connected")
        except Exception as e:
            if self.verbose:
                print(f"ICS init warning: {e}")
            self.ics = None
    
    def _init_weather(self) -> None:
        """Initialize weather if API key exists."""
        try:
            self.weather = WeatherProvider()
            if self.weather.is_configured():
                if self.verbose:
                    print("✓ Weather connected")
            else:
                if self.verbose:
                    print("⚠ Weather API key not configured")
        except Exception as e:
            if self.verbose:
                print(f"Weather init warning: {e}")
            self.weather = None
    
    def _init_reminders(self) -> None:
        """Initialize Apple Reminders integration."""
        try:
            self.reminders = RemindersProvider()
            if self.verbose:
                print("✓ Reminders connected")
        except Exception as e:
            if self.verbose:
                print(f"Reminders init warning: {e}")
            self.reminders = None
    
    def _build_system_prompt(self) -> str:
        """Build system prompt with tool definitions."""
        now = datetime.now()
        is_weekend = now.weekday() >= 5
        
        # Get current time context
        time_context = (
            f"Current time: {now.strftime('%I:%M %p')} on "
            f"{now.strftime('%A, %B %d, %Y')} "
            f"({'weekend' if is_weekend else 'weekday'})"
        )
        
        # Get user identity
        identity = self.profile.get('identity', {})
        user_name = identity.get('name', self.username)
        
        # Tool definitions
        tool_schemas = self.tools.get_schemas()
        tools_json = json.dumps(tool_schemas, indent=2)
        
        system = f"""You are Roku, a personal AI assistant for {user_name}. You are helpful, warm, and casual.

{time_context}

You have access to the following tools to help answer questions:

{tools_json}

INSTRUCTIONS:
1. When the user asks about their schedule, calendar, events, or classes - USE the get_calendar or check_availability tool.
2. When the user asks about weather - USE the get_weather tool.
3. When the user asks about their personal information, goals, or preferences - USE the get_user_info tool.
4. To call a tool, respond with ONLY a JSON object in this exact format:
   {{"name": "tool_name", "parameters": {{"param1": "value1"}}}}
5. After receiving tool results, provide a helpful, conversational answer.
6. If you don't need any tools, just answer directly.

Be concise and friendly. Use the tools when they would help provide accurate information."""

        return system
    
    def _build_prompt(
        self,
        query: str,
        tool_results: Optional[List[tuple]] = None,
    ) -> str:
        """Build the full prompt including any tool results."""
        system = self._build_system_prompt()
        
        # Start with system and user query
        messages = f"""<|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{query}<|eot_id|>"""
        
        # Add tool results if any
        if tool_results:
            for tool_call, result in tool_results:
                # Add assistant's tool call
                messages += f"""<|start_header_id|>assistant<|end_header_id|>

{{"name": "{tool_call.name}", "parameters": {json.dumps(tool_call.parameters)}}}<|eot_id|>"""
                
                # Add tool result (ipython role)
                messages += f"""<|start_header_id|>ipython<|end_header_id|>

{result.to_context_string()}<|eot_id|>"""
        
        # Add final assistant header for response
        messages += """<|start_header_id|>assistant<|end_header_id|>

"""
        
        return messages
    
    def ask(
        self,
        query: str,
        max_tokens: int = 300,
        temperature: float = 0.7,
    ) -> str:
        """
        Ask a question with agentic tool-calling.
        
        The model may call tools, and results are injected
        before generating the final answer.
        """
        tool_results: List[tuple] = []
        
        for iteration in range(self.MAX_TOOL_CALLS + 1):
            # Build prompt with any previous tool results
            prompt = self._build_prompt(query, tool_results if tool_results else None)
            
            if self.verbose:
                print(f"\n[Iteration {iteration + 1}]")
            
            # Generate response
            response = self.llm.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["<|eot_id|>"]
            )
            
            if self.verbose:
                print(f"Raw response: {response[:200]}...")
            
            # Check if response is a tool call
            tool_call = parse_tool_call(response)
            
            if tool_call and iteration < self.MAX_TOOL_CALLS:
                if self.verbose:
                    print(f"Tool call detected: {tool_call.name}({tool_call.parameters})")
                
                # Execute the tool
                result = self.executor.execute(tool_call)
                
                if self.verbose:
                    print(f"Tool result: {result.to_context_string()[:100]}...")
                
                # Store result and continue
                tool_results.append((tool_call, result))
                continue
            
            # No tool call or max iterations reached - this is the final answer
            # Clean up the response
            final_answer = self._clean_response(response)
            return final_answer
        
        # Fallback
        return "I'm having trouble processing that request. Could you try asking differently?"
    
    def _clean_response(self, response: str) -> str:
        """Clean up the model's response."""
        # Remove any partial tool call attempts
        response = re.sub(r'\{"name":[^}]+\}', '', response)
        
        # Remove common artifacts
        response = response.strip()
        
        # Remove leading "Answer:" or similar
        for prefix in ["Answer:", "Response:", "Here's my answer:"]:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        return response
    
    def quick_ask(self, query: str, max_tokens: int = 200) -> str:
        """Quick question shortcut."""
        return self.ask(query, max_tokens=max_tokens)
    
    def debug_tools(self) -> None:
        """Print available tools and their status."""
        print("Available Tools:")
        for tool in self.tools.list_tools():
            print(f"  - {tool.name}: {tool.description[:60]}...")
        
        print("\nIntegration Status:")
        print(f"  Calendar: {'Connected' if self.calendar and self.calendar.is_authenticated() else 'Not connected'}")
        print(f"  Weather: {'Connected' if self.weather and self.weather.is_configured() else 'Not configured'}")
        print(f"  Profile: {self.username}")


if __name__ == "__main__":
    print("Testing PersonalizedRoku Agent...")
    print("=" * 60)
    
    agent = PersonalizedRokuAgent(username="Srimaan", verbose=True)
    
    agent.debug_tools()
    
    print("\n" + "=" * 60)
    print("TESTING QUERIES")
    print("=" * 60)
    
    tests = [
        "What classes do I have on Monday?",
        "Am I free tomorrow evening?",
        "What time is it?",
    ]
    
    for q in tests:
        print(f"\nQ: {q}")
        print("-" * 40)
        response = agent.ask(q)
        print(f"A: {response}")
