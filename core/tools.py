"""
Tool Definitions for Roku AI Agent

Defines the tools available to the model for function calling.
Uses JSON schema format compatible with Llama-3.2-Instruct.
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import re


@dataclass
class Tool:
    """Represents a callable tool."""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Optional[Callable] = None
    
    def to_schema(self) -> Dict[str, Any]:
        """Convert to JSON schema for prompt injection."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }


class ToolRegistry:
    """
    Registry of available tools for the agent.
    
    Tools can be registered and looked up by name.
    """
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self.tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def list_tools(self) -> List[Tool]:
        """Get all registered tools."""
        return list(self.tools.values())
    
    def get_schemas(self) -> List[Dict[str, Any]]:
        """Get JSON schemas for all tools."""
        return [t.to_schema() for t in self.tools.values()]
    
    def format_for_prompt(self) -> str:
        """Format tool definitions for system prompt."""
        lines = ["AVAILABLE TOOLS:"]
        for tool in self.tools.values():
            lines.append(f"\n{tool.name}: {tool.description}")
            lines.append(f"  Parameters: {json.dumps(tool.parameters, indent=2)}")
        return "\n".join(lines)


# =============================================================================
# Tool Definitions
# =============================================================================

def create_calendar_tools() -> List[Tool]:
    """Create calendar-related tools."""
    return [
        Tool(
            name="get_calendar",
            description="Get calendar events for a specific date or date range. Use this when the user asks about their schedule, events, classes, or meetings on a specific day.",
            parameters={
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "The date to check. Can be 'today', 'tomorrow', 'monday', 'tuesday', etc., or a specific date like '2026-02-03'."
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Optional end date for a range query. If not provided, only the single date is checked."
                    }
                },
                "required": ["date"]
            }
        ),
        Tool(
            name="get_next_event",
            description="Get the next upcoming calendar event. Use this when the user asks 'what's next' or 'what do I have coming up'.",
            parameters={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="check_availability",
            description="Check if the user is free at a specific time or on a specific day. Use this when the user asks 'am I free tonight?', 'do I have anything on Saturday?', etc.",
            parameters={
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "The date to check availability for."
                    },
                    "time_of_day": {
                        "type": "string",
                        "enum": ["morning", "afternoon", "evening", "night", "all_day"],
                        "description": "Optional: specific time of day to check."
                    }
                },
                "required": ["date"]
            }
        )
    ]


def create_weather_tools() -> List[Tool]:
    """Create weather-related tools."""
    return [
        Tool(
            name="get_weather",
            description="Get current weather conditions. Use this when the user asks about weather, temperature, or if they should bring an umbrella/jacket.",
            parameters={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Optional city name. Defaults to user's location if not provided."
                    }
                },
                "required": []
            }
        )
    ]


def create_time_tools() -> List[Tool]:
    """Create time-related tools."""
    return [
        Tool(
            name="get_current_time",
            description="Get the current date and time. Use this when the user asks about the time or date.",
            parameters={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]


def create_profile_tools() -> List[Tool]:
    """Create profile-related tools."""
    return [
        Tool(
            name="get_user_info",
            description="Get information about the user from their profile. Use this when answering questions about the user's work, goals, preferences, or personal details.",
            parameters={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["identity", "work", "goals", "schedule", "preferences", "location"],
                        "description": "The category of information to retrieve."
                    }
                },
                "required": ["category"]
            }
        )
    ]


def create_default_registry() -> ToolRegistry:
    """Create a registry with all default tools."""
    registry = ToolRegistry()
    
    for tool in create_calendar_tools():
        registry.register(tool)
    
    for tool in create_weather_tools():
        registry.register(tool)
    
    for tool in create_time_tools():
        registry.register(tool)
    
    for tool in create_profile_tools():
        registry.register(tool)
    
    return registry


# =============================================================================
# Tool Call Parsing
# =============================================================================

@dataclass
class ToolCall:
    """Represents a parsed tool call from model output."""
    name: str
    parameters: Dict[str, Any]
    raw: str = ""


def parse_tool_call(text: str) -> Optional[ToolCall]:
    """
    Parse a tool call from model output.
    
    Llama-3.2-Instruct outputs tool calls as:
    {"name": "tool_name", "parameters": {...}}
    
    Returns None if no valid tool call found.
    """
    # Try to find JSON in the text
    # Look for {"name": pattern
    json_pattern = r'\{[^{}]*"name"[^{}]*"parameters"[^{}]*\{[^{}]*\}[^{}]*\}'
    
    # Also try simpler pattern
    simple_pattern = r'\{"name":\s*"([^"]+)",\s*"parameters":\s*(\{[^}]*\})\}'
    
    # Try the simple pattern first
    match = re.search(simple_pattern, text, re.DOTALL)
    if match:
        try:
            name = match.group(1)
            params = json.loads(match.group(2))
            return ToolCall(name=name, parameters=params, raw=match.group(0))
        except json.JSONDecodeError:
            pass
    
    # Try to find any JSON object with name and parameters
    try:
        # Find all potential JSON objects
        brace_count = 0
        start = None
        for i, char in enumerate(text):
            if char == '{':
                if brace_count == 0:
                    start = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start is not None:
                    candidate = text[start:i+1]
                    try:
                        obj = json.loads(candidate)
                        if "name" in obj and "parameters" in obj:
                            return ToolCall(
                                name=obj["name"],
                                parameters=obj.get("parameters", {}),
                                raw=candidate
                            )
                    except json.JSONDecodeError:
                        continue
                    start = None
    except Exception:
        pass
    
    return None


def parse_date_reference(date_str: str, reference_date: Optional[datetime] = None) -> datetime:
    """
    Parse a natural language date reference into a datetime.
    
    Handles:
    - 'today', 'tomorrow', 'yesterday'
    - Day names: 'monday', 'tuesday', etc.
    - ISO format: '2026-02-03'
    """
    reference = reference_date or datetime.now()
    date_str = date_str.lower().strip()
    
    if date_str == 'today':
        return reference.replace(hour=0, minute=0, second=0, microsecond=0)
    
    if date_str == 'tomorrow':
        return (reference + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    
    if date_str == 'yesterday':
        return (reference - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Day names
    day_names = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    if date_str in day_names:
        target_day = day_names.index(date_str)
        current_day = reference.weekday()
        days_ahead = target_day - current_day
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7
        return (reference + timedelta(days=days_ahead)).replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Try ISO format
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        pass
    
    # Default to today if can't parse
    return reference.replace(hour=0, minute=0, second=0, microsecond=0)


if __name__ == "__main__":
    # Test tool registry
    registry = create_default_registry()
    
    print("Registered tools:")
    for tool in registry.list_tools():
        print(f"  - {tool.name}: {tool.description[:50]}...")
    
    print("\n" + "="*60)
    print("Tool prompt format:")
    print("="*60)
    print(registry.format_for_prompt())
    
    # Test parsing
    print("\n" + "="*60)
    print("Testing tool call parsing:")
    print("="*60)
    
    test_cases = [
        '{"name": "get_calendar", "parameters": {"date": "monday"}}',
        'Let me check your calendar. {"name": "get_calendar", "parameters": {"date": "tomorrow"}}',
        'I\'ll look that up for you.\n{"name": "get_weather", "parameters": {}}',
    ]
    
    for test in test_cases:
        result = parse_tool_call(test)
        print(f"\nInput: {test[:50]}...")
        print(f"Parsed: {result}")
