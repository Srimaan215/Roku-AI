"""
Tool Executor for Roku AI Agent

Executes tool calls and returns results.
Bridges between the model's tool calls and actual integrations.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

from core.tools import ToolCall, ToolRegistry, parse_date_reference


@dataclass
class ToolResult:
    """Result from executing a tool."""
    success: bool
    data: Any
    error: Optional[str] = None
    
    def to_context_string(self) -> str:
        """Format result for injection back into prompt."""
        if not self.success:
            return f"[Tool Error: {self.error}]"
        
        if isinstance(self.data, str):
            return self.data
        elif isinstance(self.data, dict):
            return json.dumps(self.data, indent=2, default=str)
        elif isinstance(self.data, list):
            return "\n".join(str(item) for item in self.data)
        else:
            return str(self.data)


class ToolExecutor:
    """
    Executes tool calls using available integrations.
    
    Manages connections to:
    - Calendar (Google Calendar)
    - Weather (OpenWeatherMap)
    - User profile
    - Time
    """
    
    def __init__(
        self,
        calendar_provider=None,
        weather_provider=None,
        profile: Optional[Dict[str, Any]] = None,
        username: str = "User",
    ):
        self.calendar = calendar_provider
        self.weather = weather_provider
        self.profile = profile or {}
        self.username = username
        
        # Map tool names to executor methods
        self._executors = {
            "get_calendar": self._exec_get_calendar,
            "get_next_event": self._exec_get_next_event,
            "check_availability": self._exec_check_availability,
            "get_weather": self._exec_get_weather,
            "get_current_time": self._exec_get_current_time,
            "get_user_info": self._exec_get_user_info,
        }
    
    def execute(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute a tool call and return the result.
        """
        executor = self._executors.get(tool_call.name)
        
        if not executor:
            return ToolResult(
                success=False,
                data=None,
                error=f"Unknown tool: {tool_call.name}"
            )
        
        try:
            return executor(tool_call.parameters)
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=str(e)
            )
    
    # =========================================================================
    # Calendar Tools
    # =========================================================================
    
    def _exec_get_calendar(self, params: Dict[str, Any]) -> ToolResult:
        """Get calendar events for a date or range."""
        if not self.calendar:
            return ToolResult(False, None, "Calendar not connected")
        
        if not self.calendar.is_authenticated():
            return ToolResult(False, None, "Calendar not authenticated")
        
        date_str = params.get("date", "today")
        target_date = parse_date_reference(date_str)
        
        # Set time range for the target date
        start = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = target_date.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        # Handle end_date for range queries
        if "end_date" in params and params["end_date"]:
            end_date = parse_date_reference(params["end_date"])
            end = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        # Query all calendars
        try:
            service = self.calendar.service
            calendars = service.calendarList().list().execute().get('items', [])
            
            all_events = []
            for cal in calendars:
                cal_id = cal.get('id')
                events = self.calendar.get_events(
                    start_date=start,
                    end_date=end,
                    calendar_id=cal_id
                )
                all_events.extend(events)
            
            # Sort by start time
            all_events.sort(key=lambda e: e.start_time)
            
            if not all_events:
                date_display = target_date.strftime("%A, %B %d")
                return ToolResult(
                    True,
                    f"No events found for {date_display}. You are free that day."
                )
            
            # Format events
            lines = [f"Events for {target_date.strftime('%A, %B %d, %Y')}:"]
            for event in all_events:
                lines.append(f"  - {event.format_time_range()}: {event.title}")
            
            return ToolResult(True, "\n".join(lines))
            
        except Exception as e:
            return ToolResult(False, None, f"Calendar error: {e}")
    
    def _exec_get_next_event(self, params: Dict[str, Any]) -> ToolResult:
        """Get the next upcoming event."""
        if not self.calendar or not self.calendar.is_authenticated():
            return ToolResult(False, None, "Calendar not connected")
        
        next_event = self.calendar.get_next_event()
        
        if not next_event:
            return ToolResult(True, "No upcoming events in the next 24 hours.")
        
        time_until = next_event.time_until()
        hours = int(time_until.total_seconds() // 3600)
        mins = int((time_until.total_seconds() % 3600) // 60)
        
        if hours > 0:
            time_str = f"{hours} hours and {mins} minutes"
        else:
            time_str = f"{mins} minutes"
        
        return ToolResult(
            True,
            f"Next event: '{next_event.title}' in {time_str} ({next_event.format_time_range()})"
        )
    
    def _exec_check_availability(self, params: Dict[str, Any]) -> ToolResult:
        """Check if user is free at a specific time."""
        if not self.calendar or not self.calendar.is_authenticated():
            return ToolResult(False, None, "Calendar not connected")
        
        date_str = params.get("date", "today")
        target_date = parse_date_reference(date_str)
        time_of_day = params.get("time_of_day", "all_day")
        
        # Define time ranges
        time_ranges = {
            "morning": (6, 12),
            "afternoon": (12, 17),
            "evening": (17, 21),
            "night": (21, 24),
            "all_day": (0, 24)
        }
        
        start_hour, end_hour = time_ranges.get(time_of_day, (0, 24))
        start = target_date.replace(hour=start_hour, minute=0, second=0, microsecond=0)
        end = target_date.replace(hour=end_hour - 1, minute=59, second=59, microsecond=999999)
        
        # Query events
        try:
            service = self.calendar.service
            calendars = service.calendarList().list().execute().get('items', [])
            
            all_events = []
            for cal in calendars:
                events = self.calendar.get_events(
                    start_date=start,
                    end_date=end,
                    calendar_id=cal.get('id')
                )
                all_events.extend(events)
            
            date_display = target_date.strftime("%A, %B %d")
            
            if not all_events:
                if time_of_day == "all_day":
                    return ToolResult(True, f"You are FREE all day on {date_display}. No events scheduled.")
                else:
                    return ToolResult(True, f"You are FREE on {date_display} {time_of_day}. No events during that time.")
            
            # Has events
            lines = [f"You have {len(all_events)} event(s) on {date_display} {time_of_day}:"]
            for event in sorted(all_events, key=lambda e: e.start_time):
                lines.append(f"  - {event.format_time_range()}: {event.title}")
            
            return ToolResult(True, "\n".join(lines))
            
        except Exception as e:
            return ToolResult(False, None, f"Calendar error: {e}")
    
    # =========================================================================
    # Weather Tools
    # =========================================================================
    
    def _exec_get_weather(self, params: Dict[str, Any]) -> ToolResult:
        """Get current weather conditions."""
        if not self.weather:
            return ToolResult(False, None, "Weather not configured")
        
        if not self.weather.is_configured():
            return ToolResult(False, None, "Weather API key not set")
        
        city = params.get("city")
        weather_data = self.weather.get_current_weather(city)
        
        if not weather_data:
            return ToolResult(False, None, "Could not fetch weather data")
        
        result = weather_data.format_context()
        suggestions = weather_data.get_activity_suggestions()
        if suggestions:
            result += f"\n{suggestions}"
        
        return ToolResult(True, result)
    
    # =========================================================================
    # Time Tools
    # =========================================================================
    
    def _exec_get_current_time(self, params: Dict[str, Any]) -> ToolResult:
        """Get current date and time."""
        now = datetime.now()
        
        is_weekend = now.weekday() >= 5
        day_type = "weekend" if is_weekend else "weekday"
        
        # Time of day
        hour = now.hour
        if 6 <= hour < 12:
            period = "morning"
        elif 12 <= hour < 17:
            period = "afternoon"
        elif 17 <= hour < 21:
            period = "evening"
        else:
            period = "night"
        
        return ToolResult(
            True,
            f"Current time: {now.strftime('%I:%M %p')} on {now.strftime('%A, %B %d, %Y')} ({day_type}, {period})"
        )
    
    # =========================================================================
    # Profile Tools
    # =========================================================================
    
    def _exec_get_user_info(self, params: Dict[str, Any]) -> ToolResult:
        """Get user information from profile."""
        category = params.get("category", "identity")
        
        if not self.profile:
            return ToolResult(False, None, "No user profile available")
        
        if category not in self.profile:
            available = ", ".join(self.profile.keys())
            return ToolResult(
                False, None,
                f"Category '{category}' not found. Available: {available}"
            )
        
        data = self.profile[category]
        
        # Format based on category
        if isinstance(data, dict):
            lines = [f"{self.username}'s {category}:"]
            for key, value in data.items():
                lines.append(f"  {key}: {value}")
            return ToolResult(True, "\n".join(lines))
        else:
            return ToolResult(True, f"{self.username}'s {category}: {data}")


if __name__ == "__main__":
    # Test executor
    print("Testing Tool Executor...")
    
    # Create executor without integrations
    executor = ToolExecutor(
        profile={
            "identity": {"name": "Test User", "description": "A test user"},
            "work": {"role": "Developer", "company": "Test Corp"}
        },
        username="TestUser"
    )
    
    # Test time tool
    from core.tools import ToolCall
    
    time_call = ToolCall(name="get_current_time", parameters={})
    result = executor.execute(time_call)
    print(f"\nget_current_time result:")
    print(f"  Success: {result.success}")
    print(f"  Data: {result.data}")
    
    # Test profile tool
    profile_call = ToolCall(name="get_user_info", parameters={"category": "work"})
    result = executor.execute(profile_call)
    print(f"\nget_user_info(work) result:")
    print(f"  Success: {result.success}")
    print(f"  Data: {result.data}")
    
    # Test unknown tool
    unknown_call = ToolCall(name="unknown_tool", parameters={})
    result = executor.execute(unknown_call)
    print(f"\nunknown_tool result:")
    print(f"  Success: {result.success}")
    print(f"  Error: {result.error}")
