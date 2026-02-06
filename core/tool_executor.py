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
    - ICS feeds (Canvas, iCloud)
    - Weather (OpenWeatherMap)
    - User profile
    - Time
    """
    
    def __init__(
        self,
        calendar_provider=None,
        ics_provider=None,
        weather_provider=None,
        profile: Optional[Dict[str, Any]] = None,
        username: str = "User",
        reminders_provider=None,
    ):
        self.calendar = calendar_provider
        self.ics = ics_provider
        self.weather = weather_provider
        self.reminders = reminders_provider
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
            "get_reminders": self._exec_get_reminders,
            "create_reminder": self._exec_create_reminder,
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
        """Get calendar events for a date or range from all sources."""
        has_calendar = self.calendar and self.calendar.is_authenticated()
        has_ics = self.ics and self.ics.feeds
        
        if not has_calendar and not has_ics:
            return ToolResult(False, None, "No calendar sources connected")
        
        date_str = params.get("date", "today")
        start_date, end_date = parse_date_reference(date_str)
        
        # Set time range
        start = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        if end_date:
            # Range query (e.g., "this week")
            end = end_date
            is_range = True
        else:
            # Single day query
            end = start_date.replace(hour=23, minute=59, second=59, microsecond=999999)
            is_range = False
        
        # Handle explicit end_date parameter
        if "end_date" in params and params["end_date"]:
            explicit_end, _ = parse_date_reference(params["end_date"])
            end = explicit_end.replace(hour=23, minute=59, second=59, microsecond=999999)
            is_range = True
        
        all_events = []
        seen_titles = set()  # Dedupe events that appear in multiple sources
        
        # Query Google Calendar
        if has_calendar:
            try:
                service = self.calendar.service
                calendars = service.calendarList().list().execute().get('items', [])
                
                for cal in calendars:
                    cal_id = cal.get('id')
                    events = self.calendar.get_events(
                        start_date=start,
                        end_date=end,
                        calendar_id=cal_id
                    )
                    for event in events:
                        key = (event.title, event.start_time.date())
                        if key not in seen_titles:
                            seen_titles.add(key)
                            all_events.append(event)
            except Exception as e:
                print(f"Google Calendar error: {e}")
        
        # Query ICS feeds (Canvas, etc.)
        if has_ics:
            try:
                ics_events = self.ics.get_events(start, end)
                for event in ics_events:
                    key = (event.title, event.start_time.date())
                    if key not in seen_titles:
                        seen_titles.add(key)
                        all_events.append(event)
            except Exception as e:
                print(f"ICS feed error: {e}")
        
        # Sort by start time
        all_events.sort(key=lambda e: e.start_time)
        
        if not all_events:
            if is_range:
                return ToolResult(
                    True,
                    f"No events found from {start.strftime('%b %d')} to {end.strftime('%b %d')}. You are free!"
                )
            else:
                date_display = start_date.strftime("%A, %B %d")
                return ToolResult(
                    True,
                    f"No events found for {date_display}. You are free that day."
                )
        
        # Format events - group by day for range queries
        if is_range:
            lines = [f"Events from {start.strftime('%b %d')} to {end.strftime('%b %d')}:"]
            current_day = None
            for event in all_events:
                event_day = event.start_time.strftime("%A, %b %d")
                if event_day != current_day:
                    lines.append(f"\n{event_day}:")
                    current_day = event_day
                lines.append(f"  - {event.format_time_range()}: {event.title}")
        else:
            lines = [f"Events for {start_date.strftime('%A, %B %d, %Y')}:"]
            for event in all_events:
                lines.append(f"  - {event.format_time_range()}: {event.title}")
        
        return ToolResult(True, "\n".join(lines))
    
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
        start_date, end_date = parse_date_reference(date_str)
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
        start = start_date.replace(hour=start_hour, minute=0, second=0, microsecond=0)
        
        # Use end_date for range queries, otherwise single day
        if end_date:
            end = end_date.replace(hour=end_hour - 1, minute=59, second=59, microsecond=999999)
            is_range = True
        else:
            end = start_date.replace(hour=end_hour - 1, minute=59, second=59, microsecond=999999)
            is_range = False
        
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
            
            if is_range:
                date_display = f"{start_date.strftime('%b %d')} to {end_date.strftime('%b %d')}"
            else:
                date_display = start_date.strftime("%A, %B %d")
            
            if not all_events:
                if time_of_day == "all_day":
                    return ToolResult(True, f"You are FREE on {date_display}. No events scheduled.")
                else:
                    return ToolResult(True, f"You are FREE on {date_display} {time_of_day}. No events during that time.")
            
            # Has events
            lines = [f"You have {len(all_events)} event(s) on {date_display}:"]
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
    
    # =========================================================================
    # Reminder Tools
    # =========================================================================
    
    def _exec_get_reminders(self, params: Dict[str, Any]) -> ToolResult:
        """Get reminders from Task Master list."""
        if not self.reminders:
            return ToolResult(False, None, "Reminders not connected")
        
        # Always use Task Master
        list_name = "Task Master"
        due_soon = params.get("due_soon", False)
        include_overdue = params.get("include_overdue", True)
        
        try:
            if due_soon:
                reminders = self.reminders.get_due_soon(hours=24)
                header = "Reminders due in the next 24 hours"
            else:
                reminders = self.reminders.get_reminders(
                    list_name=list_name,
                    include_completed=False
                )
                header = "Your reminders in Task Master"
            
            # Add overdue if requested
            if include_overdue:
                overdue = self.reminders.get_overdue()
                if overdue:
                    # Dedupe
                    seen_ids = {r.id for r in reminders}
                    for r in overdue:
                        if r.id not in seen_ids:
                            reminders.insert(0, r)
            
            if not reminders:
                return ToolResult(True, f"No {header.lower()} found. All caught up!")
            
            lines = [f"{header}:"]
            for r in reminders:
                status = "⚠️ OVERDUE" if r.is_overdue() else ""
                due = r.format_due() if r.due_date else ""
                line = f"  - {r.name}"
                if due:
                    line += f" ({due})"
                if status:
                    line += f" {status}"
                if r.list_name != "Reminders":
                    line += f" [{r.list_name}]"
                lines.append(line)
            
            return ToolResult(True, "\n".join(lines))
            
        except Exception as e:
            return ToolResult(False, None, f"Reminders error: {e}")
    
    def _exec_create_reminder(self, params: Dict[str, Any]) -> ToolResult:
        """Create a new reminder in Task Master."""
        if not self.reminders:
            return ToolResult(False, None, "Reminders not connected")
        
        name = params.get("name")
        if not name:
            return ToolResult(False, None, "Reminder name is required")
        
        due_date_str = params.get("due_date")
        due_time_str = params.get("due_time")
        # Always use Task Master - ignore any list_name from model
        list_name = "Task Master"
        notes = params.get("notes")
        
        # Parse due date/time
        due_datetime = None
        if due_date_str:
            from core.integrations.reminders_provider import parse_reminder_datetime
            due_datetime = parse_reminder_datetime(due_date_str, due_time_str)
        
        try:
            success = self.reminders.create_reminder(
                name=name,
                due_date=due_datetime,
                list_name=list_name,
                body=notes
            )
            
            if success:
                response = f"✓ Reminder created: '{name}'"
                if due_datetime:
                    response += f" (due {due_datetime.strftime('%A, %b %d at %I:%M %p')})"
                if list_name != "Reminders":
                    response += f" in '{list_name}'"
                return ToolResult(True, response)
            else:
                return ToolResult(False, None, "Failed to create reminder")
                
        except Exception as e:
            return ToolResult(False, None, f"Error creating reminder: {e}")


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
