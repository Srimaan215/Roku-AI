"""
Google Calendar Integration for Roku AI

Provides real-time calendar awareness:
- Today's events with times
- Time until next event
- Tomorrow's preview (for evening context)

Requires:
- pip install google-auth-oauthlib google-api-python-client
- credentials.json from Google Cloud Console
"""

import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

# Google API imports
try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False


# If modifying these scopes, delete token.pickle
SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']


@dataclass
class CalendarEvent:
    """Represents a calendar event."""
    title: str
    start_time: datetime
    end_time: datetime
    location: Optional[str] = None
    description: Optional[str] = None
    is_all_day: bool = False
    
    @property
    def duration_minutes(self) -> int:
        """Get event duration in minutes."""
        return int((self.end_time - self.start_time).total_seconds() / 60)
    
    def time_until(self, from_time: Optional[datetime] = None) -> timedelta:
        """Get time until this event starts."""
        from_time = from_time or datetime.now()
        return self.start_time - from_time
    
    def is_happening_now(self, at_time: Optional[datetime] = None) -> bool:
        """Check if event is currently happening."""
        at_time = at_time or datetime.now()
        return self.start_time <= at_time <= self.end_time
    
    def format_time_range(self) -> str:
        """Format the time range for display."""
        if self.is_all_day:
            return "All day"
        start = self.start_time.strftime("%I:%M %p").lstrip("0")
        end = self.end_time.strftime("%I:%M %p").lstrip("0")
        return f"{start} - {end}"
    
    def to_context_string(self) -> str:
        """Format event for injection into system prompt."""
        time_str = self.format_time_range()
        location_str = f" @ {self.location}" if self.location else ""
        return f"  - {time_str}: {self.title}{location_str}"


class CalendarProvider:
    """
    Provides calendar data for context injection.
    
    Usage:
        provider = CalendarProvider()
        if provider.authenticate():
            events = provider.get_todays_events()
            context = provider.get_calendar_context()
    """
    
    DEFAULT_CREDENTIALS_DIR = Path.home() / "Roku/roku-ai/config/credentials"
    
    def __init__(
        self,
        credentials_path: Optional[str] = None,
        token_path: Optional[str] = None,
    ):
        """
        Initialize calendar provider.
        
        Args:
            credentials_path: Path to credentials.json from Google Cloud
            token_path: Path to store OAuth token (auto-generated after first auth)
        """
        if not GOOGLE_API_AVAILABLE:
            raise ImportError(
                "Google API libraries not installed. Run:\n"
                "pip install google-auth-oauthlib google-api-python-client"
            )
        
        self.credentials_dir = self.DEFAULT_CREDENTIALS_DIR
        self.credentials_dir.mkdir(parents=True, exist_ok=True)
        
        self.credentials_path = Path(credentials_path) if credentials_path else \
                               self.credentials_dir / "google_calendar_credentials.json"
        self.token_path = Path(token_path) if token_path else \
                         self.credentials_dir / "google_calendar_token.pickle"
        
        self.service = None
        self._credentials = None
    
    def authenticate(self, force_refresh: bool = False) -> bool:
        """
        Authenticate with Google Calendar API.
        
        On first run, opens browser for OAuth consent.
        After that, uses stored token (auto-refreshes if expired).
        
        Returns:
            True if authenticated successfully
        """
        creds = None
        
        # Load existing token
        if self.token_path.exists() and not force_refresh:
            with open(self.token_path, 'rb') as token:
                creds = pickle.load(token)
        
        # Refresh or get new credentials
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    print(f"Token refresh failed: {e}")
                    creds = None
            
            if not creds:
                if not self.credentials_path.exists():
                    print(f"❌ Credentials file not found: {self.credentials_path}")
                    print("\nTo set up Google Calendar:")
                    print("1. Go to https://console.cloud.google.com/")
                    print("2. Create a project and enable Google Calendar API")
                    print("3. Create OAuth 2.0 credentials (Desktop app)")
                    print(f"4. Download and save as: {self.credentials_path}")
                    return False
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self.credentials_path), SCOPES
                )
                creds = flow.run_local_server(port=0)
            
            # Save token for future use
            with open(self.token_path, 'wb') as token:
                pickle.dump(creds, token)
        
        self._credentials = creds
        self.service = build('calendar', 'v3', credentials=creds)
        return True
    
    def is_authenticated(self) -> bool:
        """Check if currently authenticated."""
        return self.service is not None
    
    def get_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_results: int = 20,
        calendar_id: str = 'primary',
    ) -> List[CalendarEvent]:
        """
        Fetch events from Google Calendar.
        
        Args:
            start_date: Start of time range (default: now)
            end_date: End of time range (default: end of today)
            max_results: Maximum events to fetch
            calendar_id: Which calendar to query (default: primary)
            
        Returns:
            List of CalendarEvent objects
        """
        if not self.is_authenticated():
            raise RuntimeError("Not authenticated. Call authenticate() first.")
        
        # Default to today
        now = datetime.now()
        if start_date is None:
            start_date = now
        if end_date is None:
            end_date = now.replace(hour=23, minute=59, second=59)
        
        # Convert to RFC3339 format
        time_min = start_date.isoformat() + 'Z' if start_date.tzinfo is None else start_date.isoformat()
        time_max = end_date.isoformat() + 'Z' if end_date.tzinfo is None else end_date.isoformat()
        
        try:
            events_result = self.service.events().list(
                calendarId=calendar_id,
                timeMin=time_min,
                timeMax=time_max,
                maxResults=max_results,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            return [self._parse_event(e) for e in events]
            
        except HttpError as e:
            print(f"Calendar API error: {e}")
            return []
    
    def get_todays_events(self) -> List[CalendarEvent]:
        """Get all events for today."""
        now = datetime.now()
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = now.replace(hour=23, minute=59, second=59, microsecond=999999)
        return self.get_events(start_date=start, end_date=end)
    
    def get_upcoming_events(self, hours: int = 4) -> List[CalendarEvent]:
        """Get events in the next N hours."""
        now = datetime.now()
        end = now + timedelta(hours=hours)
        return self.get_events(start_date=now, end_date=end)
    
    def get_tomorrows_events(self) -> List[CalendarEvent]:
        """Get all events for tomorrow."""
        tomorrow = datetime.now() + timedelta(days=1)
        start = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)
        end = tomorrow.replace(hour=23, minute=59, second=59, microsecond=999999)
        return self.get_events(start_date=start, end_date=end)
    
    def get_next_event(self) -> Optional[CalendarEvent]:
        """Get the next upcoming event."""
        upcoming = self.get_upcoming_events(hours=24)
        now = datetime.now()
        for event in upcoming:
            if event.start_time > now:
                return event
        return None
    
    def get_current_event(self) -> Optional[CalendarEvent]:
        """Get event happening right now, if any."""
        events = self.get_todays_events()
        now = datetime.now()
        for event in events:
            if event.is_happening_now(now):
                return event
        return None
    
    def _parse_event(self, event_data: Dict[str, Any]) -> CalendarEvent:
        """Parse Google Calendar event data into CalendarEvent."""
        start = event_data.get('start', {})
        end = event_data.get('end', {})
        
        # Handle all-day vs timed events
        if 'dateTime' in start:
            start_time = datetime.fromisoformat(start['dateTime'].replace('Z', '+00:00'))
            end_time = datetime.fromisoformat(end['dateTime'].replace('Z', '+00:00'))
            is_all_day = False
            # Convert to local timezone (naive datetime)
            start_time = start_time.replace(tzinfo=None)
            end_time = end_time.replace(tzinfo=None)
        else:
            # All-day event
            start_time = datetime.strptime(start['date'], '%Y-%m-%d')
            end_time = datetime.strptime(end['date'], '%Y-%m-%d')
            is_all_day = True
        
        return CalendarEvent(
            title=event_data.get('summary', 'Untitled Event'),
            start_time=start_time,
            end_time=end_time,
            location=event_data.get('location'),
            description=event_data.get('description'),
            is_all_day=is_all_day,
        )
    
    def get_calendar_context(self) -> str:
        """
        Generate calendar context for system prompt injection.
        
        Uses SEMANTIC EXPANSION: Instead of raw data, generates
        natural language that makes the meaning explicit so the
        model doesn't need to infer availability.
        """
        if not self.is_authenticated():
            return "Calendar: Not connected"
        
        now = datetime.now()
        lines = ["=== CALENDAR STATUS ==="]
        
        # Current event
        current = self.get_current_event()
        if current:
            remaining = current.end_time - now
            mins_left = int(remaining.total_seconds() / 60)
            lines.append(f"RIGHT NOW: You are in '{current.title}' ({mins_left} minutes remaining)")
        
        # Today's remaining events
        today_events = self.get_todays_events()
        remaining_today = [e for e in today_events if e.start_time > now]
        
        if remaining_today:
            # Has upcoming events
            next_event = remaining_today[0]
            until = next_event.time_until(now)
            hours, remainder = divmod(int(until.total_seconds()), 3600)
            mins = remainder // 60
            
            if hours > 0:
                time_str = f"{hours} hours and {mins} minutes"
            else:
                time_str = f"{mins} minutes"
            
            lines.append(f"NEXT EVENT: '{next_event.title}' in {time_str} ({next_event.format_time_range()})")
            lines.append(f"FREE UNTIL: {next_event.start_time.strftime('%I:%M %p').lstrip('0')}")
            
            if len(remaining_today) > 1:
                lines.append("")
                lines.append("LATER TODAY:")
                for event in remaining_today[1:]:
                    lines.append(event.to_context_string())
        else:
            # No remaining events - SEMANTIC EXPANSION
            lines.append("TODAY'S REMAINING EVENTS: None")
            lines.append("AVAILABILITY: You are FREE for the rest of today.")
            lines.append("This means: FREE this afternoon, FREE this evening, FREE tonight.")
        
        # Tomorrow preview (if after 5pm)
        if now.hour >= 17:
            tomorrow = self.get_tomorrows_events()
            if tomorrow:
                lines.append("")
                lines.append("TOMORROW:")
                for event in tomorrow[:3]:
                    lines.append(event.to_context_string())
            else:
                lines.append("")
                lines.append("TOMORROW: No events scheduled yet.")
        
        lines.append("=== END CALENDAR ===")
        return "\n".join(lines)


# Convenience function
def get_calendar_context(credentials_path: Optional[str] = None) -> str:
    """Quick way to get calendar context string."""
    try:
        provider = CalendarProvider(credentials_path=credentials_path)
        if provider.authenticate():
            return provider.get_calendar_context()
        return "Calendar: Authentication required"
    except ImportError:
        return "Calendar: Google API not installed"
    except Exception as e:
        return f"Calendar: Error - {str(e)}"


if __name__ == "__main__":
    # Test calendar integration
    print("Testing Google Calendar Integration...")
    print("=" * 50)
    
    provider = CalendarProvider()
    
    if provider.authenticate():
        print("✅ Authenticated successfully!\n")
        
        print("Today's events:")
        for event in provider.get_todays_events():
            print(f"  {event.format_time_range()}: {event.title}")
        
        print("\n" + "=" * 50)
        print("CONTEXT FOR MODEL:")
        print("=" * 50)
        print(provider.get_calendar_context())
    else:
        print("❌ Authentication failed")
