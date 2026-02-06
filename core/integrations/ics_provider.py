"""
ICS Calendar Feed Provider for Roku AI

Fetches and parses ICS feeds directly (Canvas, iCloud, etc.)
"""

import requests
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from icalendar import Calendar
import pytz


@dataclass
class ICSEvent:
    """Represents an event from an ICS feed."""
    uid: str
    title: str
    start_time: datetime
    end_time: Optional[datetime]
    description: str
    url: Optional[str]
    source: str  # Which feed it came from
    
    def format_time_range(self) -> str:
        """Format the event time range."""
        if self.end_time and self.end_time != self.start_time:
            return f"{self.start_time.strftime('%I:%M %p')} - {self.end_time.strftime('%I:%M %p')}"
        elif self.start_time.hour == 0 and self.start_time.minute == 0:
            return "All Day"
        else:
            return self.start_time.strftime('%I:%M %p')
    
    def is_assignment(self) -> bool:
        """Check if this event is an assignment/homework."""
        keywords = ['assignment', 'quiz', 'lab', 'homework', 'hw', 'due', 'exam', 'test', 'midterm', 'final']
        title_lower = self.title.lower()
        return any(kw in title_lower for kw in keywords)


class ICSProvider:
    """
    Fetches and parses ICS calendar feeds.
    
    Supports:
    - Canvas calendar feeds
    - iCloud calendar feeds
    - Any standard ICS URL
    """
    
    def __init__(self):
        self.feeds: Dict[str, str] = {}  # name -> url
        self._cache: Dict[str, tuple] = {}  # url -> (events, timestamp)
        self._cache_ttl = 300  # 5 minute cache
    
    def add_feed(self, name: str, url: str) -> None:
        """Add an ICS feed to track."""
        self.feeds[name] = url
        
    def remove_feed(self, name: str) -> None:
        """Remove a feed."""
        if name in self.feeds:
            del self.feeds[name]
    
    def _fetch_feed(self, url: str) -> Optional[str]:
        """Fetch raw ICS data from URL."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"Error fetching ICS feed: {e}")
            return None
    
    def _parse_ics(self, ics_data: str, source: str) -> List[ICSEvent]:
        """Parse ICS data into events."""
        events = []
        
        try:
            cal = Calendar.from_ical(ics_data)
            
            for component in cal.walk():
                if component.name == "VEVENT":
                    # Get event details
                    uid = str(component.get('uid', ''))
                    summary = str(component.get('summary', 'Untitled'))
                    description = str(component.get('description', ''))
                    url = str(component.get('url', '')) if component.get('url') else None
                    
                    # Parse start time
                    dtstart = component.get('dtstart')
                    if dtstart:
                        start = dtstart.dt
                        # Handle date vs datetime
                        if isinstance(start, datetime):
                            if start.tzinfo:
                                start = start.astimezone(pytz.timezone('America/New_York'))
                            start = start.replace(tzinfo=None)
                        else:
                            # It's a date, convert to datetime at midnight
                            start = datetime.combine(start, datetime.min.time())
                    else:
                        continue  # Skip events without start time
                    
                    # Parse end time
                    dtend = component.get('dtend')
                    if dtend:
                        end = dtend.dt
                        if isinstance(end, datetime):
                            if end.tzinfo:
                                end = end.astimezone(pytz.timezone('America/New_York'))
                            end = end.replace(tzinfo=None)
                        else:
                            end = datetime.combine(end, datetime.min.time())
                    else:
                        end = start
                    
                    events.append(ICSEvent(
                        uid=uid,
                        title=summary,
                        start_time=start,
                        end_time=end,
                        description=description,
                        url=url,
                        source=source
                    ))
                    
        except Exception as e:
            print(f"Error parsing ICS data: {e}")
        
        return events
    
    def get_events(
        self, 
        start_date: datetime, 
        end_date: datetime,
        feed_name: Optional[str] = None,
        use_cache: bool = True
    ) -> List[ICSEvent]:
        """
        Get events from ICS feeds within date range.
        
        Args:
            start_date: Start of date range
            end_date: End of date range
            feed_name: Specific feed to query, or None for all feeds
            use_cache: Whether to use cached data
        """
        all_events = []
        
        feeds_to_query = {feed_name: self.feeds[feed_name]} if feed_name else self.feeds
        
        for name, url in feeds_to_query.items():
            # Check cache
            if use_cache and url in self._cache:
                cached_events, timestamp = self._cache[url]
                if (datetime.now() - timestamp).total_seconds() < self._cache_ttl:
                    events = cached_events
                else:
                    events = self._fetch_and_parse(url, name)
            else:
                events = self._fetch_and_parse(url, name)
            
            # Filter to date range
            for event in events:
                if start_date <= event.start_time <= end_date:
                    all_events.append(event)
        
        # Sort by start time
        all_events.sort(key=lambda e: e.start_time)
        return all_events
    
    def _fetch_and_parse(self, url: str, name: str) -> List[ICSEvent]:
        """Fetch and parse a feed, updating cache."""
        ics_data = self._fetch_feed(url)
        if ics_data:
            events = self._parse_ics(ics_data, name)
            self._cache[url] = (events, datetime.now())
            return events
        return []
    
    def get_assignments(
        self,
        start_date: datetime,
        end_date: datetime,
        feed_name: Optional[str] = None
    ) -> List[ICSEvent]:
        """Get only assignment/homework events."""
        events = self.get_events(start_date, end_date, feed_name)
        return [e for e in events if e.is_assignment()]


# Convenience function to create pre-configured provider
def create_canvas_provider(ics_url: str) -> ICSProvider:
    """Create an ICS provider configured with Canvas feed."""
    provider = ICSProvider()
    provider.add_feed("canvas", ics_url)
    return provider
