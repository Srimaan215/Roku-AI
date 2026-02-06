#!/usr/bin/env python
"""Check Canvas calendar events."""

from core.integrations.calendar_provider import CalendarProvider
from datetime import datetime, timedelta

p = CalendarProvider()
p.authenticate()

service = p.service
cals = service.calendarList().list().execute().get('items', [])

canvas_cal = None
for cal in cals:
    summary = cal.get('summary', '')
    if 'Canvas' in summary:
        canvas_cal = cal
        print(f"Found Canvas calendar: {summary}")
        print(f"  ID: {cal.get('id')}")

if canvas_cal:
    now = datetime.now()
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = now + timedelta(days=7)
    
    print(f"\nEvents this week ({start.date()} to {end.date()}):")
    events = p.get_events(start_date=start, end_date=end, calendar_id=canvas_cal.get('id'))
    
    if events:
        for e in events:
            print(f"  - {e.start_time.strftime('%a %m/%d %I:%M %p')}: {e.title}")
    else:
        print("  No events found in Canvas calendar this week")
else:
    print("Canvas calendar not found. Available calendars:")
    for cal in cals:
        print(f"  - {cal.get('summary')}")
