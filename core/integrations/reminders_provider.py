"""
Apple Reminders Integration for Roku AI

Uses AppleScript to interact with the native Reminders app on macOS.
Supports reading reminders and creating new ones with due dates.
"""

import subprocess
import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class Reminder:
    """Represents a reminder from Apple Reminders."""
    id: str
    name: str
    body: Optional[str]
    due_date: Optional[datetime]
    completed: bool
    list_name: str
    priority: int  # 0=none, 1=high, 5=medium, 9=low
    
    def is_all_day(self) -> bool:
        """Check if this is an all-day reminder (no specific time)."""
        if not self.due_date:
            return False
        # All-day reminders have time set to midnight
        return self.due_date.hour == 0 and self.due_date.minute == 0
    
    def is_overdue(self) -> bool:
        """Check if reminder is past due."""
        if not self.due_date or self.completed:
            return False
        now = datetime.now()
        if self.is_all_day():
            # All-day reminders are only overdue after the day ends
            return now.date() > self.due_date.date()
        return now > self.due_date
    
    def time_until_due(self) -> Optional[timedelta]:
        """Get time until due date."""
        if not self.due_date:
            return None
        return self.due_date - datetime.now()
    
    def format_due(self) -> str:
        """Format due date for display."""
        if not self.due_date:
            return "No due date"
        
        now = datetime.now()
        is_all_day = self.is_all_day()
        
        if self.is_overdue():
            return f"Overdue ({self.due_date.strftime('%b %d')})"
        
        # Check if it's today or tomorrow
        if self.due_date.date() == now.date():
            if is_all_day:
                return "Due today"
            return f"Today at {self.due_date.strftime('%I:%M %p')}"
        elif self.due_date.date() == (now + timedelta(days=1)).date():
            if is_all_day:
                return "Due tomorrow"
            return f"Tomorrow at {self.due_date.strftime('%I:%M %p')}"
        elif (self.due_date - now).days < 7:
            if is_all_day:
                return f"Due {self.due_date.strftime('%A')}"
            return f"{self.due_date.strftime('%A at %I:%M %p')}"
        else:
            if is_all_day:
                return f"Due {self.due_date.strftime('%b %d')}"
            return self.due_date.strftime('%b %d at %I:%M %p')


class RemindersProvider:
    """
    Interface to Apple Reminders via AppleScript.
    
    Capabilities:
    - List all reminder lists
    - Get reminders (all, by list, incomplete only)
    - Get reminders due soon
    - Create new reminders with due dates
    - Mark reminders as complete
    """
    
    def __init__(self):
        self._verify_access()
    
    def _verify_access(self) -> bool:
        """Verify we can access Reminders."""
        script = 'tell application "Reminders" to return name of default list'
        try:
            result = self._run_applescript(script)
            return bool(result)
        except Exception:
            return False
    
    def _run_applescript(self, script: str, timeout: int = 30) -> str:
        """Execute AppleScript and return result."""
        result = subprocess.run(
            ['osascript', '-e', script],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        if result.returncode != 0:
            raise RuntimeError(f"AppleScript error: {result.stderr}")
        return result.stdout.strip()
    
    def get_lists(self) -> List[str]:
        """Get all reminder list names."""
        script = '''
        tell application "Reminders"
            set listNames to {}
            repeat with aList in lists
                set end of listNames to name of aList
            end repeat
            return listNames
        end tell
        '''
        result = self._run_applescript(script)
        if not result:
            return []
        # Parse AppleScript list format: "item1, item2, item3"
        return [name.strip() for name in result.split(", ")]
    
    def get_reminders(
        self,
        list_name: Optional[str] = None,
        include_completed: bool = False,
        due_before: Optional[datetime] = None
    ) -> List[Reminder]:
        """
        Get reminders, optionally filtered.
        
        Args:
            list_name: Filter to specific list
            include_completed: Include completed reminders
            due_before: Only get reminders due before this date
        """
        # Build AppleScript to fetch reminders
        if list_name:
            list_filter = f'list "{list_name}"'
        else:
            list_filter = 'every list'
        
        script = f'''
        tell application "Reminders"
            set output to ""
            repeat with aList in {list_filter if list_name else "lists"}
                set listName to name of aList
                repeat with r in reminders of aList
                    set rName to name of r
                    set rBody to ""
                    try
                        set rBody to body of r
                    end try
                    set rCompleted to completed of r
                    set rDueDate to ""
                    try
                        set rDueDate to due date of r as string
                    end try
                    set rPriority to priority of r
                    set rId to id of r
                    set output to output & rId & "|||" & rName & "|||" & rBody & "|||" & rDueDate & "|||" & rCompleted & "|||" & listName & "|||" & rPriority & "###"
                end repeat
            end repeat
            return output
        end tell
        '''
        
        result = self._run_applescript(script)
        if not result:
            return []
        
        reminders = []
        for item in result.split("###"):
            if not item.strip():
                continue
            
            parts = item.split("|||")
            if len(parts) < 7:
                continue
            
            rid, name, body, due_str, completed_str, list_nm, priority_str = parts[:7]
            
            # Parse completed
            completed = completed_str.lower() == "true"
            if not include_completed and completed:
                continue
            
            # Parse due date
            due_date = None
            if due_str and due_str != "missing value":
                try:
                    # AppleScript date format varies, try common formats
                    for fmt in [
                        "%A, %B %d, %Y at %I:%M:%S %p",
                        "%B %d, %Y at %I:%M:%S %p",
                        "%m/%d/%Y %I:%M:%S %p",
                        "%Y-%m-%d %H:%M:%S"
                    ]:
                        try:
                            due_date = datetime.strptime(due_str, fmt)
                            break
                        except ValueError:
                            continue
                except Exception:
                    pass
            
            # Filter by due date if specified
            if due_before and due_date and due_date > due_before:
                continue
            
            # Parse priority
            try:
                priority = int(priority_str) if priority_str else 0
            except ValueError:
                priority = 0
            
            reminders.append(Reminder(
                id=rid,
                name=name,
                body=body if body and body != "missing value" else None,
                due_date=due_date,
                completed=completed,
                list_name=list_nm,
                priority=priority
            ))
        
        # Sort by due date (None values last)
        reminders.sort(key=lambda r: (r.due_date is None, r.due_date or datetime.max))
        return reminders
    
    def get_due_soon(self, hours: int = 24) -> List[Reminder]:
        """Get reminders due within the next N hours."""
        cutoff = datetime.now() + timedelta(hours=hours)
        reminders = self.get_reminders(include_completed=False)
        return [r for r in reminders if r.due_date and r.due_date <= cutoff]
    
    def get_overdue(self) -> List[Reminder]:
        """Get all overdue reminders."""
        reminders = self.get_reminders(include_completed=False)
        return [r for r in reminders if r.is_overdue()]
    
    def create_reminder(
        self,
        name: str,
        due_date: Optional[datetime] = None,
        list_name: str = "Task Master",
        body: Optional[str] = None,
        priority: int = 0
    ) -> bool:
        """
        Create a new reminder.
        
        Args:
            name: Reminder title
            due_date: When the reminder is due
            list_name: Which list to add to (default: "Reminders")
            body: Optional notes/body text
            priority: 0=none, 1=high, 5=medium, 9=low
            
        Returns:
            True if created successfully
        """
        # Escape name
        name_escaped = name.replace('"', '\\"')
        
        # Build the reminder properties
        props = [f'name:"{name_escaped}"']
        
        if body:
            # Escape quotes in body
            body_escaped = body.replace('"', '\\"')
            props.append(f'body:"{body_escaped}"')
        
        if priority > 0:
            props.append(f'priority:{priority}')
        
        # Add due date to properties if specified
        if due_date:
            date_str = due_date.strftime("%B %d, %Y %I:%M:%S %p")
            props.append(f'due date:date "{date_str}"')
        
        props_str = ", ".join(props)
        
        # Create reminder with all properties at once
        script = f'''
        tell application "Reminders"
            tell list "{list_name}"
                make new reminder with properties {{{props_str}}}
            end tell
            return "success"
        end tell
        '''
        
        try:
            result = self._run_applescript(script)
            return result == "success"
        except Exception as e:
            print(f"Error creating reminder: {e}")
            return False
    
    def complete_reminder(self, reminder_id: str) -> bool:
        """Mark a reminder as complete."""
        script = f'''
        tell application "Reminders"
            repeat with aList in lists
                repeat with r in reminders of aList
                    if id of r is "{reminder_id}" then
                        set completed of r to true
                        return "success"
                    end if
                end repeat
            end repeat
            return "not found"
        end tell
        '''
        
        try:
            result = self._run_applescript(script)
            return result == "success"
        except Exception:
            return False


def parse_reminder_datetime(date_str: str, time_str: Optional[str] = None) -> Optional[datetime]:
    """
    Parse natural language date/time into datetime.
    
    Examples:
        "today", "tomorrow", "friday"
        "today at 3pm", "tomorrow at 10:30am"
        "february 5", "feb 5 at 2pm"
    """
    now = datetime.now()
    date_str = date_str.lower().strip()
    
    # Extract time if provided
    default_time = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
    target_time = default_time.time()
    
    if time_str:
        time_str = time_str.lower().strip()
        # Parse time like "3pm", "10:30am", "14:00"
        import re
        time_match = re.match(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', time_str)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2) or 0)
            ampm = time_match.group(3)
            
            if ampm == 'pm' and hour < 12:
                hour += 12
            elif ampm == 'am' and hour == 12:
                hour = 0
            
            target_time = datetime.now().replace(
                hour=hour, minute=minute, second=0, microsecond=0
            ).time()
    
    # Parse date
    if date_str in ['today', 'now']:
        target_date = now.date()
    elif date_str == 'tomorrow':
        target_date = (now + timedelta(days=1)).date()
    elif date_str in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        target_day = days.index(date_str)
        days_ahead = target_day - now.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        target_date = (now + timedelta(days=days_ahead)).date()
    else:
        # Try parsing as date string
        for fmt in ['%B %d', '%b %d', '%m/%d', '%m-%d']:
            try:
                parsed = datetime.strptime(date_str, fmt)
                target_date = parsed.replace(year=now.year).date()
                # If date is in past, assume next year
                if target_date < now.date():
                    target_date = target_date.replace(year=now.year + 1)
                break
            except ValueError:
                continue
        else:
            return None
    
    return datetime.combine(target_date, target_time)
