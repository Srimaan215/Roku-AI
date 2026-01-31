"""
Context Manager for Roku AI

Handles user profile injection into system prompts.
Facts are retrieved, not memorized - zero hallucination approach.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Optional calendar integration
try:
    from core.integrations.calendar_provider import CalendarProvider
    CALENDAR_AVAILABLE = True
except ImportError:
    CALENDAR_AVAILABLE = False


class ContextManager:
    """Manages user context injection for personalized responses."""
    
    def __init__(self, profiles_dir: str = "data/profiles", enable_calendar: bool = True):
        self.profiles_dir = Path(profiles_dir)
        self.current_profile: Optional[Dict[str, Any]] = None
        self.current_user: Optional[str] = None
        
        # Calendar integration
        self.calendar: Optional[CalendarProvider] = None
        self.calendar_enabled = False
        
        if enable_calendar and CALENDAR_AVAILABLE:
            self._init_calendar()
    
    def _init_calendar(self):
        """Initialize calendar provider if credentials exist."""
        try:
            self.calendar = CalendarProvider()
            if self.calendar.token_path.exists():
                # Token exists, try to authenticate silently
                self.calendar_enabled = self.calendar.authenticate()
            # If no token, don't prompt - user needs to run setup
        except Exception as e:
            print(f"Calendar init warning: {e}")
            self.calendar_enabled = False
    
    def connect_calendar(self) -> bool:
        """Explicitly connect to calendar (may prompt for OAuth)."""
        if not CALENDAR_AVAILABLE:
            print("Calendar integration not available. Install: pip install google-auth-oauthlib google-api-python-client")
            return False
        
        if self.calendar is None:
            self.calendar = CalendarProvider()
        
        self.calendar_enabled = self.calendar.authenticate()
        return self.calendar_enabled
    
    def load_profile(self, username: str) -> bool:
        """Load a user profile from disk."""
        profile_path = self.profiles_dir / f"{username}.json"
        if not profile_path.exists():
            return False
        
        with open(profile_path, 'r') as f:
            data = json.load(f)
        
        self.current_profile = data.get('profile', data)
        self.current_user = username
        return True
    
    def get_profile_summary(self) -> str:
        """Generate a compact profile summary for system prompt injection."""
        if not self.current_profile:
            return ""
        
        p = self.current_profile
        
        # Build structured summary
        lines = [
            f"=== USER PROFILE: {self.current_user} ===",
            "",
            "IDENTITY:",
            f"  Name: {p.get('identity', {}).get('name', 'Unknown')}",
            f"  Description: {p.get('identity', {}).get('description', '')}",
            "",
        ]
        
        # Add location if available
        if 'location' in p:
            loc = p['location']
            lines.extend([
                "LOCATION:",
                f"  Home: {loc.get('home', 'Unknown')}",
                f"  Work: {loc.get('work', '')}",
                f"  Timezone: {loc.get('timezone', '')}",
            ])
            if 'known_places' in loc:
                places = [f"{k}: {v}" for k, v in loc['known_places'].items()]
                lines.append(f"  Known places: {', '.join(places)}")
            lines.append("")
        
        lines.extend([
            "WORK/EDUCATION:",
            f"  Role: {p.get('work', {}).get('role', '')}",
            f"  Institutions: {p.get('work', {}).get('company', '')}",
            f"  Current Project: {p.get('work', {}).get('current_projects', '')}",
            f"  Skills: {p.get('work', {}).get('skills', '')}",
            "",
            "SCHEDULE:",
            f"  Wake time: {p.get('schedule', {}).get('wake_time', '')}",
            f"  Work hours: {p.get('schedule', {}).get('work_end', '')}",
            f"  Regular commitments: {p.get('schedule', {}).get('regular_commitments', '')}",
            f"  Ideal day: {p.get('schedule', {}).get('ideal_day', '')}",
            "",
            "PREFERENCES:",
            f"  Communication style: {p.get('preferences', {}).get('communication_style', '')}",
            f"  Helpful reminders: {p.get('preferences', {}).get('helpful_reminders', '')}",
            "",
            "GOALS:",
            f"  Short-term: {p.get('goals', {}).get('short_term', '')}",
            f"  Long-term: {p.get('goals', {}).get('long_term', '')}",
            f"  How to help: {p.get('goals', {}).get('how_to_help', '')}",
            "",
            "=== END PROFILE ===",
        ])
        
        return "\n".join(lines)
    
    def build_system_prompt(self, base_prompt: str = "") -> str:
        """Build complete system prompt with user context."""
        
        roku_personality = """You are Roku, a personal AI assistant. You are helpful, warm, and genuinely interested in the user's success. You communicate in a detailed but casual way - like a knowledgeable friend, not a corporate assistant.

CRITICAL RULES:
- If information is NOT in the profile or calendar below, say "I don't have that information" - NEVER pretend or imply you have data you don't have
- Do not hallucinate personal details like phone numbers, addresses, or specific dates not in the profile
- The profile contains a general daily routine, NOT a day-by-day schedule - do not infer specific events from it

CALENDAR RULES:
- The CALENDAR STATUS section below shows your real schedule - use it directly
- It explicitly states your availability (e.g., "FREE this afternoon, FREE tonight")
- Only say "I don't have that information" for days NOT shown in the calendar

WEEKDAY VS WEEKEND:
- Check the current date below - if it's Saturday or Sunday, the user likely has NO classes or work meetings
- On weekends, focus on personal goals, rest, hobbies, or self-care - not work unless they mention it

PROACTIVE BEHAVIOR:
- On greetings (good morning, hi, etc.), briefly mention something relevant: today's priorities, a reminder, or a goal they should focus on
- When the user mentions free time, suggest specific actions based on their goals (e.g., "great time to work on your research paper draft")
- When the user shows stress or fatigue, suggest breaks, food, or lighter tasks
- Always look for opportunities to nudge toward their short-term and long-term goals

TIME-AWARE SUGGESTIONS:
- Morning (6am-12pm): Focus on high-priority work, mention if they should have had breakfast/shake
- Afternoon (12pm-5pm): Check if they've eaten lunch, remind about classes/meetings
- Evening (5pm-9pm): Good time for gym, dinner, winding down
- Night (9pm-2am): Gentle reminders about sleep, especially if past midnight
- Late night (2am+): Strongly encourage sleep unless they have a deadline

LOCATION-AWARE BEHAVIOR:
- Reference their known places when relevant (lab, gym, library)
- Adjust suggestions based on where they likely are given the time

STYLE:
- Use the user's name naturally in conversation
- Reference their specific context (work, schedule, goals) when relevant
- Be concise but thorough - respect their time while being helpful
- Remember they prefer reminders for habits like creatine, sleep, deadlines

You have access to the user's profile below. Use ONLY this information for personal facts."""

        # Add current time context with time-of-day awareness
        now = datetime.now()
        hour = now.hour
        is_weekend = now.weekday() >= 5  # Saturday=5, Sunday=6
        
        # Determine time of day context (weekend-aware)
        if 6 <= hour < 12:
            time_period = "morning"
            if is_weekend:
                suggestion = "Weekend morning - good for rest, hobbies, or personal projects."
            else:
                suggestion = "Good time for focused work and high-priority tasks."
        elif 12 <= hour < 17:
            time_period = "afternoon"
            if is_weekend:
                suggestion = "Weekend afternoon - time for activities, errands, or relaxation."
            else:
                suggestion = "Check if lunch was had. May have classes/meetings (only if calendar shows them)."
        elif 17 <= hour < 21:
            time_period = "evening"
            suggestion = "Good time for gym, dinner, or winding down."
        elif 21 <= hour or hour < 2:
            time_period = "night"
            suggestion = "Consider wrapping up soon. Sleep is important."
        else:
            time_period = "late night"
            suggestion = "It's very late. Encourage sleep unless urgent deadline."
        
        day_type = "WEEKEND" if is_weekend else "WEEKDAY"
        
        # Build time context
        time_lines = [
            "=== CURRENT CONTEXT ===",
            f"Date: {now.strftime('%A, %B %d, %Y')} ({day_type})",
            f"Time: {now.strftime('%I:%M %p')} ({time_period})",
            f"Time guidance: {suggestion}",
        ]
        
        # Add calendar context if available
        if self.calendar_enabled and self.calendar:
            calendar_context = self.calendar.get_calendar_context()
            time_lines.append("")
            time_lines.append(calendar_context)
        else:
            time_lines.append("Calendar: Not connected - do NOT assume any specific events")
        
        time_lines.append("=== END CONTEXT ===")
        time_context = "\n".join(time_lines)

        profile_context = self.get_profile_summary()
        
        if profile_context:
            return f"{roku_personality}\n\n{time_context}\n\n{profile_context}\n\n{base_prompt}".strip()
        else:
            return f"{roku_personality}\n\n{time_context}\n\n{base_prompt}".strip()
    
    def get_context_tokens_estimate(self) -> int:
        """Estimate token count for context (rough: 4 chars ≈ 1 token)."""
        summary = self.get_profile_summary()
        return len(summary) // 4


# Convenience function
def load_user_context(username: str) -> ContextManager:
    """Quick loader for user context."""
    cm = ContextManager()
    cm.load_profile(username)
    return cm


if __name__ == "__main__":
    # Test
    cm = ContextManager()
    if cm.load_profile("Srimaan"):
        print("Profile loaded!")
        print(f"Estimated tokens: {cm.get_context_tokens_estimate()}")
        print("\n" + "="*60 + "\n")
        print(cm.build_system_prompt())
    else:
        print("Profile not found")
