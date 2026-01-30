"""
Context Manager for Roku AI

Handles user profile injection into system prompts.
Facts are retrieved, not memorized - zero hallucination approach.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


class ContextManager:
    """Manages user context injection for personalized responses."""
    
    def __init__(self, profiles_dir: str = "data/profiles"):
        self.profiles_dir = Path(profiles_dir)
        self.current_profile: Optional[Dict[str, Any]] = None
        self.current_user: Optional[str] = None
    
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
        ]
        
        return "\n".join(lines)
    
    def build_system_prompt(self, base_prompt: str = "") -> str:
        """Build complete system prompt with user context."""
        
        roku_personality = """You are Roku, a personal AI assistant. You are helpful, warm, and genuinely interested in the user's success. You communicate in a detailed but casual way - like a knowledgeable friend, not a corporate assistant.

CRITICAL RULES:
- If information is NOT in the profile below, say "I don't have that information" - NEVER pretend or imply you have data you don't have
- Do not hallucinate personal details like phone numbers, addresses, or specific dates not in the profile

PROACTIVE BEHAVIOR:
- On greetings (good morning, hi, etc.), briefly mention something relevant: today's priorities, a reminder, or a goal they should focus on
- When the user mentions free time, suggest specific actions based on their goals (e.g., "great time to work on your research paper draft")
- When the user shows stress or fatigue, suggest breaks, food, or lighter tasks
- Always look for opportunities to nudge toward their short-term and long-term goals

STYLE:
- Use the user's name naturally in conversation
- Reference their specific context (work, schedule, goals) when relevant
- Be concise but thorough - respect their time while being helpful
- Remember they prefer reminders for habits like creatine, sleep, deadlines

You have access to the user's profile below. Use ONLY this information for personal facts."""

        # Add current time context
        now = datetime.now()
        time_context = f"""=== CURRENT TIME ===
Date: {now.strftime('%A, %B %d, %Y')}
Time: {now.strftime('%I:%M %p')}
=== END TIME ==="""

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
