#!/usr/bin/env python
"""
Roku Personal Onboarding
Interactive interview to gather personal context for the personal adapter.
Like setting up a new phone - the LLM learns who you are.
"""
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

PROFILE_DIR = Path.home() / "Roku/roku-ai/data/profiles"
PROFILE_DIR.mkdir(parents=True, exist_ok=True)


class OnboardingInterview:
    """Interactive onboarding to gather personal context"""
    
    def __init__(self, username: str = "default"):
        self.username = username
        self.profile: Dict = {
            "username": username,
            "created_at": datetime.now().isoformat(),
            "identity": {},
            "work": {},
            "schedule": {},
            "preferences": {},
            "goals": {},
        }
        self.conversation_log: List[Dict] = []
    
    def ask(self, question: str, category: str, key: str, 
            followup: Optional[str] = None) -> str:
        """Ask a question and store the response"""
        print(f"\n🤖 Roku: {question}")
        response = input("You: ").strip()
        
        self.conversation_log.append({
            "question": question,
            "response": response,
            "category": category,
            "key": key,
        })
        
        if category not in self.profile:
            self.profile[category] = {}
        self.profile[category][key] = response
        
        # Optional follow-up for more detail
        if followup and response and len(response) < 50:
            print(f"\n🤖 Roku: {followup}")
            detail = input("You: ").strip()
            if detail:
                self.profile[category][f"{key}_detail"] = detail
                self.conversation_log.append({
                    "question": followup,
                    "response": detail,
                    "category": category,
                    "key": f"{key}_detail",
                })
        
        return response
    
    def run_interview(self):
        """Run the full onboarding interview"""
        print("\n" + "=" * 60)
        print("👋 Welcome to Roku!")
        print("I'm your personal AI assistant. Let's get to know each other.")
        print("Feel free to skip any question by pressing Enter.")
        print("=" * 60)
        
        # Identity
        print("\n--- About You ---")
        self.ask(
            "First, what's your name?",
            "identity", "name"
        )
        self.ask(
            "Nice to meet you! How would you describe yourself in a few words?",
            "identity", "description",
            followup="Tell me a bit more about what makes you, you."
        )
        
        # Work
        print("\n--- Your Work ---")
        self.ask(
            "What do you do for work?",
            "work", "role",
            followup="Can you tell me more about what that involves day-to-day?"
        )
        self.ask(
            "Where do you work? (Company/industry/field)",
            "work", "company"
        )
        self.ask(
            "What are you working on right now that excites you?",
            "work", "current_projects"
        )
        self.ask(
            "What skills or tools do you use most in your work?",
            "work", "skills"
        )
        
        # Schedule
        print("\n--- Your Schedule ---")
        self.ask(
            "What time do you usually start your day?",
            "schedule", "wake_time"
        )
        self.ask(
            "What time do you typically finish work?",
            "schedule", "work_end"
        )
        self.ask(
            "Do you have any regular commitments? (meetings, classes, gym, etc.)",
            "schedule", "regular_commitments"
        )
        self.ask(
            "What does your ideal productive day look like?",
            "schedule", "ideal_day"
        )
        
        # Preferences
        print("\n--- Your Preferences ---")
        self.ask(
            "How do you prefer to communicate? (brief/detailed, formal/casual)",
            "preferences", "communication_style"
        )
        self.ask(
            "What kind of reminders or nudges would be helpful?",
            "preferences", "helpful_reminders"
        )
        self.ask(
            "Are there topics you'd like me to avoid or be careful about?",
            "preferences", "avoid_topics"
        )
        
        # Goals
        print("\n--- Your Goals ---")
        self.ask(
            "What are you trying to accomplish this week/month?",
            "goals", "short_term"
        )
        self.ask(
            "What's a bigger goal you're working towards?",
            "goals", "long_term"
        )
        self.ask(
            "How can I best help you achieve these goals?",
            "goals", "how_to_help"
        )
        
        # Wrap up
        print("\n" + "=" * 60)
        print("🎉 Thanks for sharing! I now know you much better.")
        print("I'll use this to personalize my responses for you.")
        print("=" * 60)
    
    def save_profile(self) -> Path:
        """Save the profile to disk"""
        filepath = PROFILE_DIR / f"{self.username}.json"
        with open(filepath, "w") as f:
            json.dump({
                "profile": self.profile,
                "conversation_log": self.conversation_log,
            }, f, indent=2)
        print(f"\n💾 Profile saved to: {filepath}")
        return filepath
    
    def generate_training_data(self) -> List[Dict]:
        """
        Generate training data from the profile.
        Creates Q&A pairs the model can learn from.
        """
        training_data = []
        name = self.profile.get("identity", {}).get("name", "the user")
        
        # Generate training examples from profile
        templates = [
            # Identity questions
            {
                "context": "User asks about themselves",
                "questions": [
                    "Who am I?",
                    "Tell me about myself",
                    "What do you know about me?",
                    "Remind me who I am",
                ],
                "answer_template": self._build_identity_answer,
            },
            # Work questions
            {
                "context": "User asks about their work",
                "questions": [
                    "What do I do for work?",
                    "What's my job?",
                    "Tell me about my work",
                    "What am I working on?",
                ],
                "answer_template": self._build_work_answer,
            },
            # Schedule questions
            {
                "context": "User asks about their schedule",
                "questions": [
                    "What's my schedule like?",
                    "When do I usually wake up?",
                    "What are my regular commitments?",
                ],
                "answer_template": self._build_schedule_answer,
            },
            # Goals questions
            {
                "context": "User asks about their goals",
                "questions": [
                    "What are my goals?",
                    "What am I trying to achieve?",
                    "Remind me what I'm working towards",
                ],
                "answer_template": self._build_goals_answer,
            },
        ]
        
        for template in templates:
            answer = template["answer_template"]()
            if answer:  # Only include if we have data
                for question in template["questions"]:
                    training_data.append({
                        "instruction": question,
                        "input": "",
                        "output": answer,
                        "context": template["context"],
                    })
        
        # Add personalized greeting examples
        training_data.extend(self._build_greeting_examples())
        
        # Add proactive suggestion examples
        training_data.extend(self._build_proactive_examples())
        
        return training_data
    
    def _build_identity_answer(self) -> str:
        identity = self.profile.get("identity", {})
        work = self.profile.get("work", {})
        
        parts = []
        if identity.get("name"):
            parts.append(f"You're {identity['name']}")
        if identity.get("description"):
            parts.append(identity["description"])
        if work.get("role"):
            parts.append(f"You work as {work['role']}")
            if work.get("company"):
                parts.append(f"at {work['company']}")
        
        return ". ".join(parts) + "." if parts else ""
    
    def _build_work_answer(self) -> str:
        work = self.profile.get("work", {})
        
        parts = []
        if work.get("role"):
            parts.append(f"You work as {work['role']}")
        if work.get("company"):
            parts.append(f"at {work['company']}")
        if work.get("role_detail"):
            parts.append(f"Your day-to-day involves {work['role_detail']}")
        if work.get("current_projects"):
            parts.append(f"Currently, you're excited about {work['current_projects']}")
        if work.get("skills"):
            parts.append(f"You mainly use {work['skills']}")
        
        return ". ".join(parts) + "." if parts else ""
    
    def _build_schedule_answer(self) -> str:
        schedule = self.profile.get("schedule", {})
        
        parts = []
        if schedule.get("wake_time"):
            parts.append(f"You typically start your day around {schedule['wake_time']}")
        if schedule.get("work_end"):
            parts.append(f"and finish work around {schedule['work_end']}")
        if schedule.get("regular_commitments"):
            parts.append(f"Your regular commitments include {schedule['regular_commitments']}")
        
        return ". ".join(parts) + "." if parts else ""
    
    def _build_goals_answer(self) -> str:
        goals = self.profile.get("goals", {})
        
        parts = []
        if goals.get("short_term"):
            parts.append(f"This week/month, you're focused on {goals['short_term']}")
        if goals.get("long_term"):
            parts.append(f"Longer term, you're working towards {goals['long_term']}")
        if goals.get("how_to_help"):
            parts.append(f"I can help by {goals['how_to_help']}")
        
        return ". ".join(parts) + "." if parts else ""
    
    def _build_greeting_examples(self) -> List[Dict]:
        """Build personalized greeting examples"""
        name = self.profile.get("identity", {}).get("name", "")
        if not name:
            return []
        
        work = self.profile.get("work", {})
        schedule = self.profile.get("schedule", {})
        
        examples = [
            {
                "instruction": "Good morning!",
                "input": "",
                "output": f"Good morning, {name}! Ready to tackle the day?",
                "context": "Morning greeting",
            },
            {
                "instruction": "Hey Roku",
                "input": "",
                "output": f"Hey {name}! What can I help you with?",
                "context": "Casual greeting",
            },
        ]
        
        # Add context-aware greetings
        if work.get("current_projects"):
            examples.append({
                "instruction": "What should I focus on today?",
                "input": "",
                "output": f"Based on what you told me, you're currently working on {work['current_projects']}. Want to dive into that, or do you have something else in mind?",
                "context": "Focus suggestion",
            })
        
        return examples
    
    def _build_proactive_examples(self) -> List[Dict]:
        """Build proactive suggestion examples"""
        examples = []
        
        goals = self.profile.get("goals", {})
        prefs = self.profile.get("preferences", {})
        
        if goals.get("short_term"):
            examples.append({
                "instruction": "Give me a nudge about my goals",
                "input": "",
                "output": f"Just checking in - you mentioned wanting to {goals['short_term']}. How's that going? Need any help?",
                "context": "Goal reminder",
            })
        
        if prefs.get("helpful_reminders"):
            examples.append({
                "instruction": "What kind of reminders should you give me?",
                "input": "",
                "output": f"You asked me to remind you about: {prefs['helpful_reminders']}. I'll keep an eye on those for you!",
                "context": "Reminder preferences",
            })
        
        return examples
    
    def save_training_data(self) -> Path:
        """Save training data for fine-tuning"""
        training_data = self.generate_training_data()
        
        training_dir = Path.home() / "Roku/roku-ai/training/data"
        training_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = training_dir / f"personal_{self.username}.json"
        with open(filepath, "w") as f:
            json.dump(training_data, f, indent=2)
        
        print(f"\n📝 Training data saved: {filepath}")
        print(f"   Generated {len(training_data)} training examples")
        
        return filepath


def main():
    """Run the onboarding interview"""
    print("\n🚀 Starting Roku Onboarding...")
    
    username = input("What username would you like? (default: srimaan): ").strip()
    if not username:
        username = "srimaan"
    
    interview = OnboardingInterview(username)
    interview.run_interview()
    interview.save_profile()
    training_path = interview.save_training_data()
    
    print("\n" + "=" * 60)
    print("✅ Onboarding complete!")
    print(f"   Profile: ~/Roku/roku-ai/data/profiles/{username}.json")
    print(f"   Training data: {training_path}")
    print("\nNext step: Train the personal adapter with this data")
    print("=" * 60)


if __name__ == "__main__":
    main()
