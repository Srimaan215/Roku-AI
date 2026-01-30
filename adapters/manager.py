"""
Manage multiple LoRA adapters
"""
from pathlib import Path
from typing import Optional, Dict
import json


class AdapterManager:
    """Manage and switch between domain adapters"""
    
    def __init__(self, adapters_dir: str = "~/Roku/roku-ai/models/adapters"):
        """
        Initialize adapter manager
        
        Args:
            adapters_dir: Directory containing LoRA adapters
        """
        self.adapters_dir = Path(adapters_dir).expanduser()
        self.adapters_dir.mkdir(parents=True, exist_ok=True)
        self.available_adapters = self._discover_adapters()
        self.active_adapter = None
        
        if self.available_adapters:
            print(f"Found {len(self.available_adapters)} adapters:")
            for name in self.available_adapters:
                print(f"  - {name}")
        else:
            print("No adapters found (base model only)")
    
    def _discover_adapters(self) -> Dict[str, Path]:
        """Discover available adapters (GGUF or PEFT format)"""
        adapters = {}
        
        if not self.adapters_dir.exists():
            return adapters
        
        for item in self.adapters_dir.iterdir():
            # Check for GGUF adapters
            if item.suffix == ".gguf":
                name = item.stem.replace("_adapter", "").replace("_lora", "")
                adapters[name] = item
            # Check for PEFT adapter directories
            elif item.is_dir() and (item / "adapter_config.json").exists():
                name = item.name.replace("_adapter", "").replace("_lora", "")
                adapters[name] = item
        
        return adapters
    
    def detect_domain(self, user_message: str) -> str:
        """
        Detect which domain a message belongs to
        
        Args:
            user_message: User's input
            
        Returns:
            Domain name (work, home, health, personal)
        """
        message_lower = user_message.lower()
        
        # Work keywords
        work_keywords = [
            "meeting", "email", "project", "deadline", "schedule",
            "calendar", "client", "presentation", "report"
        ]
        
        # Home keywords
        home_keywords = [
            "lights", "temperature", "thermostat", "lock", "door",
            "home", "house", "room", "bed", "security"
        ]
        
        # Health keywords
        health_keywords = [
            "workout", "exercise", "sleep", "steps", "calories",
            "health", "fitness", "medication", "vitamin", "run"
        ]
        
        # Count matches
        work_score = sum(1 for kw in work_keywords if kw in message_lower)
        home_score = sum(1 for kw in home_keywords if kw in message_lower)
        health_score = sum(1 for kw in health_keywords if kw in message_lower)
        
        # Return highest scoring domain
        scores = {
            "work": work_score,
            "home": home_score,
            "health": health_score,
        }
        
        max_domain = max(scores, key=scores.get)
        max_score = scores[max_domain]
        
        if max_score > 0:
            return max_domain
        
        return "personal"
    
    def get_adapter_path(self, domain: str) -> Optional[Path]:
        """Get path to adapter for domain"""
        return self.available_adapters.get(domain)
    
    def select_adapter(self, user_message: str) -> Optional[str]:
        """Select appropriate adapter based on message"""
        domain = self.detect_domain(user_message)
        
        if domain in self.available_adapters:
            self.active_adapter = domain
            return domain
        
        return None
    
    def get_adapter_info(self, domain: str) -> Optional[Dict]:
        """Get adapter metadata"""
        adapter_path = self.get_adapter_path(domain)
        if adapter_path is None:
            return None
        
        config_path = adapter_path / "adapter_config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                return json.load(f)
        
        return None


# Example usage
if __name__ == "__main__":
    manager = AdapterManager()
    
    test_messages = [
        "Schedule a meeting for tomorrow",
        "Turn off the living room lights",
        "How many steps did I take today?",
        "What's the weather like?",
    ]
    
    for msg in test_messages:
        domain = manager.detect_domain(msg)
        adapter_path = manager.get_adapter_path(domain)
        print(f"'{msg}'")
        print(f"  → Domain: {domain}")
        print(f"  → Adapter: {adapter_path}")
        print()
