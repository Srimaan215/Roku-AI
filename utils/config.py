"""
Configuration management for Roku
"""
import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class Config:
    """Manage Roku configuration"""
    
    DEFAULT_CONFIG = {
        "model": {
            "path": "~/roku-ai/models/base/llama-3.2-3b-q4.gguf",
            "context_size": 2048,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
        },
        "voice": {
            "whisper_model": "tiny",
            "tts_voice": "Samantha",
            "listen_timeout": 5,
        },
        "display": {
            "max_chars_per_line": 40,
            "max_lines": 5,
            "width": 488,
        },
        "privacy": {
            "encrypt_conversations": True,
            "require_biometric": False,
            "audit_logging": True,
        },
        "cloud": {
            "enabled": False,
            "provider": "claude",
            "api_key_env": "ANTHROPIC_API_KEY",
        },
    }
    
    def __init__(self, config_path: str = "~/roku-ai/config.yaml"):
        """
        Initialize configuration
        
        Args:
            config_path: Path to config file
        """
        self.config_path = Path(config_path).expanduser()
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load configuration from file or create default"""
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                user_config = yaml.safe_load(f) or {}
            # Merge with defaults
            return self._merge_config(self.DEFAULT_CONFIG, user_config)
        else:
            return self.DEFAULT_CONFIG.copy()
    
    def _merge_config(self, default: Dict, override: Dict) -> Dict:
        """Deep merge two config dicts"""
        result = default.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value
        return result
    
    def save(self):
        """Save configuration to file"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value by dot-notation key
        
        Args:
            key: Key in dot notation (e.g., "model.temperature")
            default: Default value if not found
        """
        keys = key.split(".")
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set config value by dot-notation key
        
        Args:
            key: Key in dot notation
            value: Value to set
        """
        keys = key.split(".")
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global config instance"""
    global _config
    if _config is None:
        _config = Config()
    return _config


# Example usage
if __name__ == "__main__":
    config = Config()
    
    print("Model path:", config.get("model.path"))
    print("Temperature:", config.get("model.temperature"))
    print("Whisper model:", config.get("voice.whisper_model"))
    
    # Modify and save
    config.set("model.temperature", 0.8)
    config.save()
    print("Config saved!")
