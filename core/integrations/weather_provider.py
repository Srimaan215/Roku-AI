"""
Weather Provider - OpenWeatherMap API integration for Roku AI.
Provides current weather and forecast data for context-aware responses.
"""

import os
import json
import requests
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime
from pathlib import Path


@dataclass
class WeatherData:
    """Current weather conditions."""
    temperature: float  # Fahrenheit
    feels_like: float
    humidity: int
    description: str
    icon: str
    wind_speed: float
    city: str
    
    def format_context(self) -> str:
        """Format weather for context injection."""
        return (
            f"Current weather in {self.city}: {self.description}, "
            f"{self.temperature:.0f}°F (feels like {self.feels_like:.0f}°F), "
            f"humidity {self.humidity}%, wind {self.wind_speed:.0f} mph."
        )
    
    def get_activity_suggestions(self) -> str:
        """Suggest activities based on weather."""
        suggestions = []
        
        # Temperature-based
        if self.temperature < 32:
            suggestions.append("It's freezing - dress warmly if going outside.")
        elif self.temperature < 50:
            suggestions.append("It's cold - a jacket is recommended.")
        elif self.temperature > 85:
            suggestions.append("It's hot - stay hydrated and seek shade.")
        elif 65 <= self.temperature <= 80:
            suggestions.append("Great weather for outdoor activities!")
        
        # Condition-based
        desc_lower = self.description.lower()
        if "rain" in desc_lower or "drizzle" in desc_lower:
            suggestions.append("Rain expected - bring an umbrella.")
        elif "snow" in desc_lower:
            suggestions.append("Snow expected - be careful on roads.")
        elif "clear" in desc_lower or "sunny" in desc_lower:
            suggestions.append("Clear skies - good visibility.")
        elif "cloud" in desc_lower:
            suggestions.append("Overcast conditions.")
        
        return " ".join(suggestions) if suggestions else ""


class WeatherProvider:
    """OpenWeatherMap API provider."""
    
    API_BASE = "https://api.openweathermap.org/data/2.5"
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path.home() / "Roku" / "roku-ai" / "config"
        self.api_key = self._load_api_key()
        self.default_city = "Amherst, MA"  # Default for Srimaan (UMass)
        self.default_lat = 42.3732
        self.default_lon = -72.5199
    
    def _load_api_key(self) -> Optional[str]:
        """Load API key from config or environment."""
        # Try environment variable first
        if api_key := os.environ.get("OPENWEATHER_API_KEY"):
            return api_key
        
        # Try config file
        key_file = self.config_dir / "credentials" / "openweather_key.txt"
        if key_file.exists():
            return key_file.read_text().strip()
        
        return None
    
    def is_configured(self) -> bool:
        """Check if API key is available."""
        return self.api_key is not None
    
    def get_current_weather(self, city: Optional[str] = None) -> Optional[WeatherData]:
        """Fetch current weather conditions."""
        if not self.api_key:
            return None
        
        city = city or self.default_city
        
        try:
            response = requests.get(
                f"{self.API_BASE}/weather",
                params={
                    "q": city,
                    "appid": self.api_key,
                    "units": "imperial"  # Fahrenheit
                },
                timeout=5
            )
            response.raise_for_status()
            data = response.json()
            
            return WeatherData(
                temperature=data["main"]["temp"],
                feels_like=data["main"]["feels_like"],
                humidity=data["main"]["humidity"],
                description=data["weather"][0]["description"].capitalize(),
                icon=data["weather"][0]["icon"],
                wind_speed=data["wind"]["speed"],
                city=data["name"]
            )
        except Exception as e:
            print(f"Weather API error: {e}")
            return None
    
    def get_weather_context(self, city: Optional[str] = None) -> str:
        """Get formatted weather context for RAG injection."""
        weather = self.get_current_weather(city)
        
        if not weather:
            if not self.api_key:
                return "[weather] Weather data unavailable (API key not configured)."
            return "[weather] Weather data temporarily unavailable."
        
        context_parts = [
            "=== WEATHER STATUS ===",
            weather.format_context(),
        ]
        
        if suggestions := weather.get_activity_suggestions():
            context_parts.append(f"Suggestions: {suggestions}")
        
        context_parts.append("=== END WEATHER ===")
        
        return "[weather] " + "\n".join(context_parts)
    
    @staticmethod
    def setup_api_key(api_key: str, config_dir: Optional[Path] = None) -> bool:
        """Save API key to config file."""
        config_dir = config_dir or Path.home() / "Roku" / "roku-ai" / "config"
        creds_dir = config_dir / "credentials"
        creds_dir.mkdir(parents=True, exist_ok=True)
        
        key_file = creds_dir / "openweather_key.txt"
        key_file.write_text(api_key)
        print(f"✅ API key saved to {key_file}")
        return True


if __name__ == "__main__":
    # Quick test
    provider = WeatherProvider()
    
    if not provider.is_configured():
        print("⚠️  OpenWeatherMap API key not configured.")
        print("\nTo set up:")
        print("1. Get free API key from https://openweathermap.org/api")
        print("2. Run: python -c \"from core.integrations.weather_provider import WeatherProvider; WeatherProvider.setup_api_key('YOUR_KEY')\"")
    else:
        print("Testing weather API...")
        print(provider.get_weather_context())
