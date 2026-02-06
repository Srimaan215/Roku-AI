"""
Smart Home Integration for Roku AI

Provides device discovery and control for smart home devices.
Supports:
- HomeKit (via pyhap, future)
- Matter (via matter-server, future)
- Mock devices (for development/testing)

Current implementation: Mock provider for development.
Future: Real HomeKit/Matter integration.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from pathlib import Path
import json


class DeviceType(Enum):
    """Smart home device types."""
    LIGHT = "light"
    THERMOSTAT = "thermostat"
    LOCK = "lock"
    SWITCH = "switch"
    SENSOR = "sensor"
    CAMERA = "camera"
    SPEAKER = "speaker"
    BLINDS = "blinds"
    FAN = "fan"
    OUTLET = "outlet"


class DeviceState(Enum):
    """Device state values."""
    ON = "on"
    OFF = "off"
    LOCKED = "locked"
    UNLOCKED = "unlocked"
    OPEN = "open"
    CLOSED = "closed"


@dataclass
class SmartDevice:
    """Represents a smart home device."""
    id: str
    name: str
    type: DeviceType
    room: Optional[str] = None
    state: Dict[str, Any] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)
    manufacturer: Optional[str] = None
    
    def is_on(self) -> bool:
        """Check if device is on (for lights/switches)."""
        return self.state.get("power", "off").lower() == "on"
    
    def get_temperature(self) -> Optional[float]:
        """Get current temperature (for thermostats)."""
        return self.state.get("temperature")
    
    def get_target_temperature(self) -> Optional[float]:
        """Get target temperature (for thermostats)."""
        return self.state.get("target_temperature")
    
    def is_locked(self) -> bool:
        """Check if device is locked (for locks)."""
        return self.state.get("lock_state", "unlocked").lower() == "locked"
    
    def to_context_string(self) -> str:
        """Format device for context injection."""
        parts = [f"{self.name} ({self.type.value})"]
        
        if self.room:
            parts.append(f"in {self.room}")
        
        if self.type == DeviceType.LIGHT or self.type == DeviceType.SWITCH:
            parts.append(f"is {'ON' if self.is_on() else 'OFF'}")
        elif self.type == DeviceType.THERMOSTAT:
            temp = self.get_temperature()
            target = self.get_target_temperature()
            if temp:
                parts.append(f"is {temp}°F")
            if target:
                parts.append(f"(set to {target}°F)")
        elif self.type == DeviceType.LOCK:
            parts.append(f"is {'LOCKED' if self.is_locked() else 'UNLOCKED'}")
        
        return " ".join(parts)


class SmartHomeProvider:
    """
    Smart home device provider.
    
    Currently implements a mock provider for development.
    Future: Real HomeKit/Matter integration.
    
    Usage:
        provider = SmartHomeProvider()
        devices = provider.discover_devices()
        provider.control_device("living_room_light", "turn_on")
    """
    
    DEFAULT_CONFIG_DIR = Path.home() / "Roku/roku-ai/config"
    DEVICES_CONFIG_FILE = "smart_home_devices.json"
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize smart home provider.
        
        Args:
            config_dir: Directory for device configuration
        """
        self.config_dir = config_dir or self.DEFAULT_CONFIG_DIR
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.devices_file = self.config_dir / self.DEVICES_CONFIG_FILE
        self.devices: Dict[str, SmartDevice] = {}
        
        # Load devices from config
        self._load_devices()
        
        # If no devices, initialize with mock devices
        if not self.devices:
            self._initialize_mock_devices()
    
    def _load_devices(self) -> None:
        """Load devices from config file."""
        if not self.devices_file.exists():
            return
        
        try:
            with open(self.devices_file, 'r') as f:
                data = json.load(f)
            
            for device_data in data.get("devices", []):
                device = SmartDevice(
                    id=device_data["id"],
                    name=device_data["name"],
                    type=DeviceType(device_data["type"]),
                    room=device_data.get("room"),
                    state=device_data.get("state", {}),
                    capabilities=device_data.get("capabilities", []),
                    manufacturer=device_data.get("manufacturer"),
                )
                self.devices[device.id] = device
        except Exception as e:
            print(f"Error loading devices: {e}")
    
    def _save_devices(self) -> None:
        """Save devices to config file."""
        data = {
            "devices": [
                {
                    "id": d.id,
                    "name": d.name,
                    "type": d.type.value,
                    "room": d.room,
                    "state": d.state,
                    "capabilities": d.capabilities,
                    "manufacturer": d.manufacturer,
                }
                for d in self.devices.values()
            ]
        }
        
        with open(self.devices_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _initialize_mock_devices(self) -> None:
        """Initialize with mock devices for development."""
        mock_devices = [
            SmartDevice(
                id="living_room_light",
                name="Living Room Light",
                type=DeviceType.LIGHT,
                room="Living Room",
                state={"power": "off", "brightness": 100},
                capabilities=["power", "brightness"],
            ),
            SmartDevice(
                id="bedroom_light",
                name="Bedroom Light",
                type=DeviceType.LIGHT,
                room="Bedroom",
                state={"power": "off", "brightness": 80},
                capabilities=["power", "brightness"],
            ),
            SmartDevice(
                id="thermostat",
                name="Thermostat",
                type=DeviceType.THERMOSTAT,
                room="Living Room",
                state={"temperature": 72, "target_temperature": 70, "mode": "heat"},
                capabilities=["temperature", "target_temperature", "mode"],
            ),
            SmartDevice(
                id="front_door_lock",
                name="Front Door Lock",
                type=DeviceType.LOCK,
                room="Entryway",
                state={"lock_state": "locked"},
                capabilities=["lock", "unlock"],
            ),
        ]
        
        for device in mock_devices:
            self.devices[device.id] = device
        
        self._save_devices()
    
    def discover_devices(self) -> List[SmartDevice]:
        """
        Discover available smart home devices.
        
        Returns:
            List of discovered devices
        """
        return list(self.devices.values())
    
    def get_device(self, device_id: str) -> Optional[SmartDevice]:
        """Get device by ID."""
        return self.devices.get(device_id)
    
    def find_devices(
        self,
        name: Optional[str] = None,
        device_type: Optional[DeviceType] = None,
        room: Optional[str] = None,
    ) -> List[SmartDevice]:
        """
        Find devices by criteria.
        
        Args:
            name: Device name (partial match)
            device_type: Device type filter
            room: Room name filter
            
        Returns:
            List of matching devices
        """
        results = list(self.devices.values())
        
        if name:
            name_lower = name.lower()
            results = [d for d in results if name_lower in d.name.lower()]
        
        if device_type:
            results = [d for d in results if d.type == device_type]
        
        if room:
            room_lower = room.lower()
            results = [d for d in results if d.room and room_lower in d.room.lower()]
        
        return results
    
    def control_device(
        self,
        device_id: str,
        command: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Control a smart home device.
        
        Args:
            device_id: Device identifier
            command: Command to execute (turn_on, turn_off, set_temperature, etc.)
            params: Additional parameters (e.g., {"temperature": 72})
            
        Returns:
            True if command succeeded
        """
        device = self.get_device(device_id)
        if not device:
            return False
        
        params = params or {}
        command_lower = command.lower()
        
        # Handle different device types
        if device.type == DeviceType.LIGHT or device.type == DeviceType.SWITCH:
            if command_lower in ["turn_on", "on", "enable"]:
                device.state["power"] = "on"
                if "brightness" in params:
                    device.state["brightness"] = params["brightness"]
                self._save_devices()
                return True
            elif command_lower in ["turn_off", "off", "disable"]:
                device.state["power"] = "off"
                self._save_devices()
                return True
            elif command_lower == "set_brightness" and "brightness" in params:
                device.state["brightness"] = params["brightness"]
                if device.state.get("power") == "off":
                    device.state["power"] = "on"
                self._save_devices()
                return True
        
        elif device.type == DeviceType.THERMOSTAT:
            if command_lower == "set_temperature" and "temperature" in params:
                device.state["target_temperature"] = params["temperature"]
                self._save_devices()
                return True
            elif command_lower == "set_mode" and "mode" in params:
                device.state["mode"] = params["mode"]
                self._save_devices()
                return True
        
        elif device.type == DeviceType.LOCK:
            if command_lower in ["lock", "lock_door"]:
                device.state["lock_state"] = "locked"
                self._save_devices()
                return True
            elif command_lower in ["unlock", "unlock_door"]:
                device.state["lock_state"] = "unlocked"
                self._save_devices()
                return True
        
        return False
    
    def parse_command(self, natural_language: str) -> List[Dict[str, Any]]:
        """
        Parse natural language command into device actions.
        
        Args:
            natural_language: User's command in natural language
            
        Returns:
            List of action dicts: [{"device_id": "...", "command": "...", "params": {...}}]
        """
        text_lower = natural_language.lower()
        actions = []
        
        # Common patterns
        if "turn on" in text_lower or "switch on" in text_lower:
            # Find device
            if "light" in text_lower:
                devices = self.find_devices(device_type=DeviceType.LIGHT)
                if "living room" in text_lower:
                    devices = [d for d in devices if "living" in d.room.lower()]
                elif "bedroom" in text_lower:
                    devices = [d for d in devices if "bedroom" in d.room.lower()]
                
                for device in devices:
                    actions.append({
                        "device_id": device.id,
                        "command": "turn_on",
                        "params": {}
                    })
        
        elif "turn off" in text_lower or "switch off" in text_lower:
            if "light" in text_lower:
                devices = self.find_devices(device_type=DeviceType.LIGHT)
                if "living room" in text_lower:
                    devices = [d for d in devices if "living" in d.room.lower()]
                elif "bedroom" in text_lower:
                    devices = [d for d in devices if "bedroom" in d.room.lower()]
                
                for device in devices:
                    actions.append({
                        "device_id": device.id,
                        "command": "turn_off",
                        "params": {}
                    })
        
        elif "set temperature" in text_lower or "temperature to" in text_lower:
            # Extract temperature
            import re
            temp_match = re.search(r'(\d+)', text_lower)
            if temp_match:
                temp = int(temp_match.group(1))
                devices = self.find_devices(device_type=DeviceType.THERMOSTAT)
                for device in devices:
                    actions.append({
                        "device_id": device.id,
                        "command": "set_temperature",
                        "params": {"temperature": temp}
                    })
        
        elif "lock" in text_lower and "door" in text_lower:
            devices = self.find_devices(device_type=DeviceType.LOCK)
            for device in devices:
                actions.append({
                    "device_id": device.id,
                    "command": "lock",
                    "params": {}
                })
        
        elif "unlock" in text_lower and "door" in text_lower:
            devices = self.find_devices(device_type=DeviceType.LOCK)
            for device in devices:
                actions.append({
                    "device_id": device.id,
                    "command": "unlock",
                    "params": {}
                })
        
        return actions
    
    def get_smart_home_context(self) -> str:
        """
        Get formatted smart home context for RAG injection.
        
        Returns:
            Context string describing current device states
        """
        if not self.devices:
            return "[smart_home] No smart home devices configured."
        
        lines = ["=== SMART HOME STATUS ==="]
        
        # Group by room
        by_room: Dict[str, List[SmartDevice]] = {}
        for device in self.devices.values():
            room = device.room or "Other"
            if room not in by_room:
                by_room[room] = []
            by_room[room].append(device)
        
        for room, devices in by_room.items():
            lines.append(f"\n{room}:")
            for device in devices:
                lines.append(f"  - {device.to_context_string()}")
        
        lines.append("=== END SMART HOME ===")
        
        return "[smart_home] " + "\n".join(lines)
    
    def execute_natural_command(self, command: str) -> Dict[str, Any]:
        """
        Execute a natural language command.
        
        Args:
            command: Natural language command (e.g., "turn on living room light")
            
        Returns:
            Result dict with success status and message
        """
        actions = self.parse_command(command)
        
        if not actions:
            return {
                "success": False,
                "message": "Could not understand the command. Try: 'turn on/off [device]', 'set temperature to X', etc."
            }
        
        results = []
        for action in actions:
            success = self.control_device(
                action["device_id"],
                action["command"],
                action.get("params", {})
            )
            
            device = self.get_device(action["device_id"])
            if success:
                results.append(f"✓ {device.name}: {action['command']}")
            else:
                results.append(f"✗ {device.name}: Failed")
        
        return {
            "success": all(self.control_device(a["device_id"], a["command"], a.get("params", {})) for a in actions),
            "message": "\n".join(results),
            "actions": actions
        }


# Convenience function
def get_smart_home_context(config_dir: Optional[Path] = None) -> str:
    """Quick way to get smart home context string."""
    try:
        provider = SmartHomeProvider(config_dir=config_dir)
        return provider.get_smart_home_context()
    except Exception as e:
        return f"[smart_home] Error: {str(e)}"


if __name__ == "__main__":
    # Test smart home integration
    print("Testing Smart Home Integration...")
    print("=" * 50)
    
    provider = SmartHomeProvider()
    
    print("\nDiscovered devices:")
    for device in provider.discover_devices():
        print(f"  - {device.name} ({device.type.value}) in {device.room}")
    
    print("\n" + "=" * 50)
    print("CONTEXT FOR MODEL:")
    print("=" * 50)
    print(provider.get_smart_home_context())
    
    print("\n" + "=" * 50)
    print("Testing natural language commands:")
    print("=" * 50)
    
    test_commands = [
        "turn on living room light",
        "set temperature to 72",
        "turn off bedroom light",
    ]
    
    for cmd in test_commands:
        print(f"\nCommand: '{cmd}'")
        result = provider.execute_natural_command(cmd)
        print(f"Result: {result['message']}")
    
    print("\n" + "=" * 50)
    print("Updated context:")
    print("=" * 50)
    print(provider.get_smart_home_context())
