"""
Tests for query router
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.router import QueryRouter, QueryDomain, QueryComplexity


class TestQueryRouter:
    """Tests for QueryRouter"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.router = QueryRouter()
    
    def test_detect_work_domain(self):
        """Test work domain detection"""
        messages = [
            "Schedule a meeting for tomorrow",
            "Send an email to the client",
            "What's on my calendar today?",
        ]
        for msg in messages:
            domain = self.router.detect_domain(msg)
            assert domain == QueryDomain.WORK, f"Expected WORK for: {msg}"
    
    def test_detect_home_domain(self):
        """Test home domain detection"""
        messages = [
            "Turn off the lights",
            "Set the thermostat to 72",
            "Lock the front door",
        ]
        for msg in messages:
            domain = self.router.detect_domain(msg)
            assert domain == QueryDomain.HOME, f"Expected HOME for: {msg}"
    
    def test_detect_health_domain(self):
        """Test health domain detection"""
        messages = [
            "How many steps did I take?",
            "Log my workout",
            "How did I sleep last night?",
        ]
        for msg in messages:
            domain = self.router.detect_domain(msg)
            assert domain == QueryDomain.HEALTH, f"Expected HEALTH for: {msg}"
    
    def test_simple_complexity(self):
        """Test simple query detection"""
        messages = [
            "Hi",
            "What time is it?",
            "Turn on the lights",
        ]
        for msg in messages:
            complexity = self.router.assess_complexity(msg)
            assert complexity == QueryComplexity.SIMPLE, f"Expected SIMPLE for: {msg}"
    
    def test_specialized_complexity(self):
        """Test specialized query detection"""
        messages = [
            "Write code to sort a list",
            "Debug this Python function",
        ]
        for msg in messages:
            complexity = self.router.assess_complexity(msg)
            assert complexity == QueryComplexity.SPECIALIZED, f"Expected SPECIALIZED for: {msg}"
    
    def test_cloud_routing(self):
        """Test cloud routing decisions"""
        # Should use cloud
        use_cloud, _ = self.router.should_use_cloud("Write code to implement a REST API")
        assert use_cloud == True
        
        # Should not use cloud
        use_cloud, _ = self.router.should_use_cloud("What's the weather?")
        assert use_cloud == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
