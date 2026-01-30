"""
Query routing logic - determines how to handle each query
"""
from typing import Optional, Dict, Tuple
from enum import Enum


class QueryComplexity(Enum):
    """Query complexity levels"""
    SIMPLE = "simple"          # Local model can handle
    MODERATE = "moderate"      # Local with context
    COMPLEX = "complex"        # May need cloud API
    SPECIALIZED = "specialized"  # Definitely needs cloud API


class QueryDomain(Enum):
    """Query domain categories"""
    WORK = "work"
    HOME = "home"
    HEALTH = "health"
    PERSONAL = "personal"
    GENERAL = "general"


class QueryRouter:
    """Route queries to appropriate handlers"""
    
    # Keywords for domain detection
    DOMAIN_KEYWORDS = {
        QueryDomain.WORK: [
            "meeting", "email", "project", "deadline", "schedule",
            "calendar", "client", "presentation", "report", "colleague",
            "boss", "office", "work", "task", "todo"
        ],
        QueryDomain.HOME: [
            "lights", "temperature", "thermostat", "lock", "door",
            "home", "house", "room", "bed", "security", "alarm",
            "garage", "kitchen", "living room", "bedroom"
        ],
        QueryDomain.HEALTH: [
            "workout", "exercise", "sleep", "steps", "calories",
            "health", "fitness", "medication", "vitamin", "run",
            "walk", "heart", "weight", "diet", "doctor"
        ],
        QueryDomain.PERSONAL: [
            "remind", "remember", "preference", "favorite", "hobby",
            "friend", "family", "birthday", "anniversary", "like"
        ],
    }
    
    # Keywords indicating complex queries
    COMPLEX_KEYWORDS = [
        "code", "program", "debug", "algorithm", "implement",
        "analyze", "research", "explain in detail", "compare",
        "legal", "medical advice", "financial", "investment"
    ]
    
    # Keywords for simple queries
    SIMPLE_KEYWORDS = [
        "what time", "weather", "turn on", "turn off", "set",
        "remind me", "hello", "hi", "thanks", "bye"
    ]
    
    def __init__(self):
        """Initialize router"""
        self.last_domain = QueryDomain.GENERAL
        self.context_stack = []
    
    def detect_domain(self, message: str) -> QueryDomain:
        """
        Detect which domain a message belongs to
        
        Args:
            message: User's input
            
        Returns:
            Detected domain
        """
        message_lower = message.lower()
        
        # Score each domain
        scores = {}
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in message_lower)
            scores[domain] = score
        
        # Get highest scoring domain
        max_domain = max(scores, key=scores.get)
        max_score = scores[max_domain]
        
        if max_score > 0:
            self.last_domain = max_domain
            return max_domain
        
        # Return last domain for context continuity, or GENERAL
        return self.last_domain if self.last_domain != QueryDomain.GENERAL else QueryDomain.GENERAL
    
    def assess_complexity(self, message: str) -> QueryComplexity:
        """
        Assess query complexity
        
        Args:
            message: User's input
            
        Returns:
            Complexity level
        """
        message_lower = message.lower()
        
        # Check for simple queries
        for keyword in self.SIMPLE_KEYWORDS:
            if keyword in message_lower:
                return QueryComplexity.SIMPLE
        
        # Check for complex/specialized queries
        for keyword in self.COMPLEX_KEYWORDS:
            if keyword in message_lower:
                return QueryComplexity.SPECIALIZED
        
        # Check message length and structure
        word_count = len(message.split())
        
        if word_count < 10:
            return QueryComplexity.SIMPLE
        elif word_count < 30:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.COMPLEX
    
    def should_use_cloud(self, message: str) -> Tuple[bool, str]:
        """
        Determine if query should be routed to cloud API
        
        Args:
            message: User's input
            
        Returns:
            Tuple of (should_use_cloud, reason)
        """
        complexity = self.assess_complexity(message)
        
        if complexity == QueryComplexity.SPECIALIZED:
            return True, "Query requires specialized knowledge"
        
        # Check for explicit coding requests
        if any(kw in message.lower() for kw in ["write code", "debug", "program"]):
            return True, "Coding request - routing to Claude"
        
        return False, ""
    
    def route(self, message: str) -> Dict:
        """
        Route a query and return routing decision
        
        Args:
            message: User's input
            
        Returns:
            Routing decision dict
        """
        domain = self.detect_domain(message)
        complexity = self.assess_complexity(message)
        use_cloud, cloud_reason = self.should_use_cloud(message)
        
        return {
            "domain": domain.value,
            "complexity": complexity.value,
            "use_cloud": use_cloud,
            "cloud_reason": cloud_reason,
            "adapter": f"{domain.value}_adapter" if domain != QueryDomain.GENERAL else None,
        }


# Example usage
if __name__ == "__main__":
    router = QueryRouter()
    
    test_queries = [
        "Turn off the lights",
        "Schedule a meeting for tomorrow at 2pm",
        "How did I sleep last night?",
        "What's my favorite restaurant?",
        "Write a Python function to sort a list",
        "What's the weather like?",
    ]
    
    for query in test_queries:
        result = router.route(query)
        print(f"\nQuery: {query}")
        print(f"  Domain: {result['domain']}")
        print(f"  Complexity: {result['complexity']}")
        print(f"  Use Cloud: {result['use_cloud']}")
        if result['use_cloud']:
            print(f"  Reason: {result['cloud_reason']}")
