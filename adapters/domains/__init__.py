"""
Domain adapters for Roku AI.

Domain adapters provide domain-specific context preparation and response handling.
"""

from adapters.domains.base import DomainAdapter
from adapters.domains.work import WorkAdapter
from adapters.domains.home import HomeAdapter
from adapters.domains.health import HealthAdapter
from adapters.domains.personal import PersonalAdapter

__all__ = [
    "DomainAdapter",
    "WorkAdapter",
    "HomeAdapter",
    "HealthAdapter",
    "PersonalAdapter",
]
