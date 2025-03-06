"""
Data models for the Web Research Tool.
"""

from typing import Optional
from dataclasses import dataclass, asdict

@dataclass
class SearchQuery:
    """
    Represents a search query with optional site restriction and importance ranking.
    """
    query: str
    importance: int = 1
    site_restrict: Optional[str] = None

@dataclass
class Source:
    """
    Represents a source with its metadata and content.
    """
    url: str
    title: str
    snippet: str
    content: str = ""
    content_type: str = ""
    relevance_score: float = 0.0
    
    def to_dict(self):
        """
        Convert source to dictionary.
        """
        return asdict(self)
