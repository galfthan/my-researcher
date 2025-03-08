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
    query: str                          # The actual search query text to execute
    importance: int = 1                 # Priority ranking from 1-5, with 5 being highest priority
    site_restrict: Optional[str] = None # Optional site restriction (e.g., "site:example.com")

@dataclass
class Source:
    """
    Represents a source with its metadata and content.
    """
    url: str                 # URL of the source
    title: str               # Title of the source
    snippet: str             # Short snippet or description from search results
    content: str = ""        # Full extracted content of the source
    content_type: str = ""   # Content type (e.g., 'html', 'pdf')
    relevance_score: float = 0.0  # Score from 0.0 to 1.0 indicating relevance to research task
    short_summary: str = ""  # Brief bullet-point summary of key information (generated during relevance evaluation)
    research_topics: str = "" # Suggested follow-up research topics based on this source
    detailed_summary: str = "" # Comprehensive summary of the source (optional, may be generated later)
    
    def to_dict(self):
        """
        Convert source to dictionary.
        """
        return asdict(self)