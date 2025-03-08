"""
Web Research Tool - A tool for conducting comprehensive web research using Google Search and Claude AI.
"""

from .models import SearchQuery, Source
from .web_research_tool import WebResearchTool

__all__ = ['WebResearchTool', 'SearchQuery', 'Source']