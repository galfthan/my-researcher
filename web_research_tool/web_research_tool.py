"""
Main WebResearchTool class that orchestrates the research process.
"""

import os
import yaml
import json
import time
import anthropic
from datetime import datetime
from typing import List, Dict, Tuple, Set, Any, Optional
from googleapiclient.discovery import build

from .models import SearchQuery, Source
from .search import google_search
from .content_extraction import extract_content
from .source_evaluation import evaluate_source_relevance
from .query_generation import generate_initial_queries, generate_follow_up_queries
from .summarization import summarize_findings
from .output import save_source_content, prepare_for_claude

class WebResearchTool:
    def __init__(self, google_api_key: str, google_cse_id: str, anthropic_api_key: str, 
                 max_sources: int = 10, max_searches: int = 5, delay: float = 1.0,
                 verbose: bool = False, generate_detailed_summaries: bool = True):
        """
        Initialize the web research tool with API keys and configuration.
        
        Args:
            google_api_key: API key for Google Custom Search
            google_cse_id: Custom Search Engine ID
            anthropic_api_key: API key for Anthropic Claude API
            max_sources: Maximum number of sources to return
            max_searches: Maximum number of search iterations
            delay: Delay between API requests to avoid rate limiting
            verbose: Whether to print detailed progress information
            generate_detailed_summaries: Whether to generate detailed summaries for each source
        """
        self.google_api_key = google_api_key
        self.google_cse_id = google_cse_id
        self.anthropic_api_key = anthropic_api_key
        self.max_sources = max_sources
        self.max_searches = max_searches
        self.delay = delay
        self.verbose = verbose
        self.generate_detailed_summaries = generate_detailed_summaries
        
        # Initialize the Google Custom Search API client
        self.google_service = build("customsearch", "v1", developerKey=self.google_api_key)
        
        # Initialize the Anthropic Claude client
        self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
        
        self.sources = []
        self.search_queries = []
        self.completed_queries = set()
        
        # Print configuration if verbose
        if self.verbose:
            print(f"Web Research Tool Configuration:")
            print(f"- Max sources: {self.max_sources}")
            print(f"- Max searches: {self.max_searches}")
            print(f"- Request delay: {self.delay}s")
            print(f"- Generate detailed summaries: {self.generate_detailed_summaries}")
            print(f"- Google CSE ID: {self.google_cse_id[:5]}...{self.google_cse_id[-5:]}")
            print(f"- Using Anthropic API key: {self.anthropic_api_key[:5]}...{self.anthropic_api_key[-5:]}")
    
    def parse_yaml_request(self, yaml_request: str) -> Dict:
        """
        Parse the YAML request from Claude into a structured format.
        
        Args:
            yaml_request: YAML string describing the research task
            
        Returns:
            Dictionary with parsed research parameters
        """
        try:
            research_request = yaml.safe_load(yaml_request)
            return research_request
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML request: {e}")
    
    def conduct_research(self, yaml_request: str, output_dir: str = "research_output", 
                        generate_detailed_summaries: Optional[bool] = None) -> Tuple[str, List[str]]:
        """
        Main method to conduct the research based on the YAML request.
        
        Args:
            yaml_request: YAML string describing the research task
            output_dir: Directory to save output files
            generate_detailed_summaries: Override the class setting for detailed summaries
            
        Returns:
            Tuple of (research summary, list of source file paths)
        """
        # Allow overriding the summary generation option for this run
        if generate_detailed_summaries is not None:
            self.generate_detailed_summaries = generate_detailed_summaries
            
        # Create a timestamp-based output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{output_dir}_{timestamp}"
        
        # Parse the YAML request
        research_task = self.parse_yaml_request(yaml_request)
        
        print(f"Starting research on: {research_task.get('topic', 'Research Topic')}")
        
        # Generate initial search queries
        self.search_queries = generate_initial_queries(self.anthropic_client, research_task)
        print(f"Generated {len(self.search_queries)} initial search queries")
        
        # Sort queries by importance
        self.search_queries.sort(key=lambda q: q.importance, reverse=True)
        
        search_iteration = 0
        while search_iteration < self.max_searches and self.search_queries:
            # Get the next query
            current_query = self.search_queries.pop(0)
            
            # Skip if we've already processed this query
            query_key = f"{current_query.query}:{current_query.site_restrict}"
            if query_key in self.completed_queries:
                continue
            
            print(f"\nExecuting search query ({search_iteration + 1}/{self.max_searches}): {current_query.query}")
            if current_query.site_restrict:
                print(f"Site restriction: {current_query.site_restrict}")
            
            # Perform the search
            search_results = google_search(
                self.google_service, 
                self.google_cse_id,
                current_query.query, 
                current_query.site_restrict,
                delay=self.delay
            )
            print(f"Found {len(search_results)} results")
            
            # Process each search result
            for result in search_results[:5]:  # Limit to top 5 results per query
                url = result.get('link')
                
                # Skip if we already have this source
                if any(s.url == url for s in self.sources):
                    continue
                
                print(f"Processing: {url}")
                
                # Create a source object
                source = Source(
                    url=url,
                    title=result.get('title', 'No title'),
                    snippet=result.get('snippet', 'No snippet')
                )
                
                # Extract the content
                content, content_type = extract_content(url)
                source.content = content
                source.content_type = content_type
                
                # Skip if we couldn't extract content
                if 'ERROR' in content:
                    print(f"Skipping due to extraction error")
                    continue
                
                # Evaluate relevance and get short summary and research topics
                relevance_score, short_summary, research_topics = evaluate_source_relevance(
                    self.anthropic_client, 
                    source, 
                    research_task,
                    self.verbose
                )
                
                source.relevance_score = relevance_score
                source.short_summary = short_summary
                source.research_topics = research_topics
                
                print(f"Relevance score: {source.relevance_score}")
                
                # Add to our sources if it's relevant enough
                if source.relevance_score >= 0.5:
                    self.sources.append(source)
                    print(f"Added to sources (total: {len(self.sources)})")
                    
                    # Extract research topics for potential follow-up queries
                    if source.research_topics and self.verbose:
                        print(f"Suggested research topics: {source.research_topics}")
                
                # Break if we have enough sources
                if len(self.sources) >= self.max_sources:
                    break
            
            # Mark this query as completed
            self.completed_queries.add(query_key)
            
            # Generate follow-up queries if needed
            if len(self.sources) < self.max_sources and search_iteration < self.max_searches - 1:
                previous_queries = [q.query for q in 
                                   [sq for sq_key, sq in 
                                    [(f"{q.query}:{q.site_restrict}", q) for q in self.search_queries] 
                                    if sq_key in self.completed_queries]]
                
                follow_up_queries = generate_follow_up_queries(
                    self.anthropic_client,
                    research_task, 
                    self.sources,
                    previous_queries
                )
                
                # Add new queries to our list
                for query in follow_up_queries:
                    query_key = f"{query.query}:{query.site_restrict}"
                    if query_key not in self.completed_queries:
                        self.search_queries.append(query)
                
                # Re-sort queries by importance
                self.search_queries.sort(key=lambda q: q.importance, reverse=True)
                
                print(f"Generated {len(follow_up_queries)} follow-up queries")
            
            search_iteration += 1
        
        # Sort sources by relevance
        self.sources.sort(key=lambda s: s.relevance_score, reverse=True)
        
        # Keep only the top sources
        self.sources = self.sources[:self.max_sources]
        
        print(f"\nResearch completed. Found {len(self.sources)} relevant sources.")
        
        # Generate detailed summaries if enabled
        if self.generate_detailed_summaries:
            print("Generating detailed research summary...")
            summary = summarize_findings(self.anthropic_client, research_task, self.sources, self.verbose)
        else:
            print("Generating basic research summary with short bullet points...")
            # Use the short summaries already generated during evaluation
            summary = self._generate_basic_summary(research_task)
        
        # Save source content
        print(f"Saving source content to {output_dir}...")
        file_paths = save_source_content(self.sources, output_dir)
        
        # Prepare output for Claude
        output_text = prepare_for_claude(research_task, self.sources, summary, output_dir)
        
        # Save the output
        output_file = os.path.join(output_dir, "research_summary.md")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output_text)
        
        print(f"Research output saved to {output_file}")
        
        return output_text, file_paths
    
    def _generate_basic_summary(self, research_task: Dict) -> str:
        """
        Generate a basic summary using the short summaries already collected.
        
        Args:
            research_task: Dictionary containing research task details
            
        Returns:
            Basic research summary
        """
        summary = [
            f"# Research Summary: {research_task.get('topic', 'Research Topic')}",
            "\n## Key Sources and Findings\n"
        ]
        
        for i, source in enumerate(self.sources):
            summary.append(f"### {i+1}. {source.title}")
            summary.append(f"**URL:** {source.url}")
            summary.append(f"**Relevance Score:** {source.relevance_score:.2f}\n")
            summary.append(f"**Key Points:**\n{source.short_summary}\n")
        
        # Collect unique research topics from all sources
        all_topics = []
        for source in self.sources:
            if source.research_topics:
                # Split by bullet points and clean up
                topics = [t.strip() for t in source.research_topics.replace('•', '').split('\n') if t.strip()]
                all_topics.extend(topics)
        
        # Remove duplicates while preserving order
        unique_topics = []
        for topic in all_topics:
            if topic not in unique_topics:
                unique_topics.append(topic)
        
        if unique_topics:
            summary.append("\n## Suggested Further Research\n")
            for topic in unique_topics[:10]:  # Limit to top 10 topics
                summary.append(f"- {topic}")
        
        summary.append("\n\n*Note: This is a basic summary. For detailed analysis, use the `generate_detailed_summaries=True` option.*")
        
        return "\n".join(summary)