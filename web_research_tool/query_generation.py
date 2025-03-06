"""
Functions for generating search queries.
"""

import yaml
import json
from typing import Dict, List, Any
from .models import SearchQuery, Source

def generate_initial_queries(anthropic_client: Any, research_task: Dict) -> List[SearchQuery]:
    """
    Use Claude to generate initial search queries based on the research task.
    
    Args:
        anthropic_client: Anthropic API client
        research_task: Dictionary containing research task details
        
    Returns:
        List of SearchQuery objects
    """
    prompt = f"""
    You are a research assistant helping to generate effective search queries.
    
    RESEARCH TASK:
    {json.dumps(research_task, indent=2)}
    
    Based on this research task, generate 3-5 specific search queries that would be most effective for finding relevant information.
    For each query, assign an importance score from 1-5 (5 being highest priority).
    You may optionally specify site restrictions for any query (like site:example.com).
    
    FORMAT YOUR RESPONSE AS A YAML LIST:
    
    ```yaml
    - query: "first search query"
      importance: 5
      site_restrict: "optional_site_restriction"
    - query: "second search query"
      importance: 3
    ```
    
    ONLY INCLUDE THE YAML IN YOUR RESPONSE, NO OTHER TEXT.
    """
    
    response = anthropic_client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=1000,
        temperature=0.2,
        system="You are a helpful research assistant that generates effective search queries.",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    result = response.content[0].text
    
    # Extract YAML content
    if "```yaml" in result:
        yaml_content = result.split("```yaml")[1].split("```")[0].strip()
    elif "```" in result:
        yaml_content = result.split("```")[1].strip()
    else:
        yaml_content = result.strip()
    
    try:
        queries_data = yaml.safe_load(yaml_content)
        return [SearchQuery(**query) for query in queries_data]
    except Exception as e:
        print(f"Error parsing Claude's query suggestions: {e}")
        print(f"Raw response: {result}")
        return [SearchQuery(query=f"Research {research_task.get('topic', 'topic')}")]

def generate_follow_up_queries(anthropic_client: Any, research_task: Dict, 
                              sources: List[Source], 
                              previous_queries: List[str]) -> List[SearchQuery]:
    """
    Generate follow-up search queries based on sources found so far.
    
    Args:
        anthropic_client: Anthropic API client
        research_task: Dictionary containing research task details
        sources: List of sources found so far
        previous_queries: List of queries already executed
        
    Returns:
        List of new SearchQuery objects
    """
    # Create a summary of sources found so far
    sources_summary = []
    for s in sources:
        sources_summary.append({
            "title": s.title,
            "url": s.url,
            "snippet": s.snippet,
            "relevance": s.relevance_score
        })
    
    prompt = f"""
    You are a research assistant helping to generate effective follow-up search queries.
    
    RESEARCH TASK:
    {json.dumps(research_task, indent=2)}
    
    PREVIOUS QUERIES:
    {json.dumps(previous_queries, indent=2)}
    
    SOURCES FOUND SO FAR:
    {json.dumps(sources_summary, indent=2)}
    
    Based on the research task and sources found so far, generate 2-3 new search queries that would help find additional relevant information.
    Focus on:
    1. Filling knowledge gaps in the current sources
    2. Exploring aspects of the topic not yet covered
    3. Finding more specific or authoritative sources
    
    For each query, assign an importance score from 1-5 (5 being highest priority).
    You may optionally specify site restrictions for any query (like site:example.com).
    
    FORMAT YOUR RESPONSE AS A YAML LIST:
    
    ```yaml
    - query: "first search query"
      importance: 5
      site_restrict: "optional_site_restriction"
    - query: "second search query"
      importance: 3
    ```
    
    ONLY INCLUDE THE YAML IN YOUR RESPONSE, NO OTHER TEXT.
    """
    
    response = anthropic_client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1000,
        temperature=0.3,
        system="You are a helpful research assistant that generates effective follow-up search queries.",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    result = response.content[0].text
    
    # Extract YAML content
    if "```yaml" in result:
        yaml_content = result.split("```yaml")[1].split("```")[0].strip()
    elif "```" in result:
        yaml_content = result.split("```")[1].strip()
    else:
        yaml_content = result.strip()
    
    try:
        queries_data = yaml.safe_load(yaml_content)
        return [SearchQuery(**query) for query in queries_data]
    except Exception as e:
        print(f"Error parsing Claude's follow-up query suggestions: {e}")
        print(f"Raw response: {result}")
        return []
