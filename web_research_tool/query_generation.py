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
    Enhanced to utilize short summaries and research topics.
    
    Args:
        anthropic_client: Anthropic API client
        research_task: Dictionary containing research task details
        sources: List of sources found so far
        previous_queries: List of queries already executed
        
    Returns:
        List of new SearchQuery objects
    """
    # Create a more detailed summary of sources found so far
    sources_summary = []
    collected_research_topics = []
    knowledge_gaps = []
    
    for s in sources:
        source_info = {
            "title": s.title,
            "url": s.url,
            "relevance": s.relevance_score
        }
        
        # Include short summaries if available
        if s.short_summary:
            source_info["key_points"] = s.short_summary
            
        # Extract and collect research topics
        if s.research_topics:
            source_info["suggested_topics"] = s.research_topics
            
            # Clean up the research topics and add to our collection
            topics = s.research_topics.replace('•', '').split('\n')
            topics = [t.strip() for t in topics if t.strip()]
            collected_research_topics.extend(topics)
            
        sources_summary.append(source_info)
        
    # Process collected research topics to identify potential knowledge gaps
    # Only include topics that appear in highly relevant sources (relevance > 0.7)
    high_relevance_topics = []
    for s in sources:
        if s.relevance_score > 0.7 and s.research_topics:
            topics = s.research_topics.replace('•', '').split('\n')
            topics = [t.strip() for t in topics if t.strip()]
            high_relevance_topics.extend(topics)
    
    prompt = f"""
    You are a research assistant helping to generate effective follow-up search queries.
    
    RESEARCH TASK:
    {json.dumps(research_task, indent=2)}
    
    PREVIOUS QUERIES:
    {json.dumps(previous_queries, indent=2)}
    
    SOURCES FOUND SO FAR:
    {json.dumps(sources_summary, indent=2)}
    
    SUGGESTED RESEARCH TOPICS FROM SOURCES:
    {json.dumps(high_relevance_topics, indent=2)}
    
    Based on the research task, sources found so far, and suggested research topics, generate 2-3 new search queries that would help find additional relevant information.
    
    Focus on:
    1. Filling knowledge gaps in the current sources
    2. Exploring the suggested research topics from highly relevant sources
    3. Finding more specific or authoritative sources
    4. Exploring aspects of the topic not yet covered
    
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

def extract_topics_from_sources(sources: List[Source]) -> List[str]:
    """
    Extract unique research topics from all sources.
    
    Args:
        sources: List of Source objects
        
    Returns:
        List of unique research topics
    """
    all_topics = []
    for source in sources:
        if source.research_topics:
            # Split by bullet points and clean up
            topics = source.research_topics.replace('•', '').split('\n')
            topics = [t.strip() for t in topics if t.strip()]
            all_topics.extend(topics)
    
    # Remove duplicates while preserving order
    unique_topics = []
    for topic in all_topics:
        normalized_topic = ' '.join(topic.lower().split())
        if not any(normalized_topic in ' '.join(t.lower().split()) for t in unique_topics):
            unique_topics.append(topic)
    
    return unique_topics
