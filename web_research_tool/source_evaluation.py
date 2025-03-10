"""
Functions for evaluating source relevance.
"""

import re
import json
import time
from typing import Dict, Any, Tuple
import anthropic
from .models import Source

def evaluate_source_relevance(anthropic_client: Any, source: Source, research_task: Dict, 
                             verbose: bool = False) -> Tuple[float, str, str]:
    """
    Use Claude to evaluate the relevance of a source to the research task.
    Also returns a short summary and potential research topics.
    Processes the entire document, chunking if necessary for large documents.
    
    Args:
        anthropic_client: Anthropic API client
        source: Source object with content
        research_task: Dictionary containing research task details
        verbose: Whether to print detailed information
        
    Returns:
        Tuple of (relevance_score, short_summary, research_topics)
    """
    # For very large documents, we need to process them in chunks
    # but still maintain a comprehensive analysis
    # For extremely long ones we just read first parts
    # TODO: make this optional

    if len(source.content) > 30000:
        full_content = source.content[0:30000]
    else:
        full_content = source.content
    


    # If content is short enough, analyze it directly
    if len(full_content) < 12000:  # Claude Haiku has ~16K token limit
        return _evaluate_content_chunk(anthropic_client, source, full_content, research_task)
    else:
        # For longer documents, we'll analyze in chunks and combine scores
        if verbose:
            print(f"Document is large ({len(full_content)} chars), analyzing in chunks...")
        
        # Split into chunks with some overlap
        chunk_size = 10000
        overlap = 1000
        chunks = []
        
        for i in range(0, len(full_content), chunk_size - overlap):
            chunk = full_content[i:i + chunk_size]
            chunks.append(chunk)
            
        # Analyze each chunk
        chunk_scores = []
        chunk_summaries = []
        chunk_topics = []
        
        for i, chunk in enumerate(chunks):
            if verbose:
                print(f"Analyzing chunk {i+1}/{len(chunks)}...")
            
            score, summary, topics = _evaluate_content_chunk(
                anthropic_client, 
                source, 
                chunk, 
                research_task, 
                is_chunk=True, 
                chunk_info=f"Chunk {i+1} of {len(chunks)}"
            )
            chunk_scores.append(score)
            chunk_summaries.append(summary)
            chunk_topics.append(topics)
            
            # Add a small delay to avoid rate limiting
            time.sleep(0.5)
        
        # Combine scores - we'll take a weighted average that prioritizes 
        # the highest scores since relevant sections are most important
        chunk_scores.sort(reverse=True)
        if len(chunk_scores) > 2:
            # Give higher weight to the top scores
            final_score = (chunk_scores[0] * 0.5 + 
                          chunk_scores[1] * 0.3 + 
                          sum(chunk_scores[2:]) * 0.2 / max(1, len(chunk_scores) - 2))
        else:
            final_score = sum(chunk_scores) / len(chunk_scores)
            
        # Combine summaries and topics from the most relevant chunks
        zipped_chunks = list(zip(chunk_scores, chunk_summaries, chunk_topics))
        zipped_chunks.sort(reverse=True, key=lambda x: x[0])  # Sort by relevance score
        
        # Take summaries and topics from the top 2 chunks or all if fewer
        top_chunks = zipped_chunks[:min(2, len(zipped_chunks))]
        final_summary = " ".join([f"- {chunk[1]}" for chunk in top_chunks])
        final_topics = " ".join([chunk[2] for chunk in top_chunks])
            
        if verbose:
            print(f"Final combined relevance score: {final_score:.2f}")
            
        return final_score, final_summary, final_topics

def _evaluate_content_chunk(anthropic_client: Any, source: Source, content: str, 
                           research_task: Dict, is_chunk: bool = False,
                           chunk_info: str = "") -> Tuple[float, str, str]:
    """
    Evaluate a specific chunk of content for relevance.
    
    Args:
        anthropic_client: Anthropic API client
        source: Source object
        content: Content text to evaluate
        research_task: Research task details
        is_chunk: Whether this is a chunk of a larger document
        chunk_info: Information about the chunk position
        
    Returns:
        Tuple of (relevance_score, short_summary, research_topics)
    """
    chunk_context = f"\nThis is {chunk_info} from the full document." if is_chunk else ""
    
    # Truncate content if it's too long to avoid API errors
    MAX_CONTENT_LENGTH = 8000
    if len(content) > MAX_CONTENT_LENGTH:
        truncated_content = content[:MAX_CONTENT_LENGTH] + "... [content truncated due to length]"
    else:
        truncated_content = content
    
    prompt = f"""
    You are evaluating the relevance of a source for a research task.
    
    RESEARCH TASK:
    {json.dumps(research_task, indent=2)}
    
    SOURCE DETAILS:
    Title: {source.title}
    URL: {source.url}
    Snippet: {source.snippet}{chunk_context}
    
    DOCUMENT CONTENT:
    {truncated_content}
    
    I need three pieces of information:
    
    1. RELEVANCE SCORE: Evaluate how relevant this source is to the research task on a scale from 0.0 to 1.0:
    - 0.0: Completely irrelevant
    - 0.3: Tangentially related but not useful
    - 0.5: Somewhat relevant
    - 0.7: Relevant with good information
    - 0.9-1.0: Highly relevant, exactly what we need
    
    2. SUMMARY: Provide a very concise 3-bullet summary of the key points in this content relevant to the research task.
    
    3. RESEARCH TOPICS: Suggest 2-3 potential follow-up research topics or questions based on this content.
    
    FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
    SCORE: [number between 0.0 and 1.0]
    
    SUMMARY:
    • [First key point]
    • [Second key point]
    • [Third key point]
    
    TOPICS:
    • [First research topic]
    • [Second research topic]
    • [Third research topic]
    """
    
    try:
        response = anthropic_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=500,
            temperature=0.0,
            system="You evaluate source relevance and provide concise summaries.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        result = response.content[0].text.strip()
        
        # Extract the score
        score_match = re.search(r'SCORE:\s*(\d+\.\d+|\d+)', result)
        if score_match:
            score = float(score_match.group(1))
            # Ensure the score is between 0 and 1
            score = max(0.0, min(score, 1.0))
        else:
            print(f"Could not extract score from Claude's response: {result}")
            score = 0.5  # Default to neutral relevance
        
        # Extract the summary and topics
        summary_match = re.search(r'SUMMARY:(.*?)TOPICS:', result, re.DOTALL)
        topics_match = re.search(r'TOPICS:(.*)', result, re.DOTALL)
        
        summary = summary_match.group(1).strip() if summary_match else "No summary available."
        topics = topics_match.group(1).strip() if topics_match else "No research topics suggested."
        
        return score, summary, topics
    except Exception as e:
        print(f"Error evaluating source relevance: {e}")
        return 0.5, "Error generating summary.", "Error generating research topics."