"""
Functions for summarizing research findings.
"""

import json
from typing import Dict, List, Any
from .models import Source

def summarize_findings(anthropic_client: Any, research_task: Dict, sources: List[Source],
                       verbose: bool = False) -> str:
    """
    Use Claude to generate a summary of the research findings.
    For sources with longer content, this implementation processes them
    in a way that captures the full depth of the information.
    Ensures ALL sources are included in the summary.
    
    Args:
        anthropic_client: Anthropic API client
        research_task: Dictionary containing research task details
        sources: List of sources found
        verbose: Whether to print detailed information
        
    Returns:
        Summary of the research findings
    """
    # First pass: Create individual source summaries
    source_summaries = []
    
    for i, source in enumerate(sources):
        if len(source.content) > 8000:
            # For longer documents, generate a standalone summary first
            if verbose:
                print(f"Generating summary for large source {i+1}: {source.title}")
            
            summary = _generate_source_summary(anthropic_client, source, research_task)
            source_summaries.append({
                "index": i + 1,
                "title": source.title,
                "url": source.url,
                "relevance": source.relevance_score,
                "summary": summary
            })
        else:
            # For shorter documents, include full content
            source_summaries.append({
                "index": i + 1,
                "title": source.title,
                "url": source.url,
                "relevance": source.relevance_score,
                "content": source.content
            })
    
    # If we have a large number of sources, we may need to process them in batches
    # to avoid exceeding Claude's context window
    MAX_SOURCES_PER_BATCH = 7
    if len(source_summaries) > MAX_SOURCES_PER_BATCH:
        if verbose:
            print(f"Processing {len(source_summaries)} sources in batches for summary generation")
        
        return _batch_process_summaries(
            anthropic_client, 
            research_task, 
            source_summaries, 
            MAX_SOURCES_PER_BATCH,
            verbose
        )
    else:
        # Second pass: Generate comprehensive research summary for a manageable number of sources
        return _generate_research_summary(anthropic_client, research_task, source_summaries)

def _batch_process_summaries(anthropic_client: Any, research_task: Dict, 
                            source_summaries: List[Dict], batch_size: int,
                            verbose: bool = False) -> str:
    """
    Process a large number of sources in batches to generate summaries,
    then combine them into a final research summary.
    
    Args:
        anthropic_client: Anthropic API client
        research_task: Dictionary containing research task details
        source_summaries: List of source summary dictionaries
        batch_size: Maximum number of sources to process in a single batch
        verbose: Whether to print detailed information
        
    Returns:
        Combined research summary
    """
    # Sort sources by relevance
    sorted_summaries = sorted(source_summaries, key=lambda s: s.get('relevance', 0), reverse=True)
    
    # Process in batches
    batch_summaries = []
    for i in range(0, len(sorted_summaries), batch_size):
        batch = sorted_summaries[i:i+batch_size]
        if verbose:
            print(f"Processing batch {i//batch_size + 1} with sources {i+1}-{min(i+batch_size, len(sorted_summaries))}")
        
        # Generate summary for this batch
        batch_summary = _generate_research_summary(
            anthropic_client, 
            research_task, 
            batch,
            is_batch=True,
            batch_info=f"Batch {i//batch_size + 1}/{(len(sorted_summaries) + batch_size - 1)//batch_size}"
        )
        
        batch_summaries.append(batch_summary)
    
    # Combine batch summaries into a final summary
    combined_summary = "\n\n".join(batch_summaries)
    
    # Generate a final integrated summary
    final_prompt = f"""
    You are a research assistant creating a unified research summary from multiple batch summaries.
    
    RESEARCH TASK:
    {json.dumps(research_task, indent=2)}
    
    BATCH SUMMARIES:
    {combined_summary}
    
    Please integrate these batch summaries into a single coherent research summary. 
    
    Your final summary MUST include:
    1. An executive summary (2-3 paragraphs)
    2. A "Key Findings" section with the most important information
    3. A "Sources" section that lists ALL sources from ALL batches with:
       - Full URL of each source
       - Relevance score
       - Comprehensive description (1-2 paragraphs minimum per source)
    4. "Suggested Next Steps" for further research
    
    Make sure NO sources are omitted - you must include EVERY source from ALL batches.
    
    FORMAT your response to be well-organized with clear sections and subsections.
    """
    
    if verbose:
        print("Generating final integrated summary from all batches...")
    
    response = anthropic_client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=8000,  # Increased for comprehensive summary
        temperature=0.2,
        system="You are a helpful research assistant creating a comprehensive integrated research summary.",
        messages=[
            {"role": "user", "content": final_prompt}
        ]
    )
    
    return response.content[0].text

def _generate_research_summary(anthropic_client: Any, research_task: Dict, 
                             source_summaries: List[Dict], is_batch: bool = False, 
                             batch_info: str = "") -> str:
    """
    Generate a research summary for a set of sources.
    
    Args:
        anthropic_client: Anthropic API client
        research_task: Dictionary containing research task details
        source_summaries: List of source summary dictionaries
        is_batch: Whether this is processing a batch of a larger set
        batch_info: Information about the batch position
        
    Returns:
        Research summary
    """
    batch_context = f"\nThis is {batch_info} of sources being processed." if is_batch else ""
    
    prompt = f"""
    You are a research assistant summarizing findings from web research.{batch_context}
    
    RESEARCH TASK:
    {json.dumps(research_task, indent=2)}
    
    SOURCES FOUND:
    {json.dumps(source_summaries, indent=2)}
    
    Please provide a comprehensive research summary that:
    1. Provides an overview of the topic and key findings
    2. Highlights the most important information from each relevant source
    3. Notes any gaps or areas for further research
    4. Lists ALL sources in order of relevance with detailed descriptions
    
    FORMAT:
    - Start with an executive summary (2-3 paragraphs)
    - Include a "Key Findings" section with the most important information
    - Include a detailed "Sources" section that:
      * Lists EVERY source with its full URL (not abbreviated)
      * Provides its relevance score
      * Includes a substantial summary (at least 1-2 paragraphs per source)
      * Highlights key facts and insights from each source
    - End with "Suggested Next Steps" for further research
    
    IMPORTANT:
    - Be detailed and comprehensive in your summaries
    - Include full URLs of each source for easy reference
    - Make sure each source summary gives a thorough overview of what the source contains
    - Do NOT omit any sources - ALL sources must be included in your summary
    
    YOUR RESPONSE SHOULD BE WELL-FORMATTED AND READY TO PRESENT TO THE USER.
    """
    
    # Since the main summary might be complex, we use a more robust model with increased token allocation
    response = anthropic_client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=8000,  # Increased from 4000 to accommodate more sources
        temperature=0.2,
        system="You are a helpful research assistant summarizing web research findings.",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.content[0].text

def _generate_source_summary(anthropic_client: Any, source: Source, research_task: Dict) -> str:
    """
    Generate a detailed summary of a single large source.
    
    Args:
        anthropic_client: Anthropic API client
        source: Source object with content
        research_task: Dictionary containing research task details
        
    Returns:
        Detailed summary of the source
    
    Raises:
        Exception: If there's an error generating the summary, provides a fallback
    """
    # For very large content, process in chunks
    if len(source.content) > 12000:
        # Split into chunks with overlap
        chunk_size = 10000
        overlap = 1000
        chunks = []
        
        for i in range(0, len(source.content), chunk_size - overlap):
            chunk = source.content[i:i + chunk_size]
            chunks.append(chunk)
        
        # Generate summary for each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            print(f"Summarizing chunk {i+1}/{len(chunks)} for source: {source.title}")
            
            prompt = f"""
            You are summarizing a portion of a document for research purposes.
            
            RESEARCH TASK:
            {json.dumps(research_task, indent=2)}
            
            SOURCE:
            Title: {source.title}
            URL: {source.url}
            
            This is chunk {i+1} of {len(chunks)} from the document.
            
            DOCUMENT CHUNK CONTENT:
            {chunk}
            
            Provide a comprehensive and detailed summary of the key information in this document chunk 
            that is relevant to the research task. Be thorough in capturing facts, data, statistics, 
            methodology, findings, conclusions, and insights. Include specific details where possible.
            """
            
            try:
                response = anthropic_client.messages.create(
                    model="claude-3-5-haiku-20241022",
                    max_tokens=3000,  # Increased from 2000
                    temperature=0.1,
                    system="You are a helpful research assistant extracting key information from documents.",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                chunk_summaries.append(response.content[0].text)
            except Exception as e:
                print(f"Error summarizing chunk {i+1}: {e}")
                # Provide a basic fallback summary for this chunk
                fallback_summary = f"[Content from chunk {i+1} could not be summarized due to API error: {e}]"
                chunk_summaries.append(fallback_summary)
        
        # Combine chunk summaries
        combined_summary = "\n\n".join([f"--- Chunk {i+1} Summary ---\n{summary}" 
                                      for i, summary in enumerate(chunk_summaries)])
        
        # Generate a unified summary from the chunk summaries
        final_prompt = f"""
        You are creating a unified summary of a document based on summaries of different chunks.
        
        RESEARCH TASK:
        {json.dumps(research_task, indent=2)}
        
        SOURCE:
        Title: {source.title}
        URL: {source.url}
        
        CHUNK SUMMARIES:
        {combined_summary}
        
        Create a comprehensive and detailed unified summary of this document that captures all the key 
        information from the different chunks that is relevant to the research task. Include specific 
        facts, figures, methodology, and conclusions. Be thorough while still eliminating redundancies 
        and organizing the information logically. Your summary should be substantial enough to give readers
        a complete understanding of the document's relevant content.
        """
        
        try:
            response = anthropic_client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=4000,  # Increased from 2500
                temperature=0.1,
                system="You are a helpful research assistant creating unified document summaries.",
                messages=[
                    {"role": "user", "content": final_prompt}
                ]
            )
            
            return response.content[0].text
        except Exception as e:
            print(f"Error generating unified summary: {e}")
            # Provide a simple combined summary of the chunks
            return "\n\n".join([f"### Chunk {i+1} Summary\n{summary}" 
                              for i, summary in enumerate(chunk_summaries)])
    else:
        # For content that fits within context window
        prompt = f"""
        You are summarizing a document for research purposes.
        
        RESEARCH TASK:
        {json.dumps(research_task, indent=2)}
        
        SOURCE:
        Title: {source.title}
        URL: {source.url}
        
        DOCUMENT CONTENT:
        {source.content}
        
        Provide a comprehensive and detailed summary of the key information in this document
        that is relevant to the research task. Be thorough in capturing facts, data, statistics, 
        methodology, findings, conclusions, and insights. Include specific details where possible.
        Your summary should be substantial enough to give readers a complete understanding of 
        the document's relevant content.
        """
        
        try:
            response = anthropic_client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=4000,  # Increased from 2500
                temperature=0.1,
                system="You are a helpful research assistant summarizing documents.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text
        except Exception as e:
            print(f"Error summarizing document: {e}")
            # Provide a basic fallback summary
            return f"[Document summary could not be generated due to API error: {e}]\n\nTitle: {source.title}\nURL: {source.url}\n\nThis document appears relevant to the research task but could not be summarized automatically."
