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
    Ensures all sources and their individual summaries are included in the final report.
    
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
        if verbose:
            print(f"Processing source {i+1}/{len(sources)}: {source.title}")
            
        # For all documents, generate a summary
        if len(source.content) > 20000:
            if verbose:
                print(f"Generating summary for large source {i+1}: {source.title}")
            
            summary = _generate_source_summary(anthropic_client, source, research_task)
        else:
            # For shorter documents, still generate a summary
            summary = _generate_source_summary(anthropic_client, source, research_task)
            
        source_summaries.append({
            "index": i + 1,
            "title": source.title,
            "url": source.url,
            "relevance": source.relevance_score,
            "summary": summary
        })
    
    # Generate the main research summary (executive summary, key findings, etc.)
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
    Generate a research summary that includes executive summary, key findings,
    and references to the numbered source list.
    
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
    
    # Include only essential source info for the main summary generation
    source_ref_list = []
    for src in source_summaries:
        source_ref_list.append({
            "index": src["index"],
            "title": src["title"],
            "url": src["url"],
            "relevance": src["relevance"]
        })
    
    prompt = f"""
    You are a research assistant summarizing findings from web research.{batch_context}
    
    RESEARCH TASK:
    {json.dumps(research_task, indent=2)}
    
    SOURCES FOUND:
    {json.dumps(source_ref_list, indent=2)}
    
    Please provide a comprehensive research summary that:
    1. Provides an overview of the topic and key findings
    2. Highlights the most important information from the sources
    3. Notes any gaps or areas for further research
    
    FORMAT:
    - Start with an "Executive Summary" (2-3 paragraphs)
    - Include a "Key Findings" section with the most important information
    - Whenever you reference information from a specific source, include the source number in brackets, e.g., [1], [3]
    - Include a brief "Sources" section that lists ONLY the numbered references and their URLs (detailed summaries will be added separately)
    - End with "Suggested Next Steps" for further research
    
    IMPORTANT:
    - Refer to sources by their number throughout your summary (e.g., "According to Source [3]...")
    - Your executive summary and key findings should reference the source numbers to support claims
    - Do NOT include the detailed source summaries - these will be appended separately
    
    YOUR RESPONSE SHOULD BE WELL-FORMATTED AND READY TO PRESENT TO THE USER.
    """
    
    response = anthropic_client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=4000,
        temperature=0.2,
        system="You are a helpful research assistant summarizing web research findings.",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    # Get the main summary
    main_summary = response.content[0].text
    
    # Now append the detailed source summaries
    source_detail_section = "\n\n## Detailed Source Summaries\n\n"
    for src in source_summaries:
        source_detail_section += f"### Source [{src['index']}]: {src['title']}\n"
        source_detail_section += f"**URL:** {src['url']}\n"
        source_detail_section += f"**Relevance Score:** {src['relevance']:.2f}\n\n"
        source_detail_section += f"{src['summary']}\n\n"
        source_detail_section += "---\n\n"
    
    # Combine the main summary with the source details
    complete_summary = main_summary + "\n\n" + source_detail_section
    
    return complete_summary

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
    if len(source.content) > 20000:
        # Split into chunks with overlap
        chunk_size = 18000
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
