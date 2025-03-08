"""
Functions for formatting and saving research output.
"""

import os
from typing import Dict, List, Tuple
from datetime import datetime
from urllib.parse import urlparse
from .models import Source

def save_source_content(sources: List[Source], output_dir: str) -> List[str]:
    """
    Save the content of each source to a file and return file paths.
    
    Args:
        sources: List of sources
        output_dir: Directory to save files in
        
    Returns:
        List of saved file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    file_paths = []
    
    for i, source in enumerate(sources):
        # Create a safe filename from the URL
        parsed_url = urlparse(source.url)
        domain = parsed_url.netloc.replace(".", "_")
        path = parsed_url.path.replace("/", "_")
        if not path:
            path = "_index"
        
        # Determine file extension based on content type
        if source.content_type == 'pdf':
            ext = '.pdf'
        else:
            ext = '.txt'
        
        # Create filename
        filename = f"{i+1:02d}_{domain}{path[:50]}{ext}"
        filepath = os.path.join(output_dir, filename)
        
        # Write content to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"SOURCE: {source.title}\n")
            f.write(f"URL: {source.url}\n")
            f.write(f"RELEVANCE: {source.relevance_score}\n")
            
            # Add short summary if available
            if source.short_summary:
                f.write(f"\nKEY POINTS:\n{source.short_summary}\n")
            
            # Add research topics if available
            if source.research_topics:
                f.write(f"\nSUGGESTED RESEARCH TOPICS:\n{source.research_topics}\n")
            
            f.write("\n" + "="*80 + "\n\n")
            f.write(source.content)
        
        file_paths.append(filepath)
    
    return file_paths

def prepare_for_claude(research_task: Dict, sources: List[Source], summary: str, output_dir: str) -> str:
    """
    Prepare the research output in a format suitable for Claude Web.
    
    Args:
        research_task: Dictionary containing research task details
        sources: List of sources
        summary: Research summary
        output_dir: Directory where source files are saved
        
    Returns:
        Text output for Claude Web
    """
    output = []
    
    # Add header
    output.append(f"# Research Results: {research_task.get('topic', 'Research Topic')}")
    output.append(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    # Add summary
    output.append(summary)
    output.append("\n" + "="*80 + "\n")
    
    # Add information about source files with full URLs
    output.append("\n## Source Files and URLs")
    output.append(f"Source content has been saved to: {output_dir}\n")
    
    # List all sources with their full URLs for extended summary reports (short ones already has this info in summary)
    if any(s.detailed_summary for s in sources):
        output.append("Full source URLs for easy reference:")
        for i, source in enumerate(sources):
            output.append(f"{i+1}. [{source.title}]({source.url})")
            output.append(f"   - Relevance score: {source.relevance_score:.2f}")
        
            # Add short summary if available
            if source.short_summary:
                output.append(f"   - Key points: {source.short_summary.replace('•', '-').strip()}")
        
            output.append(f"   - URL: {source.url}")
            output.append("")
    
    # Add instructions for using with Claude
    output.append("\n## Using These Results with Claude")
    output.append("Copy this summary and upload the source files to continue your research conversation with Claude.")
    
    # Add option to get detailed summaries if they weren't generated
    if not any(s.detailed_summary for s in sources):
        output.append("\n## Generating Detailed Summaries")
        output.append("This research was conducted in quick mode. To generate detailed summaries, run again with:")
        output.append("```")
        output.append("python -m web_research_tool.main --input your_request.yaml --detailed-summaries --output research_output")
        output.append("```")
        output.append("Or add `detailed_summaries: true` to your YAML request.")
    
    return "\n".join(output)