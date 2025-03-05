import os
import yaml
import json
import argparse
import requests
import anthropic
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from urllib.parse import urlparse, parse_qs
import PyPDF2
import io
import time
import sys
import platform
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from pprint import pprint
from datetime import datetime

@dataclass
class SearchQuery:
    query: str
    importance: int = 1
    site_restrict: Optional[str] = None

@dataclass
class Source:
    url: str
    title: str
    snippet: str
    content: str = ""
    content_type: str = ""
    relevance_score: float = 0.0
    
    def to_dict(self):
        return asdict(self)

class WebResearchTool:
    def __init__(self, google_api_key: str, google_cse_id: str, anthropic_api_key: str, 
                 max_sources: int = 10, max_searches: int = 5):
        """
        Initialize the web research tool with API keys and configuration.
        
        Args:
            google_api_key: API key for Google Custom Search
            google_cse_id: Custom Search Engine ID
            anthropic_api_key: API key for Anthropic Claude API
            max_sources: Maximum number of sources to return
            max_searches: Maximum number of search iterations
        """
        self.google_api_key = google_api_key
        self.google_cse_id = google_cse_id
        self.anthropic_api_key = anthropic_api_key
        self.max_sources = max_sources
        self.max_searches = max_searches
        
        # Initialize the Google Custom Search API client
        self.google_service = build("customsearch", "v1", developerKey=self.google_api_key)
        
        # Initialize the Anthropic Claude client
        self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
        
        self.sources = []
        self.search_queries = []
        self.completed_queries = set()
        
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
    
    def generate_initial_queries(self, research_task: Dict) -> List[SearchQuery]:
        """
        Use Claude to generate initial search queries based on the research task.
        
        Args:
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
        
        response = self.anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
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
    
    def google_search(self, query: str, site_restrict: Optional[str] = None, start_index: int = 1, 
                   delay: float = 1.0, max_retries: int = 3) -> List[Dict]:
        """
        Perform a Google search using the Custom Search API with rate limiting and retries.
        
        Args:
            query: Search query string
            site_restrict: Optional site restriction (e.g., "site:example.com")
            start_index: Starting index for pagination
            delay: Time to wait between requests in seconds
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of search result items
        """
        full_query = query
        if site_restrict:
            full_query = f"{query} {site_restrict}"
        
        for attempt in range(max_retries):
            try:
                # Implement rate limiting
                if attempt > 0:
                    sleep_time = delay * (2 ** attempt)  # Exponential backoff
                    print(f"Retrying in {sleep_time:.2f} seconds (attempt {attempt+1}/{max_retries})...")
                    time.sleep(sleep_time)
                
                result = self.google_service.cse().list(
                    q=full_query,
                    cx=self.google_cse_id,
                    start=start_index
                ).execute()
                
                # Add a small delay to avoid hitting rate limits
                time.sleep(delay)
                
                if 'items' in result:
                    return result['items']
                return []
                
            except Exception as e:
                if "quota" in str(e).lower() and attempt < max_retries - 1:
                    print(f"Quota error: {e}")
                    continue  # Retry
                elif "rate limit" in str(e).lower() and attempt < max_retries - 1:
                    print(f"Rate limit error: {e}")
                    continue  # Retry
                else:
                    print(f"Error during Google search: {e}")
                    return []
    
    def extract_content(self, url: str) -> Tuple[str, str]:
        """
        Extract content from a URL (handles both HTML and PDF).
        
        Args:
            url: URL to fetch content from
            
        Returns:
            Tuple of (content, content_type)
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            content_type = response.headers.get('Content-Type', '').lower()
            
            if 'application/pdf' in content_type:
                # Handle PDF
                try:
                    pdf_file = io.BytesIO(response.content)
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    content = ""
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        content += page.extract_text() + "\n"
                    return content, 'pdf'
                except Exception as e:
                    print(f"Error extracting PDF content: {e}")
                    return f"[PDF EXTRACTION ERROR: {e}]", 'pdf'
            else:
                # Handle HTML
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.extract()
                
                # Get text
                text = soup.get_text(separator=' ', strip=True)
                
                # Remove extra whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                return text, 'html'
                
        except Exception as e:
            print(f"Error fetching URL {url}: {e}")
            return f"[CONTENT EXTRACTION ERROR: {e}]", 'error'
    
    def evaluate_source_relevance(self, source: Source, research_task: Dict) -> float:
        """
        Use Claude to evaluate the relevance of a source to the research task.
        
        Args:
            source: Source object with content
            research_task: Dictionary containing research task details
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        # Create a summary of the source content if it's too long
        content_preview = source.content[:3000] + "..." if len(source.content) > 3000 else source.content
        
        prompt = f"""
        You are evaluating the relevance of a source for a research task.
        
        RESEARCH TASK:
        {json.dumps(research_task, indent=2)}
        
        SOURCE DETAILS:
        Title: {source.title}
        URL: {source.url}
        Snippet: {source.snippet}
        
        CONTENT PREVIEW:
        {content_preview}
        
        Evaluate how relevant this source is to the research task on a scale from 0.0 to 1.0:
        - 0.0: Completely irrelevant
        - 0.3: Tangentially related but not useful
        - 0.5: Somewhat relevant
        - 0.7: Relevant with good information
        - 0.9-1.0: Highly relevant, exactly what we need
        
        RESPOND WITH ONLY A SINGLE NUMBER BETWEEN 0.0 AND 1.0, NO OTHER TEXT.
        """
        
        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=100,
                temperature=0.0,
                system="You evaluate source relevance with a single number between 0.0 and 1.0.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            result = response.content[0].text.strip()
            
            # Extract just the number
            import re
            score_match = re.search(r'(\d+\.\d+|\d+)', result)
            if score_match:
                score = float(score_match.group(1))
                # Ensure the score is between 0 and 1
                score = max(0.0, min(score, 1.0))
                return score
            else:
                print(f"Could not extract score from Claude's response: {result}")
                return 0.5  # Default to neutral relevance
        except Exception as e:
            print(f"Error evaluating source relevance: {e}")
            return 0.5  # Default to neutral relevance
    
    def generate_follow_up_queries(self, research_task: Dict, 
                                  sources: List[Source], 
                                  previous_queries: List[str]) -> List[SearchQuery]:
        """
        Generate follow-up search queries based on sources found so far.
        
        Args:
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
        
        response = self.anthropic_client.messages.create(
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
    
    def summarize_findings(self, research_task: Dict, sources: List[Source]) -> str:
        """
        Use Claude to generate a summary of the research findings.
        
        Args:
            research_task: Dictionary containing research task details
            sources: List of sources found
            
        Returns:
            Summary of the research findings
        """
        # Create a summary of the top sources
        sources_info = []
        for i, s in enumerate(sources):
            # Limit content preview to reduce token usage
            content_preview = s.content[:1000] + "..." if len(s.content) > 1000 else s.content
            sources_info.append({
                "index": i + 1,
                "title": s.title,
                "url": s.url,
                "snippet": s.snippet,
                "relevance": s.relevance_score,
                "content_preview": content_preview
            })
        
        prompt = f"""
        You are a research assistant summarizing findings from web research.
        
        RESEARCH TASK:
        {json.dumps(research_task, indent=2)}
        
        SOURCES FOUND:
        {json.dumps(sources_info, indent=2)}
        
        Please provide a comprehensive research summary that:
        1. Provides an overview of the topic and key findings
        2. Highlights the most important information from each relevant source
        3. Notes any gaps or areas for further research
        4. Lists the top sources in order of relevance with brief descriptions
        
        FORMAT:
        - Start with an executive summary (2-3 paragraphs)
        - Include a "Key Findings" section with the most important information
        - Include a "Sources" section listing each source with its relevance and brief description
        - End with "Suggested Next Steps" for further research
        
        YOUR RESPONSE SHOULD BE WELL-FORMATTED AND READY TO PRESENT TO THE USER.
        """
        
        response = self.anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=4000,
            temperature=0.2,
            system="You are a helpful research assistant summarizing web research findings.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.content[0].text
    
    def save_source_content(self, sources: List[Source], output_dir: str) -> List[str]:
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
                f.write("\n" + "="*80 + "\n\n")
                f.write(source.content)
            
            file_paths.append(filepath)
        
        return file_paths
    
    def prepare_for_claude(self, research_task: Dict, sources: List[Source], summary: str, output_dir: str) -> str:
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
        
        # Add information about source files
        output.append("\n## Source Files")
        output.append(f"Source content has been saved to: {output_dir}\n")
        
        # Add instructions for using with Claude
        output.append("\n## Using These Results with Claude")
        output.append("Copy this summary and upload the source files to continue your research conversation with Claude.")
        
        return "\n".join(output)
    
    def conduct_research(self, yaml_request: str, output_dir: str = "research_output") -> Tuple[str, List[str]]:
        """
        Main method to conduct the research based on the YAML request.
        
        Args:
            yaml_request: YAML string describing the research task
            output_dir: Directory to save output files
            
        Returns:
            Tuple of (research summary, list of source file paths)
        """
        # Create a timestamp-based output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{output_dir}_{timestamp}"
        
        # Parse the YAML request
        research_task = self.parse_yaml_request(yaml_request)
        
        print(f"Starting research on: {research_task.get('topic', 'Research Topic')}")
        
        # Generate initial search queries
        self.search_queries = self.generate_initial_queries(research_task)
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
            search_results = self.google_search(current_query.query, current_query.site_restrict)
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
                content, content_type = self.extract_content(url)
                source.content = content
                source.content_type = content_type
                
                # Skip if we couldn't extract content
                if 'ERROR' in content:
                    print(f"Skipping due to extraction error")
                    continue
                
                # Evaluate relevance
                source.relevance_score = self.evaluate_source_relevance(source, research_task)
                print(f"Relevance score: {source.relevance_score}")
                
                # Add to our sources if it's relevant enough
                if source.relevance_score >= 0.5:
                    self.sources.append(source)
                    print(f"Added to sources (total: {len(self.sources)})")
                
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
                
                follow_up_queries = self.generate_follow_up_queries(
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
        
        # Generate summary
        print("Generating research summary...")
        summary = self.summarize_findings(research_task, self.sources)
        
        # Save source content
        print(f"Saving source content to {output_dir}...")
        file_paths = self.save_source_content(self.sources, output_dir)
        
        # Prepare output for Claude
        output_text = self.prepare_for_claude(research_task, self.sources, summary, output_dir)
        
        # Save the output
        output_file = os.path.join(output_dir, "research_summary.md")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output_text)
        
        print(f"Research output saved to {output_file}")
        
        return output_text, file_paths

def main():
    """Main entry point for the command-line interface."""
    parser = argparse.ArgumentParser(description="Web Research Tool with Claude")
    parser.add_argument("--input", "-i", type=str, help="YAML input file for research request")
    parser.add_argument("--output", "-o", type=str, default="research_output", help="Output directory for research results")
    parser.add_argument("--max-sources", "-s", type=int, default=10, help="Maximum number of sources to retrieve")
    parser.add_argument("--max-searches", "-q", type=int, default=5, help="Maximum number of search iterations")
    parser.add_argument("--config", "-c", type=str, help="Path to config.json file with API keys")
    
    args = parser.parse_args()
    
    # Try to load config file if specified
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
                for key, value in config.items():
                    os.environ[key] = value
            print(f"Loaded API keys from {args.config}")
        except Exception as e:
            print(f"Error loading config file: {e}")
    
    # Check for environment variables
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    google_cse_id = os.environ.get("GOOGLE_CSE_ID")
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if not google_api_key:
        print("Error: GOOGLE_API_KEY environment variable not set")
        return 1
    
    if not google_cse_id:
        print("Error: GOOGLE_CSE_ID environment variable not set")
        return 1
    
    if not anthropic_api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        return 1
    
    # Read input file if provided
    if args.input:
        try:
            with open(args.input, "r", encoding="utf-8") as f:
                yaml_request = f.read()
        except Exception as e:
            print(f"Error reading input file: {e}")
            return 1
    else:
        # Prompt for YAML input
        if platform.system() == "Windows":
            print("Enter YAML research request (type 'END_OF_YAML' on a new line when finished):")
            yaml_lines = []
            line = input()
            while line != "END_OF_YAML":
                yaml_lines.append(line)
                line = input()
            yaml_request = "\n".join(yaml_lines)
        else:
            # Unix-like systems can use EOF (Ctrl+D)
            print("Enter YAML research request (end with EOF or Ctrl+D):")
            yaml_lines = []
            try:
                while True:
                    line = input()
                    yaml_lines.append(line)
            except EOFError:
                pass
            yaml_request = "\n".join(yaml_lines)
    
    # Create the research tool
    research_tool = WebResearchTool(
        google_api_key=google_api_key,
        google_cse_id=google_cse_id,
        anthropic_api_key=anthropic_api_key,
        max_sources=args.max_sources,
        max_searches=args.max_searches
    )
    
    # Conduct the research
    try:
        output_text, file_paths = research_tool.conduct_research(yaml_request, args.output)
        print(f"Research completed successfully.")
        print(f"Summary saved to {args.output}/research_summary.md")
        print(f"Found {len(file_paths)} relevant sources.")
        return 0
    except Exception as e:
        print(f"Error during research: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())