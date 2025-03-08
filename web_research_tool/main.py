"""
Command-line interface for the Web Research Tool.
"""

import sys
import argparse
import platform
from typing import List, Tuple
import yaml

from .config import load_config, validate_config
from .web_research_tool import WebResearchTool

def main():
    """Main entry point for the command-line interface."""
    parser = argparse.ArgumentParser(description="Web Research Tool with Claude")
    parser.add_argument("--input", "-i", type=str, help="YAML input file for research request")
    parser.add_argument("--output", "-o", type=str, default="research_output", help="Output directory for research results")
    parser.add_argument("--max-sources", "-s", type=int, default=10, help="Maximum number of sources to retrieve")
    parser.add_argument("--max-searches", "-q", type=int, default=5, help="Maximum number of search iterations")
    parser.add_argument("--config", "-c", type=str, help="Path to config.json file with API keys")
    parser.add_argument("--delay", "-d", type=float, default=1.0, help="Delay between API requests (in seconds)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--quick", "-Q", action="store_true", help="Quick mode - skip detailed summaries")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Validate configuration
    if not validate_config(config):
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
    
    # Handle YAML content
    try:
        request_data = yaml.safe_load(yaml_request)
        
        # Check if detailed_summaries is specified in the YAML
        generate_detailed_summaries = not args.quick
        if "detailed_summaries" in request_data:
            generate_detailed_summaries = request_data["detailed_summaries"]
            # Remove from request to avoid confusion in processing
            del request_data["detailed_summaries"]
            yaml_request = yaml.dump(request_data)
            
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}")
        return 1
    
    # Create the research tool
    research_tool = WebResearchTool(
        google_api_key=config.get("google_api_key"),
        google_cse_id=config.get("google_cse_id"),
        anthropic_api_key=config.get("anthropic_api_key"),
        max_sources=args.max_sources,
        max_searches=args.max_searches,
        delay=args.delay,
        verbose=args.verbose,
        generate_detailed_summaries=generate_detailed_summaries
    )
    
    # Conduct the research
    try:
        print(f"Running in {'quick' if not generate_detailed_summaries else 'detailed'} mode")
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