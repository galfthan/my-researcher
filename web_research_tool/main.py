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
    
    # Create the research tool
    research_tool = WebResearchTool(
        google_api_key=config.get("google_api_key"),
        google_cse_id=config.get("google_cse_id"),
        anthropic_api_key=config.get("anthropic_api_key"),
        max_sources=args.max_sources,
        max_searches=args.max_searches,
        delay=args.delay,
        verbose=args.verbose
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
