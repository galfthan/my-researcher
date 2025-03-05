# Claude Web Research Tool

A Python tool that acts as a "human in the loop" agent for Claude Web Chat to gather relevant information from the internet for research purposes. This tool uses Google Search and Claude Haiku to intelligently search for, evaluate, and collect relevant sources based on your research questions.

## Features

- Intelligent search query generation using Claude Haiku
- Automatic evaluation of source relevance
- Iterative searching based on initial findings
- Support for HTML and PDF content extraction
- Comprehensive research summary generation
- Export of findings in a format ready for Claude Web

## Installation

### Prerequisites

- Python 3.7+
- Google Custom Search API key and Search Engine ID
- Anthropic API key (for Claude Haiku)

### Setup

1. Clone this repository:
   ```
   git clone <repository-url>
   cd claude-web-research-tool
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```
   # On Windows
   set GOOGLE_API_KEY=your_google_api_key
   set GOOGLE_CSE_ID=your_google_cse_id
   set ANTHROPIC_API_KEY=your_anthropic_api_key

   # On Linux/macOS
   export GOOGLE_API_KEY=your_google_api_key
   export GOOGLE_CSE_ID=your_google_cse_id
   export ANTHROPIC_API_KEY=your_anthropic_api_key
   ```

### Google Custom Search Setup

1. Create a Google Custom Search Engine at https://programmablesearchengine.google.com/
2. Enable "Search the entire web" option
3. Get your CSE ID from the setup page
4. Create a Google Cloud project and enable the Custom Search API
5. Create an API key for your project

## Usage

### Command Line Interface

Run the tool with a YAML file containing your research request:

```
python web_research_tool.py --input research_request.yaml --output research_results
```

Or provide the YAML through standard input:

```
python web_research_tool.py
```

Then paste your YAML and press Ctrl+D (or Ctrl+Z on Windows) to finish input.

### YAML Research Request Format

Your research request should be structured like this:

```yaml
topic: "The main research topic or question"
objective: "The specific objective of this research"
depth: "shallow | moderate | deep"  # How deep should the research go
format: "academic | casual | technical"  # What style of sources to prioritize
time_sensitivity: "recent | any"  # Whether to prioritize recent sources
domains:  # Specific domains to focus on or exclude
  include:
    - "example.com"
    - "trusteddomain.org"
  exclude:
    - "untrustedsource.com"
context: "Additional context about the research question"
expected_output: "What specifically you want to know or learn"
```

See the `example_request.yaml` file for a complete example.

### Workflow

1. Start a conversation with Claude Web and discuss your research topic
2. Ask Claude to help you craft a YAML research request based on your discussion
3. Run the research tool with the YAML request
4. Upload the research summary and source files to your Claude Web conversation
5. Continue your discussion with Claude, now informed by the research results

## Output

The tool generates:

1. A research summary in Markdown format
2. Text files containing the extracted content from each source
3. A directory structure that includes all findings

## Limitations

- The tool is limited by Google's Search API quota (100 queries per day for free tier)
- Claude Haiku has a context window limitation, so very long documents may be truncated
- PDF extraction may not work perfectly for all PDFs, especially scanned documents

## Troubleshooting

- **API Key Issues**: Ensure all environment variables are set correctly
- **PDF Extraction Errors**: Try installing additional dependencies: `pip install PyMuPDF`
- **Rate Limiting**: If you encounter rate limits, add a delay parameter: `--delay 5`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
