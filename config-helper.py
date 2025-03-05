import os
import json
import argparse

def create_config_file(api_keys):
    """Create a config.json file with API keys."""
    with open('config.json', 'w') as f:
        json.dump(api_keys, f, indent=2)
    print("Created config.json with your API keys")

def load_config():
    """Load API keys from config.json if it exists."""
    if os.path.exists('config.json'):
        with open('config.json', 'r') as f:
            return json.load(f)
    return {}

def set_environment_variables(api_keys):
    """Set environment variables from the API keys."""
    for key, value in api_keys.items():
        os.environ[key] = value
    print("Environment variables set successfully")

def main():
    parser = argparse.ArgumentParser(description="Configure API keys for Web Research Tool")
    parser.add_argument("--google-api-key", help="Google API Key for Custom Search")
    parser.add_argument("--google-cse-id", help="Google Custom Search Engine ID")
    parser.add_argument("--anthropic-api-key", help="Anthropic API Key for Claude")
    parser.add_argument("--save", action="store_true", help="Save configuration to config.json")
    
    args = parser.parse_args()
    
    # Load existing config if available
    api_keys = load_config()
    
    # Update with new values if provided
    if args.google_api_key:
        api_keys["GOOGLE_API_KEY"] = args.google_api_key
    
    if args.google_cse_id:
        api_keys["GOOGLE_CSE_ID"] = args.google_cse_id
    
    if args.anthropic_api_key:
        api_keys["ANTHROPIC_API_KEY"] = args.anthropic_api_key
    
    # Save to file if requested
    if args.save:
        create_config_file(api_keys)
    
    # Set environment variables
    set_environment_variables(api_keys)
    
    # Check if all required keys are present
    required_keys = ["GOOGLE_API_KEY", "GOOGLE_CSE_ID", "ANTHROPIC_API_KEY"]
    missing_keys = [key for key in required_keys if key not in api_keys or not api_keys[key]]
    
    if missing_keys:
        print(f"Warning: The following required API keys are missing: {', '.join(missing_keys)}")
        print("Please set them using --google-api-key, --google-cse-id, and --anthropic-api-key arguments")
    else:
        print("All required API keys are set")

if __name__ == "__main__":
    main()
