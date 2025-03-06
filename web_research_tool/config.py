"""
Configuration handling for the Web Research Tool.
"""

import os
import json
from typing import Dict, Any, Optional

def load_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from environment variables or config file.
    
    Args:
        config_file: Optional path to JSON config file
        
    Returns:
        Dictionary with configuration
    """
    config = {}
    
    # Try to load config file if specified
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                for key, value in file_config.items():
                    # Update environment variables for compatibility
                    os.environ[key] = value
                    config[key] = value
            print(f"Loaded API keys from {config_file}")
        except Exception as e:
            print(f"Error loading config file: {e}")
    
    # Get values from environment variables
    env_keys = {
        "GOOGLE_API_KEY": "google_api_key",
        "GOOGLE_CSE_ID": "google_cse_id",
        "ANTHROPIC_API_KEY": "anthropic_api_key"
    }
    
    for env_key, config_key in env_keys.items():
        if env_key in os.environ:
            config[config_key] = os.environ[env_key]
    
    # Validate required keys
    missing_keys = []
    for key in ["google_api_key", "google_cse_id", "anthropic_api_key"]:
        if key not in config:
            missing_keys.append(key)
    
    if missing_keys:
        keys_str = ", ".join(missing_keys)
        print(f"Warning: Missing required configuration keys: {keys_str}")
    
    return config

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate that all required configuration is present.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if configuration is valid, False otherwise
    """
    required_keys = ["google_api_key", "google_cse_id", "anthropic_api_key"]
    
    for key in required_keys:
        if key not in config or not config[key]:
            print(f"Error: Missing required configuration key: {key}")
            return False
    
    return True
