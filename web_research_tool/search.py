"""
Functions for performing Google searches.
"""

import time
from typing import List, Dict, Optional
from googleapiclient.discovery import build

def google_search(google_service, google_cse_id: str, query: str, 
                 site_restrict: Optional[str] = None, start_index: int = 1, 
                 delay: float = 1.0, max_retries: int = 3) -> List[Dict]:
    """
    Perform a Google search using the Custom Search API with rate limiting and retries.
    
    Args:
        google_service: Google API service instance
        google_cse_id: Custom Search Engine ID
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
            
            result = google_service.cse().list(
                q=full_query,
                cx=google_cse_id,
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
