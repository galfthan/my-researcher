"""
Functions for extracting content from URLs.
"""

import io
import requests
from typing import Tuple
from bs4 import BeautifulSoup
import PyPDF2

def extract_content(url: str) -> Tuple[str, str]:
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
