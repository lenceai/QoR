"""
General utility functions for CERN Knowledge Explorer.
"""

import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse


def generate_uuid() -> str:
    """Generate a UUID string."""
    return str(uuid.uuid4())


def get_utc_now() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)


def normalize_string(text: str) -> str:
    """Normalize a string by removing extra whitespace and converting to lowercase."""
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text.strip().lower())


def extract_keywords(text: str, min_length: int = 3) -> List[str]:
    """Extract keywords from text by splitting on common delimiters."""
    if not text:
        return []
    
    # Remove special characters and split
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Filter by minimum length and remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
        'could', 'can', 'may', 'might', 'must', 'this', 'that', 'these', 'those'
    }
    
    return [word for word in words if len(word) >= min_length and word not in stop_words]


def validate_email(email: str) -> bool:
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_url(url: str) -> bool:
    """Validate URL format."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def safe_get(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Safely get a value from a dictionary."""
    return data.get(key, default)


def flatten_dict(data: Dict[str, Any], separator: str = ".") -> Dict[str, Any]:
    """Flatten a nested dictionary."""
    def _flatten(obj, parent_key=""):
        items = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{parent_key}{separator}{k}" if parent_key else k
                items.extend(_flatten(v, new_key).items())
        else:
            return {parent_key: obj}
        return dict(items)
    
    return _flatten(data)


def parse_doi(doi: str) -> Optional[str]:
    """Parse and normalize a DOI."""
    if not doi:
        return None
    
    # Remove common prefixes
    doi = doi.strip()
    if doi.startswith('doi:'):
        doi = doi[4:]
    elif doi.startswith('http://dx.doi.org/'):
        doi = doi[18:]
    elif doi.startswith('https://dx.doi.org/'):
        doi = doi[19:]
    elif doi.startswith('http://doi.org/'):
        doi = doi[15:]
    elif doi.startswith('https://doi.org/'):
        doi = doi[16:]
    
    # Validate DOI format
    if re.match(r'^10\.\d+/.*', doi):
        return doi
    
    return None


def chunk_list(data: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of specified size."""
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}" 