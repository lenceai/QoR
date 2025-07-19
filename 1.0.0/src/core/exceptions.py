"""
Custom exception classes for CERN Knowledge Explorer.
"""

from typing import Any, Dict, Optional


class CERNExplorerException(Exception):
    """Base exception for CERN Knowledge Explorer."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 500
    ):
        self.message = message
        self.details = details or {}
        self.status_code = status_code
        super().__init__(self.message)


class ValidationError(CERNExplorerException):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        details = {"field": field} if field else {}
        details.update(kwargs)
        super().__init__(message, details, status_code=400)


class AuthenticationError(CERNExplorerException):
    """Raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(message, kwargs, status_code=401)


class AuthorizationError(CERNExplorerException):
    """Raised when authorization fails."""
    
    def __init__(self, message: str = "Access denied", **kwargs):
        super().__init__(message, kwargs, status_code=403)


class NotFoundError(CERNExplorerException):
    """Raised when a resource is not found."""
    
    def __init__(self, message: str, resource_type: Optional[str] = None, **kwargs):
        details = {"resource_type": resource_type} if resource_type else {}
        details.update(kwargs)
        super().__init__(message, details, status_code=404)


class ConflictError(CERNExplorerException):
    """Raised when there's a conflict with existing data."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, kwargs, status_code=409)


class RateLimitError(CERNExplorerException):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, message: str = "Rate limit exceeded", **kwargs):
        super().__init__(message, kwargs, status_code=429)


class DatabaseError(CERNExplorerException):
    """Raised when database operations fail."""
    
    def __init__(self, message: str, operation: Optional[str] = None, **kwargs):
        details = {"operation": operation} if operation else {}
        details.update(kwargs)
        super().__init__(message, details, status_code=500)


class SearchError(CERNExplorerException):
    """Raised when search operations fail."""
    
    def __init__(self, message: str, query: Optional[str] = None, **kwargs):
        details = {"query": query} if query else {}
        details.update(kwargs)
        super().__init__(message, details, status_code=500)


class ProcessingError(CERNExplorerException):
    """Raised when data processing fails."""
    
    def __init__(self, message: str, task_type: Optional[str] = None, **kwargs):
        details = {"task_type": task_type} if task_type else {}
        details.update(kwargs)
        super().__init__(message, details, status_code=500)


class ExternalServiceError(CERNExplorerException):
    """Raised when external service calls fail."""
    
    def __init__(self, message: str, service: Optional[str] = None, **kwargs):
        details = {"service": service} if service else {}
        details.update(kwargs)
        super().__init__(message, details, status_code=502)


class ConfigurationError(CERNExplorerException):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        details = {"config_key": config_key} if config_key else {}
        details.update(kwargs)
        super().__init__(message, details, status_code=500)


class DataIngestionError(CERNExplorerException):
    """Raised when data ingestion operations fail."""
    
    def __init__(self, message: str, source: Optional[str] = None, **kwargs):
        details = {"source": source} if source else {}
        details.update(kwargs)
        super().__init__(message, details, status_code=500)


class IndexingError(CERNExplorerException):
    """Raised when search indexing operations fail."""
    
    def __init__(self, message: str, index_name: Optional[str] = None, **kwargs):
        details = {"index_name": index_name} if index_name else {}
        details.update(kwargs)
        super().__init__(message, details, status_code=500) 