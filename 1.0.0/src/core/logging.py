"""
Structured logging configuration for CERN Knowledge Explorer.
"""

import json
import logging
import logging.config
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import structlog
from structlog.stdlib import LoggerFactory

from src.core.config import settings


def setup_logging() -> None:
    """Configure structured logging for the application."""
    
    # Ensure log directory exists
    log_dir = Path("output/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure timestamper
    timestamper = structlog.processors.TimeStamper(fmt="ISO")
    
    # Custom filter_by_level processor that handles None loggers safely
    def safe_filter_by_level(logger, method_name, event_dict):
        """Safely filter by log level, handling None loggers."""
        if logger is None:
            return event_dict
        try:
            return structlog.stdlib.filter_by_level(logger, method_name, event_dict)
        except AttributeError:
            # If logger doesn't have isEnabledFor, just return the event dict
            return event_dict
    
    # Configure shared processors
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        safe_filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    # Configure formatting based on environment
    if settings.LOG_FORMAT == "json":
        # JSON formatting for production
        formatter = structlog.processors.JSONRenderer()
    else:
        # Human-readable formatting for development
        formatter = structlog.dev.ConsoleRenderer(colors=True)
    
    # Configure structlog
    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processor": structlog.processors.JSONRenderer(),
                "foreign_pre_chain": shared_processors,
            },
            "console": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processor": structlog.dev.ConsoleRenderer(colors=True),
                "foreign_pre_chain": shared_processors,
            },
        },
        "handlers": {
            "console": {
                "level": settings.LOG_LEVEL,
                "class": "logging.StreamHandler",
                "formatter": "console" if settings.DEBUG else "json",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "level": settings.LOG_LEVEL,
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "json",
                "filename": "output/logs/cern_explorer.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
            },
            "error_file": {
                "level": "ERROR",
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "json",
                "filename": "output/logs/errors.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
            },
        },
        "loggers": {
            "": {
                "handlers": ["console", "file", "error_file"],
                "level": settings.LOG_LEVEL,
                "propagate": True,
            },
            "uvicorn": {
                "handlers": ["console", "file"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn.error": {
                "handlers": ["console", "error_file"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["file"],
                "level": "INFO",
                "propagate": False,
            },
            "sqlalchemy.engine": {
                "handlers": ["file"],
                "level": "WARNING" if not settings.DEBUG else "INFO",
                "propagate": False,
                "qualname": "sqlalchemy.engine",
            },
            "sqlalchemy.pool": {
                "handlers": ["file"],
                "level": "WARNING",
                "propagate": False,
            },
            "sqlalchemy.dialects": {
                "handlers": ["file"],
                "level": "WARNING",
                "propagate": False,
            },
            "elasticsearch": {
                "handlers": ["file"],
                "level": "WARNING",
                "propagate": False,
            },
        },
    }
    
    logging.config.dictConfig(logging_config)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


class RequestLogger:
    """Request/response logging middleware."""
    
    def __init__(self):
        self.logger = get_logger("request")
    
    async def log_request(self, request, response=None, duration: Optional[float] = None):
        """Log HTTP request and response details."""
        log_data = {
            "method": request.method,
            "url": str(request.url),
            "headers": dict(request.headers),
            "client_ip": request.client.host if request.client else None,
        }
        
        if response:
            log_data.update({
                "status_code": response.status_code,
                "response_headers": dict(response.headers),
            })
        
        if duration:
            log_data["duration_ms"] = round(duration * 1000, 2)
        
        self.logger.info("HTTP request", **log_data)


class AuditLogger:
    """Audit logging for security and compliance."""
    
    def __init__(self):
        self.logger = get_logger("audit")
    
    def log_auth_event(self, event_type: str, user_id: Optional[str] = None, **kwargs):
        """Log authentication events."""
        self.logger.info(
            "Authentication event",
            event_type=event_type,
            user_id=user_id,
            **kwargs
        )
    
    def log_data_access(self, user_id: str, resource: str, action: str, **kwargs):
        """Log data access events."""
        self.logger.info(
            "Data access",
            user_id=user_id,
            resource=resource,
            action=action,
            **kwargs
        )
    
    def log_admin_action(self, user_id: str, action: str, target: str, **kwargs):
        """Log administrative actions."""
        self.logger.warning(
            "Admin action",
            user_id=user_id,
            action=action,
            target=target,
            **kwargs
        )


class PerformanceLogger:
    """Performance monitoring and logging."""
    
    def __init__(self):
        self.logger = get_logger("performance")
    
    def log_query_performance(self, query_type: str, duration: float, result_count: int, **kwargs):
        """Log database/search query performance."""
        self.logger.info(
            "Query performance",
            query_type=query_type,
            duration_ms=round(duration * 1000, 2),
            result_count=result_count,
            **kwargs
        )
    
    def log_processing_performance(self, task_type: str, duration: float, items_processed: int, **kwargs):
        """Log data processing performance."""
        self.logger.info(
            "Processing performance",
            task_type=task_type,
            duration_seconds=round(duration, 2),
            items_processed=items_processed,
            throughput_per_second=round(items_processed / duration, 2) if duration > 0 else 0,
            **kwargs
        )


def setup_request_id_context():
    """Setup request ID context for tracing."""
    import uuid
    from contextvars import ContextVar
    
    request_id_var: ContextVar[str] = ContextVar("request_id", default="")
    
    def add_request_id(logger, method_name, event_dict):
        """Add request ID to log entries."""
        request_id = request_id_var.get()
        if request_id:
            event_dict["request_id"] = request_id
        return event_dict
    
    # Add processor to include request ID in logs
    structlog.configure(
        processors=[
            add_request_id,
        ] + structlog.get_config()["processors"]
    )
    
    return request_id_var


# Global logger instances
logger = get_logger(__name__)
request_logger = RequestLogger()
audit_logger = AuditLogger()
performance_logger = PerformanceLogger()

# Initialize logging on module import
setup_logging() 