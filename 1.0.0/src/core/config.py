"""
Configuration management for CERN Knowledge Explorer.
"""

import os
from typing import Any, Dict, List, Optional, Union
from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application Configuration
    DEBUG: bool = False
    SECRET_KEY: str = "change-this-secret-key-in-production"
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "CERN Knowledge Explorer"
    VERSION: str = "1.0.0"
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    
    # Database Configuration
    DATABASE_URL: str = "postgresql://cern_user:cern_password@localhost:5432/cern_explorer"
    DATABASE_TEST_URL: str = "postgresql://cern_user:cern_password@localhost:5432/cern_explorer_test"
    
    # Elasticsearch Configuration
    ELASTICSEARCH_URL: str = "http://localhost:9200"
    ELASTICSEARCH_INDEX_PREFIX: str = "cern_explorer"
    
    # Redis Configuration
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_CACHE_TTL: int = 3600
    
    # Celery Configuration
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"
    
    # Security Configuration
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_MINUTES: int = 10080  # 7 days
    ALGORITHM: str = "HS256"
    
    # CORS Configuration
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    
    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, list):
            return v
        elif isinstance(v, str):
            # Handle both JSON format and comma-separated format
            if v.startswith("["):
                import json
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    pass
            # Fall back to comma-separated parsing
            return [i.strip() for i in v.split(",") if i.strip()]
        return []
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    
    # CERN Data Sources
    CERN_DOCUMENT_API_URL: str = "https://cds.cern.ch/api"
    CERN_INSPIRE_API_URL: str = "https://inspirehep.net/api"
    CERN_ZENODO_API_URL: str = "https://zenodo.org/api"
    
    # File Upload Configuration
    MAX_FILE_SIZE: str = "100MB"
    ALLOWED_FILE_TYPES: List[str] = ["pdf", "doc", "docx", "txt", "xml", "json"]
    
    @field_validator("ALLOWED_FILE_TYPES", mode="before")
    @classmethod
    def assemble_file_types(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, list):
            return v
        return []
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 100
    RATE_LIMIT_BURST: int = 20
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    
    # Email Configuration (Optional)
    SMTP_TLS: bool = True
    SMTP_PORT: int = 587
    SMTP_HOST: Optional[str] = None
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    
    # External Services
    ORCID_CLIENT_ID: Optional[str] = None
    ORCID_CLIENT_SECRET: Optional[str] = None
    
    # Data Processing
    BATCH_SIZE: int = 1000
    MAX_CONCURRENT_JOBS: int = 5
    PROCESSING_TIMEOUT: int = 300
    
    @property
    def max_file_size_bytes(self) -> int:
        """Convert MAX_FILE_SIZE string to bytes."""
        size_str = self.MAX_FILE_SIZE.upper()
        if size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def get_database_url() -> str:
    """Get database URL based on environment."""
    settings = get_settings()
    if os.getenv("TESTING"):
        return settings.DATABASE_TEST_URL
    return settings.DATABASE_URL


def get_elasticsearch_config() -> Dict[str, Any]:
    """Get Elasticsearch configuration."""
    settings = get_settings()
    return {
        "hosts": [settings.ELASTICSEARCH_URL],
        "verify_certs": False,
        "ssl_show_warn": False,
    }


def get_redis_config() -> Dict[str, Any]:
    """Get Redis configuration."""
    settings = get_settings()
    return {
        "url": settings.REDIS_URL,
        "decode_responses": True,
    }


def get_celery_config() -> Dict[str, Any]:
    """Get Celery configuration."""
    settings = get_settings()
    return {
        "broker_url": settings.CELERY_BROKER_URL,
        "result_backend": settings.CELERY_RESULT_BACKEND,
        "task_serializer": "json",
        "result_serializer": "json",
        "accept_content": ["json"],
        "timezone": "UTC",
        "enable_utc": True,
    }


# Global settings instance
settings = get_settings() 