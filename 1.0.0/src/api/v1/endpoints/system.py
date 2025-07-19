"""
System endpoints for health checks and system information.
"""

from typing import Any, Dict
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.core.logging import get_logger
from src.data.database import get_db, db_manager

logger = get_logger(__name__)
router = APIRouter()


@router.get("/info")
async def get_system_info() -> Dict[str, Any]:
    """Get system information."""
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "debug": settings.DEBUG,
        "environment": "development" if settings.DEBUG else "production",
        "api_version": "v1"
    }


@router.get("/health")
async def detailed_health_check(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Detailed health check with service status."""
    
    # Check database health
    db_info = await db_manager.get_database_info()
    
    # TODO: Add other service checks
    services = {
        "database": db_info,
        # "elasticsearch": {...},  # TODO
        # "redis": {...},  # TODO
    }
    
    # Determine overall status
    all_healthy = all(
        service.get("status") == "healthy" 
        for service in services.values()
    )
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "services": services,
        "timestamp": db_info.get("timestamp"),
        "version": settings.VERSION
    }


@router.get("/stats")
async def get_system_stats(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Get system statistics."""
    
    try:
        # Get database statistics
        db_stats = await db_manager.get_database_info()
        
        # TODO: Add more statistics
        # - Number of papers, authors, institutions
        # - Search index status
        # - Cache statistics
        
        return {
            "database": db_stats,
            "timestamp": db_stats.get("timestamp"),
            # TODO: Add more stats
        }
        
    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        return {
            "error": "Failed to retrieve system statistics",
            "timestamp": None
        }


@router.get("/config")
async def get_public_config() -> Dict[str, Any]:
    """Get public configuration settings."""
    return {
        "project_name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "api_version": "v1",
        "debug": settings.DEBUG,
        "rate_limits": {
            "per_minute": settings.RATE_LIMIT_PER_MINUTE,
            "burst": settings.RATE_LIMIT_BURST
        },
        "file_upload": {
            "max_size": settings.MAX_FILE_SIZE,
            "allowed_types": settings.ALLOWED_FILE_TYPES
        },
        "features": {
            "search": True,
            "visualization": True,  # TODO: Will be implemented in later phases
            "user_accounts": True,
            "collections": True
        }
    } 