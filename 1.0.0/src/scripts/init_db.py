#!/usr/bin/env python3
"""
Database initialization script for CERN Knowledge Explorer.
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.config import settings
from src.core.logging import setup_logging, get_logger
from src.data.database import init_database, close_database
from src.data.models import Base

logger = get_logger(__name__)


async def create_tables():
    """Create all database tables."""
    logger.info("Creating database tables...")
    
    try:
        await init_database()
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        raise
    finally:
        await close_database()


async def main():
    """Main initialization function."""
    # Ensure logging is set up first
    setup_logging()
    
    # Get logger after setup
    logger = get_logger(__name__)
    logger.info("Starting database initialization...")
    
    logger.info(f"Database URL: {settings.DATABASE_URL}")
    logger.info(f"Environment: {'development' if settings.DEBUG else 'production'}")
    
    await create_tables()
    
    logger.info("Database initialization completed successfully")


if __name__ == "__main__":
    asyncio.run(main()) 