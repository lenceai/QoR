"""
Database connection and session management for CERN Knowledge Explorer.
"""

from typing import AsyncGenerator, Optional
import asyncio
from contextlib import asynccontextmanager

from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.pool import NullPool

from src.core.config import get_database_url, settings
from src.core.logging import get_logger

logger = get_logger(__name__)

# Database metadata and base
metadata = MetaData()
Base = declarative_base()

# Database engines
engine = None
async_engine = None
SessionLocal = None
AsyncSessionLocal = None


def get_sync_database_url() -> str:
    """Get synchronous database URL."""
    return get_database_url()


def get_async_database_url() -> str:
    """Get asynchronous database URL."""
    url = get_database_url()
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgresql+psycopg2://"):
        return url.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)
    return url


def create_database_engines():
    """Create database engines for sync and async operations."""
    global engine, async_engine, SessionLocal, AsyncSessionLocal
    
    # Synchronous engine for migrations and initial setup
    engine = create_engine(
        get_sync_database_url(),
        pool_pre_ping=True,
        pool_recycle=300,
        echo=settings.DEBUG,
    )
    
    # Asynchronous engine for application operations
    async_engine = create_async_engine(
        get_async_database_url(),
        pool_pre_ping=True,
        pool_recycle=300,
        echo=settings.DEBUG,
        poolclass=NullPool if settings.DEBUG else None,
    )
    
    # Session factories
    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
    )
    
    AsyncSessionLocal = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    logger.info("Database engines created successfully")


def get_sync_session() -> Session:
    """Get synchronous database session."""
    if SessionLocal is None:
        create_database_engines()
    return SessionLocal()


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Get asynchronous database session."""
    if AsyncSessionLocal is None:
        create_database_engines()
    
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_async_session_context() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session as context manager."""
    if AsyncSessionLocal is None:
        create_database_engines()
    
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise


class DatabaseManager:
    """Database operations manager."""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
    
    async def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            async with get_async_session_context() as session:
                from sqlalchemy import text
                result = await session.execute(text("SELECT 1"))
                return result.scalar() == 1
        except Exception as e:
            self.logger.error(f"Database health check failed: {e}")
            return False
    
    async def get_database_info(self) -> dict:
        """Get database information and statistics."""
        try:
            async with get_async_session_context() as session:
                # Get database version
                version_result = await session.execute("SELECT version()")
                version = version_result.scalar()
                
                # Get database size
                size_query = """
                SELECT pg_size_pretty(pg_database_size(current_database())) as size
                """
                size_result = await session.execute(size_query)
                size = size_result.scalar()
                
                # Get connection count
                conn_query = """
                SELECT count(*) as connections 
                FROM pg_stat_activity 
                WHERE datname = current_database()
                """
                conn_result = await session.execute(conn_query)
                connections = conn_result.scalar()
                
                return {
                    "version": version,
                    "size": size,
                    "connections": connections,
                    "status": "healthy"
                }
        except Exception as e:
            self.logger.error(f"Failed to get database info: {e}")
            return {"status": "error", "error": str(e)}
    
    async def close_connections(self):
        """Close all database connections."""
        try:
            if async_engine:
                await async_engine.dispose()
            if engine:
                engine.dispose()
            self.logger.info("Database connections closed")
        except Exception as e:
            self.logger.error(f"Error closing database connections: {e}")


class TransactionManager:
    """Database transaction management."""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
    
    @asynccontextmanager
    async def transaction(self, session: Optional[AsyncSession] = None):
        """Manage database transactions with automatic rollback on error."""
        if session:
            # Use existing session
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                self.logger.error(f"Transaction rolled back: {e}")
                raise
        else:
            # Create new session and transaction
            async with get_async_session_context() as session:
                try:
                    yield session
                    await session.commit()
                except Exception as e:
                    await session.rollback()
                    self.logger.error(f"Transaction rolled back: {e}")
                    raise


# Global instances
db_manager = DatabaseManager()
transaction_manager = TransactionManager()


async def init_database():
    """Initialize database connection and create tables."""
    try:
        create_database_engines()
        logger.info("Database initialization completed")
        
        # Create tables
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")
        
        # Test connection
        is_healthy = await db_manager.health_check()
        if is_healthy:
            logger.info("Database connection test successful")
        else:
            logger.error("Database connection test failed")
            
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


async def close_database():
    """Close database connections."""
    await db_manager.close_connections()


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


# Dependency for FastAPI
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency to get database session."""
    async for session in get_async_session():
        yield session 