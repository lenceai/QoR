"""
Main application entry point for CERN Knowledge Explorer.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.core.config import settings
from src.core.logging import get_logger, request_logger, audit_logger
from src.core.exceptions import CERNExplorerException
from src.data.database import init_database, close_database
from src.api.v1.api import api_router

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting CERN Knowledge Explorer", version=settings.VERSION)
    
    try:
        # Initialize database
        await init_database()
        logger.info("Database initialized successfully")
        
        # TODO: Initialize Elasticsearch
        # TODO: Initialize Redis
        # TODO: Start background tasks
        
        logger.info("Application startup completed")
        yield
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down CERN Knowledge Explorer")
        
        try:
            await close_database()
            logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        
        logger.info("Application shutdown completed")


def create_application() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="A comprehensive system for exploring, analyzing, and visualizing scientific data and research from CERN",
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    
    # Configure middleware
    setup_middleware(app)
    
    # Configure exception handlers
    setup_exception_handlers(app)
    
    # Include API routers
    app.include_router(api_router, prefix=settings.API_V1_STR)
    
    # Add request logging middleware
    app.middleware("http")(log_requests)
    
    return app


def setup_middleware(app: FastAPI) -> None:
    """Configure application middleware."""
    
    # CORS middleware
    if settings.BACKEND_CORS_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Trusted host middleware for production
    if not settings.DEBUG:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.BACKEND_CORS_ORIGINS or ["localhost", "127.0.0.1"]
        )


def setup_exception_handlers(app: FastAPI) -> None:
    """Configure custom exception handlers."""
    
    @app.exception_handler(CERNExplorerException)
    async def cern_explorer_exception_handler(request: Request, exc: CERNExplorerException):
        """Handle custom application exceptions."""
        logger.error(
            "Application exception",
            exception=exc.__class__.__name__,
            message=exc.message,
            details=exc.details,
            url=str(request.url),
            method=request.method
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "type": exc.__class__.__name__,
                    "message": exc.message,
                    "details": exc.details
                }
            }
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors."""
        logger.warning(
            "Validation error",
            errors=exc.errors(),
            url=str(request.url),
            method=request.method
        )
        
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "type": "ValidationError",
                    "message": "Request validation failed",
                    "details": exc.errors()
                }
            }
        )
    
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle HTTP exceptions."""
        logger.warning(
            "HTTP exception",
            status_code=exc.status_code,
            detail=exc.detail,
            url=str(request.url),
            method=request.method
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "type": "HTTPException",
                    "message": exc.detail,
                    "status_code": exc.status_code
                }
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions."""
        logger.error(
            "Unexpected exception",
            exception=exc.__class__.__name__,
            message=str(exc),
            url=str(request.url),
            method=request.method,
            exc_info=True
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "type": "InternalServerError",
                    "message": "An unexpected error occurred",
                    "details": {"exception": exc.__class__.__name__} if settings.DEBUG else {}
                }
            }
        )


async def log_requests(request: Request, call_next):
    """Log HTTP requests and responses."""
    # Generate request ID for tracing
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # Record start time
    start_time = time.time()
    
    try:
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log request details
        await request_logger.log_request(request, response, duration)
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response
        
    except Exception as e:
        # Log failed request
        duration = time.time() - start_time
        logger.error(
            "Request failed",
            request_id=request_id,
            method=request.method,
            url=str(request.url),
            duration_ms=round(duration * 1000, 2),
            error=str(e)
        )
        raise


# Create the application instance
app = create_application()


@app.get("/", tags=["Root"])
async def root() -> Dict[str, Any]:
    """Root endpoint returning application information."""
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "description": "CERN Knowledge Explorer API",
        "status": "healthy",
        "docs": "/docs",
        "redoc": "/redoc",
        "api": settings.API_V1_STR
    }


@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    from src.data.database import db_manager
    
    # Check database connectivity
    db_healthy = await db_manager.health_check()
    
    # TODO: Check Elasticsearch connectivity
    # TODO: Check Redis connectivity
    
    status = "healthy" if db_healthy else "unhealthy"
    
    return {
        "status": status,
        "timestamp": time.time(),
        "version": settings.VERSION,
        "services": {
            "database": "healthy" if db_healthy else "unhealthy",
            # "elasticsearch": "healthy",  # TODO
            # "redis": "healthy",  # TODO
        }
    }


if __name__ == "__main__":
    # Run the application with uvicorn
    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=False,  # We handle our own access logging
    ) 