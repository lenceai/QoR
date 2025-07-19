"""
Main API router for v1 endpoints.
"""

from fastapi import APIRouter
from src.api.v1.endpoints import system, papers, authors, institutions, documents

api_router = APIRouter()

# Include endpoint routers
api_router.include_router(system.router, prefix="/system", tags=["System"])
api_router.include_router(papers.router, prefix="/papers", tags=["Papers"])
api_router.include_router(authors.router, prefix="/authors", tags=["Authors"])
api_router.include_router(institutions.router, prefix="/institutions", tags=["Institutions"])
api_router.include_router(documents.router, prefix="/documents", tags=["Documents"]) 