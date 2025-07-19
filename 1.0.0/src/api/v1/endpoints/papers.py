"""
Papers endpoints - Phase 2 implementation.
Basic CRUD operations for papers with database integration.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, Query, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
import yaml
from pathlib import Path

from src.core.logging import get_logger
from src.data.database import get_db
from src.data.models import Paper, Author, Institution
from src.services.data_sources import DataIngestionService
from src.services.data_persistence import DataPersistenceService, process_ingestion_batch

logger = get_logger(__name__)
router = APIRouter()


@router.get("/")
async def list_papers(
    skip: int = Query(0, ge=0, description="Number of papers to skip"),
    limit: int = Query(10, ge=1, le=100, description="Number of papers to return"),
    source: Optional[str] = Query(None, description="Filter by data source"),
    category: Optional[str] = Query(None, description="Filter by category"),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """List papers with pagination and filtering."""
    try:
        # Build query with filters
        stmt = select(Paper).options(
            selectinload(Paper.authors),
            selectinload(Paper.institutions)
        )
        
        if source:
            stmt = stmt.where(Paper.source == source)
        
        if category:
            stmt = stmt.where(Paper.categories.contains([category]))
        
        # Get total count
        count_stmt = select(func.count(Paper.id))
        if source:
            count_stmt = count_stmt.where(Paper.source == source)
        if category:
            count_stmt = count_stmt.where(Paper.categories.contains([category]))
        
        total = await db.scalar(count_stmt)
        
        # Get paginated results
        stmt = stmt.offset(skip).limit(limit).order_by(Paper.created_at.desc())
        result = await db.execute(stmt)
        papers = result.scalars().all()
        
        # Format response
        papers_data = []
        for paper in papers:
            papers_data.append({
                "id": str(paper.id),
                "external_id": paper.external_id,
                "title": paper.title,
                "abstract": paper.abstract[:200] + "..." if len(paper.abstract) > 200 else paper.abstract,
                "publication_date": paper.publication_date.isoformat() if paper.publication_date else None,
                "journal": paper.journal,
                "doi": paper.doi,
                "arxiv_id": paper.arxiv_id,
                "categories": paper.categories,
                "keywords": paper.keywords[:10],  # First 10 keywords
                "citation_count": paper.citation_count,
                "source": paper.source,
                "authors": [{"name": author.name} for author in paper.authors],
                "created_at": paper.created_at.isoformat()
            })
        
        return {
            "papers": papers_data,
            "total": total or 0,
            "skip": skip,
            "limit": limit,
            "filters": {"source": source, "category": category}
        }
        
    except Exception as e:
        logger.error(f"Error listing papers: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve papers")


@router.get("/{paper_id}")
async def get_paper(
    paper_id: str,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Get a specific paper by ID."""
    try:
        # Try to find by UUID first, then by external_id
        stmt = select(Paper).options(
            selectinload(Paper.authors),
            selectinload(Paper.institutions)
        )
        
        try:
            # Try UUID format
            import uuid
            uuid_obj = uuid.UUID(paper_id)
            stmt = stmt.where(Paper.id == uuid_obj)
        except ValueError:
            # Fall back to external_id
            stmt = stmt.where(Paper.external_id == paper_id)
        
        result = await db.execute(stmt)
        paper = result.scalar_one_or_none()
        
        if not paper:
            raise HTTPException(status_code=404, detail="Paper not found")
        
        return {
            "id": str(paper.id),
            "external_id": paper.external_id,
            "title": paper.title,
            "abstract": paper.abstract,
            "publication_date": paper.publication_date.isoformat() if paper.publication_date else None,
            "journal": paper.journal,
            "doi": paper.doi,
            "arxiv_id": paper.arxiv_id,
            "categories": paper.categories,
            "keywords": paper.keywords,
            "citation_count": paper.citation_count,
            "download_count": paper.download_count,
            "source": paper.source,
            "authors": [
                {
                    "id": str(author.id),
                    "name": author.name,
                    "email": author.email,
                    "orcid": author.orcid
                } for author in paper.authors
            ],
            "institutions": [
                {
                    "id": str(inst.id),
                    "name": inst.name,
                    "country": inst.country,
                    "type": inst.institution_type
                } for inst in paper.institutions
            ],
            "created_at": paper.created_at.isoformat(),
            "updated_at": paper.updated_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving paper {paper_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve paper")


@router.get("/search")
async def search_papers(
    q: str = Query(..., description="Search query"),
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Basic search papers by title and abstract."""
    try:
        # Simple text search in title and abstract for Phase 2
        # Will be enhanced with Elasticsearch in Phase 3
        search_term = f"%{q}%"
        
        stmt = select(Paper).where(
            (Paper.title.ilike(search_term)) |
            (Paper.abstract.ilike(search_term))
        ).options(
            selectinload(Paper.authors)
        )
        
        # Get total count
        count_stmt = select(func.count(Paper.id)).where(
            (Paper.title.ilike(search_term)) |
            (Paper.abstract.ilike(search_term))
        )
        total = await db.scalar(count_stmt)
        
        # Get results
        stmt = stmt.offset(skip).limit(limit).order_by(Paper.citation_count.desc())
        result = await db.execute(stmt)
        papers = result.scalars().all()
        
        # Format results
        results = []
        for paper in papers:
            results.append({
                "id": str(paper.id),
                "external_id": paper.external_id,
                "title": paper.title,
                "abstract": paper.abstract[:200] + "..." if len(paper.abstract) > 200 else paper.abstract,
                "authors": [{"name": author.name} for author in paper.authors],
                "journal": paper.journal,
                "publication_date": paper.publication_date.isoformat() if paper.publication_date else None,
                "doi": paper.doi,
                "arxiv_id": paper.arxiv_id,
                "citation_count": paper.citation_count,
                "source": paper.source,
                "relevance": 1.0  # Placeholder for relevance scoring
            })
        
        return {
            "results": results,
            "total": total or 0,
            "query": q,
            "skip": skip,
            "limit": limit,
            "search_type": "basic_text",
            "message": "Enhanced search with Elasticsearch will be available in Phase 3"
        }
        
    except Exception as e:
        logger.error(f"Error searching papers: {e}")
        raise HTTPException(status_code=500, detail="Search failed")


@router.post("/ingest")
async def trigger_ingestion(
    background_tasks: BackgroundTasks,
    query: str = Query(..., description="Search query for ingestion"),
    sources: List[str] = Query(["arxiv"], description="Data sources to use"),
    max_results: int = Query(50, ge=1, le=500, description="Maximum results per source"),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Trigger data ingestion in the background."""
    try:
        # Load configuration
        config_path = Path("data/config/data_sources.yaml")
        if not config_path.exists():
            raise HTTPException(status_code=500, detail="Data sources configuration not found")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Start background ingestion
        background_tasks.add_task(
            _background_ingestion,
            config,
            query,
            sources,
            max_results
        )
        
        return {
            "message": "Data ingestion started in background",
            "query": query,
            "sources": sources,
            "max_results": max_results,
            "status": "started"
        }
        
    except Exception as e:
        logger.error(f"Error triggering ingestion: {e}")
        raise HTTPException(status_code=500, detail="Failed to start ingestion")


@router.get("/stats")
async def get_papers_stats(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Get papers statistics."""
    try:
        persistence_service = DataPersistenceService()
        stats = await persistence_service.get_ingestion_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")


async def _background_ingestion(
    config: dict,
    query: str,
    sources: List[str],
    max_results: int
):
    """Background task for data ingestion."""
    try:
        logger.info(f"Starting background ingestion: query='{query}', sources={sources}")
        
        ingestion_service = DataIngestionService(config)
        persistence_service = DataPersistenceService()
        
        result = await process_ingestion_batch(
            ingestion_service,
            persistence_service,
            query,
            sources,
            max_results
        )
        
        logger.info(f"Background ingestion completed: {result}")
        
    except Exception as e:
        logger.error(f"Background ingestion failed: {e}") 