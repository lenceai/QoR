"""
Indexing service to sync database data with Elasticsearch for CERN Knowledge Explorer.
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload

from src.core.logging import get_logger
from src.core.exceptions import IndexingError
from src.data.database import get_db_manager
from src.data.models import Paper, Author, Institution
from src.services.search_service import search_service

logger = get_logger(__name__)


class IndexingService:
    """Service for indexing database content to Elasticsearch."""
    
    def __init__(self):
        self.db_manager = get_db_manager()
        self.search_service = search_service
    
    async def initialize(self):
        """Initialize the indexing service."""
        await self.search_service.initialize()
    
    async def index_all_papers(self, batch_size: int = 100) -> Dict[str, int]:
        """Index all papers from database to Elasticsearch."""
        logger.info("Starting full paper indexing...")
        
        total_indexed = 0
        total_errors = 0
        
        async with self.db_manager.get_async_session() as session:
            # Get total count
            total_count = await session.scalar(select(func.count(Paper.id)))
            logger.info(f"Found {total_count} papers to index")
            
            # Process in batches
            offset = 0
            while offset < total_count:
                logger.info(f"Processing batch {offset // batch_size + 1}, papers {offset + 1}-{min(offset + batch_size, total_count)}")
                
                # Fetch batch of papers with relationships
                stmt = (
                    select(Paper)
                    .options(
                        selectinload(Paper.authors),
                        selectinload(Paper.institutions)
                    )
                    .offset(offset)
                    .limit(batch_size)
                    .order_by(Paper.created_at)
                )
                
                result = await session.execute(stmt)
                papers = result.scalars().all()
                
                if not papers:
                    break
                
                # Convert to documents and index
                paper_docs = [self._paper_to_document(paper) for paper in papers]
                
                # Bulk index to Elasticsearch
                result = await self.search_service.bulk_index_papers(paper_docs)
                total_indexed += result["indexed"]
                total_errors += result["errors"]
                
                offset += batch_size
                
                # Small delay to avoid overwhelming Elasticsearch
                await asyncio.sleep(0.1)
        
        # Refresh index to make changes visible
        await self.search_service.refresh_index()
        
        logger.info(f"Full indexing completed: {total_indexed} indexed, {total_errors} errors")
        return {"indexed": total_indexed, "errors": total_errors, "total": total_count}
    
    async def index_paper(self, paper_id: str) -> bool:
        """Index a single paper by ID."""
        try:
            async with self.db_manager.get_async_session() as session:
                # Fetch paper with relationships
                stmt = (
                    select(Paper)
                    .options(
                        selectinload(Paper.authors),
                        selectinload(Paper.institutions)
                    )
                    .where(Paper.id == paper_id)
                )
                
                result = await session.execute(stmt)
                paper = result.scalar_one_or_none()
                
                if not paper:
                    logger.warning(f"Paper not found: {paper_id}")
                    return False
                
                # Convert to document and index
                paper_doc = self._paper_to_document(paper)
                success = await self.search_service.index_paper(paper_doc)
                
                if success:
                    logger.info(f"Successfully indexed paper: {paper_id}")
                    await self.search_service.refresh_index(self.search_service.papers_index)
                
                return success
                
        except Exception as e:
            logger.error(f"Failed to index paper {paper_id}: {e}")
            return False
    
    async def index_new_papers(self, since: Optional[datetime] = None) -> Dict[str, int]:
        """Index papers created or updated since a specific time."""
        if since is None:
            # Default to papers from last hour
            from datetime import timedelta
            since = datetime.utcnow() - timedelta(hours=1)
        
        logger.info(f"Indexing papers since {since}")
        
        async with self.db_manager.get_async_session() as session:
            # Find papers to index
            stmt = (
                select(Paper)
                .options(
                    selectinload(Paper.authors),
                    selectinload(Paper.institutions)
                )
                .where(
                    (Paper.created_at >= since) | (Paper.updated_at >= since)
                )
                .order_by(Paper.updated_at.desc())
            )
            
            result = await session.execute(stmt)
            papers = result.scalars().all()
            
            if not papers:
                logger.info("No new papers to index")
                return {"indexed": 0, "errors": 0, "total": 0}
            
            logger.info(f"Found {len(papers)} papers to index")
            
            # Convert and bulk index
            paper_docs = [self._paper_to_document(paper) for paper in papers]
            result = await self.search_service.bulk_index_papers(paper_docs)
            
            # Refresh index
            await self.search_service.refresh_index(self.search_service.papers_index)
            
            logger.info(f"Indexed {result['indexed']} new papers, {result['errors']} errors")
            return {**result, "total": len(papers)}
    
    async def remove_paper(self, paper_id: str) -> bool:
        """Remove a paper from the search index."""
        try:
            success = await self.search_service.delete_paper(paper_id)
            if success:
                logger.info(f"Removed paper from index: {paper_id}")
                await self.search_service.refresh_index(self.search_service.papers_index)
            return success
            
        except Exception as e:
            logger.error(f"Failed to remove paper {paper_id}: {e}")
            return False
    
    def _paper_to_document(self, paper: Paper) -> Dict[str, Any]:
        """Convert a Paper model to an Elasticsearch document."""
        return {
            "id": str(paper.id),
            "external_id": paper.external_id,
            "title": paper.title,
            "abstract": paper.abstract,
            "publication_date": paper.publication_date.isoformat() if paper.publication_date else None,
            "journal": paper.journal,
            "doi": paper.doi,
            "arxiv_id": paper.arxiv_id,
            "categories": paper.categories or [],
            "keywords": paper.keywords or [],
            "citation_count": paper.citation_count or 0,
            "download_count": paper.download_count or 0,
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
                    "id": str(institution.id),
                    "name": institution.name,
                    "country": institution.country,
                    "type": institution.institution_type
                } for institution in paper.institutions
            ],
            "created_at": paper.created_at.isoformat(),
            "updated_at": paper.updated_at.isoformat()
        }
    
    async def reindex_all(self) -> Dict[str, int]:
        """Drop and recreate all indices, then reindex all data."""
        logger.info("Starting full reindexing...")
        
        try:
            # Delete existing indices
            if await self.search_service.client.indices.exists(index=self.search_service.papers_index):
                await self.search_service.client.indices.delete(index=self.search_service.papers_index)
                logger.info(f"Deleted existing index: {self.search_service.papers_index}")
            
            # Recreate indices
            await self.search_service._create_indices()
            
            # Index all papers
            result = await self.index_all_papers()
            
            logger.info("Full reindexing completed")
            return result
            
        except Exception as e:
            logger.error(f"Reindexing failed: {e}")
            raise IndexingError(f"Reindexing failed: {e}")
    
    async def get_indexing_status(self) -> Dict[str, Any]:
        """Get current indexing status and statistics."""
        try:
            # Database stats
            async with self.db_manager.get_async_session() as session:
                total_papers_db = await session.scalar(select(func.count(Paper.id)))
                recent_papers_db = await session.scalar(
                    select(func.count(Paper.id)).where(
                        Paper.created_at >= datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                    )
                )
            
            # Elasticsearch stats
            es_stats = {"papers": 0, "authors": 0, "institutions": 0}
            
            if self.search_service.client:
                # Papers index stats
                try:
                    papers_stats = await self.search_service.client.count(
                        index=self.search_service.papers_index
                    )
                    es_stats["papers"] = papers_stats["count"]
                except Exception:
                    pass
                
                # Authors index stats
                try:
                    authors_stats = await self.search_service.client.count(
                        index=self.search_service.authors_index
                    )
                    es_stats["authors"] = authors_stats["count"]
                except Exception:
                    pass
                
                # Institutions index stats
                try:
                    institutions_stats = await self.search_service.client.count(
                        index=self.search_service.institutions_index
                    )
                    es_stats["institutions"] = institutions_stats["count"]
                except Exception:
                    pass
            
            return {
                "database": {
                    "total_papers": total_papers_db or 0,
                    "recent_papers": recent_papers_db or 0
                },
                "elasticsearch": es_stats,
                "sync_status": {
                    "papers_synced": es_stats["papers"] == (total_papers_db or 0),
                    "last_check": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get indexing status: {e}")
            return {
                "database": {"total_papers": 0, "recent_papers": 0},
                "elasticsearch": {"papers": 0, "authors": 0, "institutions": 0},
                "sync_status": {"papers_synced": False, "last_check": datetime.utcnow().isoformat()},
                "error": str(e)
            }
    
    async def sync_incremental(self, check_interval_hours: int = 1) -> Dict[str, int]:
        """Perform incremental sync of recent changes."""
        from datetime import timedelta
        
        since = datetime.utcnow() - timedelta(hours=check_interval_hours)
        return await self.index_new_papers(since)


# Global indexing service instance
indexing_service = IndexingService() 