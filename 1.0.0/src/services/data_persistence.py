"""
Data persistence service for storing ingested data to the database.
"""

import asyncio
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
import uuid

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func
from sqlalchemy.orm import selectinload

from src.core.logging import get_logger
from src.core.exceptions import DatabaseError
from src.data.database import get_db_manager
from src.data.models import Paper, Author, Institution, paper_authors, paper_institutions
from src.services.data_sources import PaperData
from src.utils.helpers import generate_uuid, normalize_string

logger = get_logger(__name__)


class DataPersistenceService:
    """Service for persisting ingested data to the database."""
    
    def __init__(self):
        self.db_manager = get_db_manager()
    
    async def save_paper(self, paper_data: PaperData, session: AsyncSession) -> Optional[Paper]:
        """Save a single paper to the database."""
        try:
            # Check if paper already exists
            existing_paper = await self._find_existing_paper(paper_data, session)
            if existing_paper:
                logger.debug(f"Paper already exists: {paper_data.id}")
                return existing_paper
            
            # Create new paper
            paper = Paper(
                id=uuid.uuid4(),
                external_id=paper_data.id,
                title=paper_data.title,
                abstract=paper_data.abstract,
                publication_date=paper_data.publication_date,
                journal=paper_data.journal,
                doi=paper_data.doi,
                arxiv_id=paper_data.arxiv_id,
                categories=paper_data.categories,
                keywords=paper_data.keywords,
                citation_count=paper_data.citation_count,
                download_count=paper_data.download_count,
                source=paper_data.source,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            session.add(paper)
            await session.flush()  # Get the paper ID
            
            # Process authors
            authors = await self._process_authors(paper_data.authors, session)
            for i, author in enumerate(authors):
                # Create paper-author relationship
                stmt = paper_authors.insert().values(
                    paper_id=paper.id,
                    author_id=author.id,
                    order=i,
                    is_corresponding=False  # Could be enhanced with more data
                )
                await session.execute(stmt)
            
            # Process institutions
            institutions = await self._process_institutions(paper_data.institutions, session)
            for institution in institutions:
                # Create paper-institution relationship
                stmt = paper_institutions.insert().values(
                    paper_id=paper.id,
                    institution_id=institution.id
                )
                await session.execute(stmt)
            
            await session.commit()
            logger.info(f"Saved paper: {paper_data.title[:50]}...")
            return paper
            
        except Exception as e:
            await session.rollback()
            logger.error(f"Failed to save paper {paper_data.id}: {e}")
            raise DatabaseError(f"Failed to save paper: {e}")
    
    async def save_papers_batch(self, papers_data: List[PaperData]) -> Dict[str, Any]:
        """Save a batch of papers to the database."""
        saved_count = 0
        error_count = 0
        duplicate_count = 0
        
        async with self.db_manager.get_async_session() as session:
            for paper_data in papers_data:
                try:
                    existing = await self._find_existing_paper(paper_data, session)
                    if existing:
                        duplicate_count += 1
                        continue
                    
                    await self.save_paper(paper_data, session)
                    saved_count += 1
                    
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error saving paper {paper_data.id}: {e}")
        
        result = {
            "saved": saved_count,
            "duplicates": duplicate_count,
            "errors": error_count,
            "total": len(papers_data)
        }
        
        logger.info(f"Batch save results: {result}")
        return result
    
    async def _find_existing_paper(self, paper_data: PaperData, session: AsyncSession) -> Optional[Paper]:
        """Find existing paper by various identifiers."""
        conditions = []
        
        # Check by external ID
        if paper_data.id:
            conditions.append(Paper.external_id == paper_data.id)
        
        # Check by DOI
        if paper_data.doi:
            conditions.append(Paper.doi == paper_data.doi)
        
        # Check by ArXiv ID
        if paper_data.arxiv_id:
            conditions.append(Paper.arxiv_id == paper_data.arxiv_id)
        
        # Check by title similarity (for papers without DOI/ArXiv)
        if paper_data.title and not paper_data.doi and not paper_data.arxiv_id:
            normalized_title = normalize_string(paper_data.title)
            conditions.append(Paper.title_normalized == normalized_title)
        
        if not conditions:
            return None
        
        stmt = select(Paper).where(or_(*conditions))
        result = await session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def _process_authors(self, authors_data: List[Dict[str, str]], session: AsyncSession) -> List[Author]:
        """Process and create/find authors."""
        authors = []
        
        for author_data in authors_data:
            author_name = author_data.get("name", "").strip()
            if not author_name:
                continue
            
            # Find or create author
            author = await self._find_or_create_author(author_data, session)
            if author:
                authors.append(author)
        
        return authors
    
    async def _find_or_create_author(self, author_data: Dict[str, str], session: AsyncSession) -> Optional[Author]:
        """Find existing author or create new one."""
        name = author_data.get("name", "").strip()
        email = author_data.get("email", "").strip()
        
        if not name:
            return None
        
        # Try to find existing author by name or email
        conditions = [Author.name == name]
        if email:
            conditions.append(Author.email == email)
        
        stmt = select(Author).where(or_(*conditions))
        result = await session.execute(stmt)
        existing_author = result.scalar_one_or_none()
        
        if existing_author:
            # Update author info if we have more complete data
            if email and not existing_author.email:
                existing_author.email = email
                existing_author.updated_at = datetime.utcnow()
            return existing_author
        
        # Create new author
        author = Author(
            id=uuid.uuid4(),
            name=name,
            email=email,
            orcid=None,  # Could be enhanced with ORCID lookup
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        session.add(author)
        await session.flush()
        logger.debug(f"Created new author: {name}")
        return author
    
    async def _process_institutions(self, institutions_data: List[Dict[str, str]], session: AsyncSession) -> List[Institution]:
        """Process and create/find institutions."""
        institutions = []
        
        # Extract unique institution names from author affiliations
        institution_names = set()
        for inst_data in institutions_data:
            name = inst_data.get("name", "").strip()
            if name:
                institution_names.add(name)
        
        for inst_name in institution_names:
            institution = await self._find_or_create_institution(inst_name, session)
            if institution:
                institutions.append(institution)
        
        return institutions
    
    async def _find_or_create_institution(self, name: str, session: AsyncSession) -> Optional[Institution]:
        """Find existing institution or create new one."""
        if not name:
            return None
        
        # Normalize institution name for comparison
        normalized_name = normalize_string(name)
        
        # Try to find existing institution
        stmt = select(Institution).where(Institution.name_normalized == normalized_name)
        result = await session.execute(stmt)
        existing_institution = result.scalar_one_or_none()
        
        if existing_institution:
            return existing_institution
        
        # Create new institution
        institution = Institution(
            id=uuid.uuid4(),
            name=name,
            name_normalized=normalized_name,
            country=None,  # Could be enhanced with country detection
            institution_type=None,  # Could be inferred from name patterns
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        session.add(institution)
        await session.flush()
        logger.debug(f"Created new institution: {name}")
        return institution
    
    async def get_ingestion_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics."""
        async with self.db_manager.get_async_session() as session:
            # Count papers by source
            stmt = select(Paper.source, func.count(Paper.id)).group_by(Paper.source)
            result = await session.execute(stmt)
            papers_by_source = dict(result.fetchall())
            
            # Total counts
            total_papers = await session.scalar(select(func.count(Paper.id)))
            total_authors = await session.scalar(select(func.count(Author.id)))
            total_institutions = await session.scalar(select(func.count(Institution.id)))
            
            # Recent papers (last 30 days)
            recent_cutoff = datetime.utcnow() - timedelta(days=30)
            recent_papers = await session.scalar(
                select(func.count(Paper.id)).where(Paper.created_at >= recent_cutoff)
            )
            
            return {
                "total_papers": total_papers or 0,
                "total_authors": total_authors or 0,
                "total_institutions": total_institutions or 0,
                "recent_papers": recent_papers or 0,
                "papers_by_source": papers_by_source,
                "last_updated": datetime.utcnow().isoformat()
            }


async def process_ingestion_batch(
    ingestion_service: Any,
    persistence_service: DataPersistenceService,
    query: str,
    sources: Optional[List[str]] = None,
    max_results: int = 100
) -> Dict[str, Any]:
    """Process a complete ingestion batch from sources to database."""
    logger.info(f"Starting ingestion batch: query='{query}', sources={sources}, max_results={max_results}")
    
    papers_data = []
    
    # Collect data from sources
    async for paper_data in ingestion_service.ingest_data(query, sources, max_results):
        papers_data.append(paper_data)
        
        # Process in batches to avoid memory issues
        if len(papers_data) >= 50:
            result = await persistence_service.save_papers_batch(papers_data)
            papers_data = []  # Clear the batch
    
    # Process remaining papers
    if papers_data:
        final_result = await persistence_service.save_papers_batch(papers_data)
    else:
        final_result = {"saved": 0, "duplicates": 0, "errors": 0, "total": 0}
    
    logger.info(f"Ingestion batch completed: {final_result}")
    return final_result 