"""
Data sources service for ingesting data from various research databases.
"""

import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, AsyncGenerator
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
import xml.etree.ElementTree as ET
import json
import re

from src.core.logging import get_logger
from src.core.exceptions import DataIngestionError
from src.utils.helpers import parse_doi, normalize_string, extract_keywords

logger = get_logger(__name__)


@dataclass
class PaperData:
    """Standardized paper data structure."""
    id: str
    title: str
    abstract: str
    authors: List[Dict[str, str]]
    institutions: List[Dict[str, str]]
    publication_date: Optional[datetime]
    journal: Optional[str]
    doi: Optional[str]
    arxiv_id: Optional[str]
    categories: List[str]
    keywords: List[str]
    citation_count: int = 0
    download_count: int = 0
    source: str = ""


class DataSource(ABC):
    """Abstract base class for data sources."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get("name", "Unknown")
        self.enabled = config.get("enabled", False)
        self.rate_limit = config.get("rate_limit", 1)
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self._session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()
    
    @abstractmethod
    async def fetch_papers(self, query: str, max_results: int = 100) -> AsyncGenerator[PaperData, None]:
        """Fetch papers from the data source."""
        pass
    
    async def _rate_limited_request(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make a rate-limited HTTP request."""
        await asyncio.sleep(1.0 / self.rate_limit)
        
        if not self._session:
            raise DataIngestionError("Session not initialized")
        
        try:
            response = await self._session.get(url, **kwargs)
            response.raise_for_status()
            return response
        except aiohttp.ClientError as e:
            logger.error(f"HTTP request failed for {url}: {e}")
            raise DataIngestionError(f"Request failed: {e}")


class ArXivDataSource(DataSource):
    """ArXiv data source implementation."""
    
    async def fetch_papers(self, query: str, max_results: int = 100) -> AsyncGenerator[PaperData, None]:
        """Fetch papers from ArXiv API."""
        if not self.enabled:
            logger.info(f"ArXiv data source is disabled")
            return
        
        base_url = self.config["base_url"]
        categories = self.config.get("categories", [])
        
        # Build category query
        if categories:
            cat_query = " OR ".join([f"cat:{cat}" for cat in categories])
            if query:
                full_query = f"({query}) AND ({cat_query})"
            else:
                full_query = cat_query
        else:
            full_query = query or "cat:hep-ex OR cat:hep-ph OR cat:hep-th"
        
        # ArXiv API parameters
        params = {
            "search_query": full_query,
            "start": 0,
            "max_results": min(max_results, self.config.get("max_results", 1000)),
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }
        
        try:
            response = await self._rate_limited_request(base_url, params=params)
            content = await response.text()
            
            # Parse XML response
            root = ET.fromstring(content)
            
            # Define namespaces
            ns = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            for entry in root.findall('atom:entry', ns):
                try:
                    paper_data = await self._parse_arxiv_entry(entry, ns)
                    if paper_data:
                        yield paper_data
                except Exception as e:
                    logger.warning(f"Failed to parse ArXiv entry: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to fetch from ArXiv: {e}")
            raise DataIngestionError(f"ArXiv fetch failed: {e}")
    
    async def _parse_arxiv_entry(self, entry: ET.Element, ns: Dict[str, str]) -> Optional[PaperData]:
        """Parse a single ArXiv entry."""
        try:
            # Extract basic information
            title_elem = entry.find('atom:title', ns)
            title = title_elem.text.strip() if title_elem is not None else ""
            
            summary_elem = entry.find('atom:summary', ns)
            abstract = summary_elem.text.strip() if summary_elem is not None else ""
            
            # Extract ArXiv ID
            id_elem = entry.find('atom:id', ns)
            arxiv_url = id_elem.text if id_elem is not None else ""
            arxiv_id = arxiv_url.split('/')[-1] if arxiv_url else ""
            
            # Extract publication date
            published_elem = entry.find('atom:published', ns)
            pub_date = None
            if published_elem is not None:
                try:
                    pub_date = datetime.fromisoformat(published_elem.text.replace('Z', '+00:00'))
                except ValueError:
                    pass
            
            # Extract authors
            authors = []
            for author_elem in entry.findall('atom:author', ns):
                name_elem = author_elem.find('atom:name', ns)
                if name_elem is not None:
                    authors.append({
                        "name": name_elem.text.strip(),
                        "affiliation": "",  # ArXiv doesn't provide affiliation in API
                        "email": ""
                    })
            
            # Extract categories
            categories = []
            for category_elem in entry.findall('atom:category', ns):
                term = category_elem.get('term')
                if term:
                    categories.append(term)
            
            # Extract DOI if available
            doi = None
            for link_elem in entry.findall('atom:link', ns):
                if link_elem.get('title') == 'doi':
                    doi_url = link_elem.get('href', '')
                    doi = parse_doi(doi_url)
                    break
            
            # Extract keywords from title and abstract
            keywords = extract_keywords(f"{title} {abstract}")
            
            return PaperData(
                id=f"arxiv:{arxiv_id}",
                title=title,
                abstract=abstract,
                authors=authors,
                institutions=[],  # Will be populated later
                publication_date=pub_date,
                journal=None,  # ArXiv preprints don't have journals initially
                doi=doi,
                arxiv_id=arxiv_id,
                categories=categories,
                keywords=keywords,
                source="arxiv"
            )
            
        except Exception as e:
            logger.error(f"Error parsing ArXiv entry: {e}")
            return None


class InspireHEPDataSource(DataSource):
    """INSPIRE-HEP data source implementation."""
    
    async def fetch_papers(self, query: str, max_results: int = 100) -> AsyncGenerator[PaperData, None]:
        """Fetch papers from INSPIRE-HEP API."""
        if not self.enabled:
            logger.info(f"INSPIRE-HEP data source is disabled")
            return
        
        base_url = self.config["base_url"]
        literature_endpoint = self.config["endpoints"]["literature"]
        
        # INSPIRE-HEP API parameters
        params = {
            "q": query or "subject:hep-ex OR subject:hep-ph OR subject:hep-th",
            "size": min(max_results, 250),  # INSPIRE-HEP max per request
            "sort": "mostrecent",
            "format": "json"
        }
        
        try:
            url = f"{base_url}{literature_endpoint}"
            response = await self._rate_limited_request(url, params=params)
            data = await response.json()
            
            hits = data.get("hits", {}).get("hits", [])
            
            for hit in hits:
                try:
                    paper_data = await self._parse_inspire_record(hit)
                    if paper_data:
                        yield paper_data
                except Exception as e:
                    logger.warning(f"Failed to parse INSPIRE-HEP record: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to fetch from INSPIRE-HEP: {e}")
            raise DataIngestionError(f"INSPIRE-HEP fetch failed: {e}")
    
    async def _parse_inspire_record(self, record: Dict[str, Any]) -> Optional[PaperData]:
        """Parse a single INSPIRE-HEP record."""
        try:
            metadata = record.get("metadata", {})
            
            # Extract basic information
            titles = metadata.get("titles", [])
            title = titles[0].get("title", "") if titles else ""
            
            abstracts = metadata.get("abstracts", [])
            abstract = abstracts[0].get("value", "") if abstracts else ""
            
            # Extract ArXiv ID and DOI
            arxiv_id = None
            doi = None
            
            arxiv_eprints = metadata.get("arxiv_eprints", [])
            if arxiv_eprints:
                arxiv_id = arxiv_eprints[0].get("value")
            
            dois = metadata.get("dois", [])
            if dois:
                doi = dois[0].get("value")
            
            # Extract authors
            authors = []
            for author_data in metadata.get("authors", []):
                full_name = author_data.get("full_name", "")
                affiliations = author_data.get("affiliations", [])
                affiliation = affiliations[0].get("value", "") if affiliations else ""
                
                authors.append({
                    "name": full_name,
                    "affiliation": affiliation,
                    "email": ""  # Not typically provided
                })
            
            # Extract publication info
            pub_date = None
            journal = None
            
            imprints = metadata.get("imprints", [])
            if imprints:
                date_str = imprints[0].get("date")
                if date_str:
                    try:
                        pub_date = datetime.strptime(date_str, "%Y-%m-%d")
                    except ValueError:
                        pass
            
            publication_info = metadata.get("publication_info", [])
            if publication_info:
                journal = publication_info[0].get("journal_title", "")
            
            # Extract subjects/categories
            inspire_categories = metadata.get("inspire_categories", [])
            categories = [cat.get("term", "") for cat in inspire_categories]
            
            # Citation count
            citation_count = metadata.get("citation_count", 0)
            
            # Generate keywords
            keywords = extract_keywords(f"{title} {abstract}")
            
            # Generate ID
            paper_id = f"inspire:{metadata.get('control_number', '')}"
            if arxiv_id:
                paper_id = f"arxiv:{arxiv_id}"
            
            return PaperData(
                id=paper_id,
                title=title,
                abstract=abstract,
                authors=authors,
                institutions=[],  # Will be populated separately
                publication_date=pub_date,
                journal=journal,
                doi=doi,
                arxiv_id=arxiv_id,
                categories=categories,
                keywords=keywords,
                citation_count=citation_count,
                source="inspire-hep"
            )
            
        except Exception as e:
            logger.error(f"Error parsing INSPIRE-HEP record: {e}")
            return None


class DataIngestionService:
    """Main service for coordinating data ingestion from multiple sources."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sources: Dict[str, DataSource] = {}
        self._initialize_sources()
    
    def _initialize_sources(self):
        """Initialize data sources from configuration."""
        data_sources_config = self.config.get("data_sources", {})
        
        if "arxiv" in data_sources_config:
            self.sources["arxiv"] = ArXivDataSource(data_sources_config["arxiv"])
        
        if "inspire_hep" in data_sources_config:
            self.sources["inspire_hep"] = InspireHEPDataSource(data_sources_config["inspire_hep"])
    
    async def ingest_data(
        self, 
        query: str, 
        sources: Optional[List[str]] = None, 
        max_results_per_source: int = 100
    ) -> AsyncGenerator[PaperData, None]:
        """Ingest data from specified sources."""
        if sources is None:
            sources = list(self.sources.keys())
        
        logger.info(f"Starting data ingestion from sources: {sources}")
        
        for source_name in sources:
            if source_name not in self.sources:
                logger.warning(f"Unknown data source: {source_name}")
                continue
            
            source = self.sources[source_name]
            if not source.enabled:
                logger.info(f"Skipping disabled source: {source_name}")
                continue
            
            logger.info(f"Ingesting from {source_name}")
            
            try:
                async with source:
                    async for paper in source.fetch_papers(query, max_results_per_source):
                        yield paper
            except Exception as e:
                logger.error(f"Error ingesting from {source_name}: {e}")
                continue
        
        logger.info("Data ingestion completed") 