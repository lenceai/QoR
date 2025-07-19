"""
Tests for Phase 2 - Data Ingestion functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

# Import from src
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.services.data_sources import DataIngestionService, PaperData, ArXivDataSource
from src.services.data_persistence import DataPersistenceService
from src.utils.helpers import parse_doi, extract_keywords


def test_paper_data_creation():
    """Test PaperData dataclass creation."""
    paper = PaperData(
        id="test-001",
        title="Test Paper",
        abstract="This is a test abstract",
        authors=[{"name": "John Doe", "affiliation": "Test University", "email": ""}],
        institutions=[{"name": "Test University", "country": "US", "type": "university"}],
        publication_date=datetime(2023, 1, 1),
        journal="Test Journal",
        doi="10.1000/test.001",
        arxiv_id="2301.00001",
        categories=["hep-ex"],
        keywords=["test", "paper"],
        source="test"
    )
    
    assert paper.id == "test-001"
    assert paper.title == "Test Paper"
    assert len(paper.authors) == 1
    assert paper.authors[0]["name"] == "John Doe"


def test_doi_parsing():
    """Test DOI parsing utility function."""
    # Valid DOIs
    assert parse_doi("10.1103/PhysRevLett.130.111801") == "10.1103/PhysRevLett.130.111801"
    assert parse_doi("doi:10.1103/PhysRevLett.130.111801") == "10.1103/PhysRevLett.130.111801"
    assert parse_doi("https://doi.org/10.1103/PhysRevLett.130.111801") == "10.1103/PhysRevLett.130.111801"
    
    # Invalid DOIs
    assert parse_doi("not-a-doi") is None
    assert parse_doi("") is None


def test_keyword_extraction():
    """Test keyword extraction from text."""
    text = "This paper discusses the Higgs boson discovery at the Large Hadron Collider"
    keywords = extract_keywords(text)
    
    assert "higgs" in keywords
    assert "boson" in keywords
    assert "discovery" in keywords
    assert "large" in keywords
    assert "hadron" in keywords
    assert "collider" in keywords
    
    # Stop words should be filtered out
    assert "the" not in keywords
    assert "at" not in keywords


@pytest.mark.asyncio
async def test_arxiv_data_source_initialization():
    """Test ArXiv data source initialization."""
    config = {
        "name": "ArXiv Test",
        "enabled": True,
        "base_url": "http://export.arxiv.org/api/query",
        "categories": ["hep-ex", "hep-ph"],
        "max_results": 100,
        "rate_limit": 3
    }
    
    source = ArXivDataSource(config)
    assert source.name == "ArXiv Test"
    assert source.enabled is True
    assert source.rate_limit == 3


@pytest.mark.asyncio
async def test_data_ingestion_service_initialization():
    """Test DataIngestionService initialization."""
    config = {
        "data_sources": {
            "arxiv": {
                "name": "ArXiv",
                "enabled": True,
                "base_url": "http://export.arxiv.org/api/query",
                "categories": ["hep-ex"],
                "rate_limit": 3
            }
        }
    }
    
    service = DataIngestionService(config)
    assert "arxiv" in service.sources
    assert service.sources["arxiv"].enabled is True


@pytest.mark.asyncio
async def test_data_persistence_service_initialization():
    """Test DataPersistenceService initialization."""
    # Mock database manager
    with patch('src.services.data_persistence.get_db_manager') as mock_db_manager:
        mock_db_manager.return_value = Mock()
        
        service = DataPersistenceService()
        assert service.db_manager is not None


class TestDataIngestionIntegration:
    """Integration tests for data ingestion."""
    
    @pytest.mark.asyncio
    async def test_sample_data_format(self):
        """Test that sample data matches expected format."""
        # Load sample data
        sample_file = Path(__file__).parent.parent / "data" / "samples" / "sample_papers.json"
        
        if sample_file.exists():
            import json
            with open(sample_file) as f:
                sample_data = json.load(f)
            
            # Verify structure
            assert isinstance(sample_data, list)
            assert len(sample_data) > 0
            
            paper = sample_data[0]
            required_fields = ["id", "title", "abstract", "authors", "institutions"]
            for field in required_fields:
                assert field in paper, f"Missing required field: {field}"
            
            # Verify authors structure
            assert isinstance(paper["authors"], list)
            if paper["authors"]:
                author = paper["authors"][0]
                assert "name" in author
                assert "affiliation" in author
            
            # Verify institutions structure
            assert isinstance(paper["institutions"], list)
            if paper["institutions"]:
                institution = paper["institutions"][0]
                assert "name" in institution
                assert "country" in institution


def test_data_source_config_loading():
    """Test data source configuration loading."""
    config_file = Path(__file__).parent.parent / "data" / "config" / "data_sources.yaml"
    
    if config_file.exists():
        import yaml
        with open(config_file) as f:
            config = yaml.safe_load(f)
        
        # Verify structure
        assert "data_sources" in config
        assert isinstance(config["data_sources"], dict)
        
        # Check ArXiv configuration
        if "arxiv" in config["data_sources"]:
            arxiv_config = config["data_sources"]["arxiv"]
            assert "name" in arxiv_config
            assert "base_url" in arxiv_config
            assert "categories" in arxiv_config


class MockArXivResponse:
    """Mock ArXiv API response for testing."""
    
    @staticmethod
    def get_sample_xml():
        return '''<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2301.00001v1</id>
    <title>Test Paper Title</title>
    <summary>This is a test abstract for a physics paper.</summary>
    <published>2023-01-01T00:00:00Z</published>
    <author>
      <name>John Doe</name>
    </author>
    <author>
      <name>Jane Smith</name>
    </author>
    <category term="hep-ex" scheme="http://arxiv.org/schemas/atom"/>
    <link href="http://arxiv.org/abs/2301.00001v1" rel="alternate" type="text/html"/>
  </entry>
</feed>'''


@pytest.mark.asyncio
async def test_arxiv_parsing():
    """Test ArXiv XML parsing."""
    import xml.etree.ElementTree as ET
    
    xml_content = MockArXivResponse.get_sample_xml()
    root = ET.fromstring(xml_content)
    
    ns = {
        'atom': 'http://www.w3.org/2005/Atom',
        'arxiv': 'http://arxiv.org/schemas/atom'
    }
    
    entry = root.find('atom:entry', ns)
    assert entry is not None
    
    # Test title extraction
    title_elem = entry.find('atom:title', ns)
    assert title_elem is not None
    assert title_elem.text == "Test Paper Title"
    
    # Test authors extraction
    authors = []
    for author_elem in entry.findall('atom:author', ns):
        name_elem = author_elem.find('atom:name', ns)
        if name_elem is not None:
            authors.append(name_elem.text)
    
    assert len(authors) == 2
    assert "John Doe" in authors
    assert "Jane Smith" in authors 