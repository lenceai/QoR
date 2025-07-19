"""
Data models for CERN Knowledge Explorer.
"""

import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum

from sqlalchemy import (
    Boolean, Column, DateTime, String, Text, Integer, Float, 
    JSON, ForeignKey, Table, Index, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from src.data.database import Base


# Association tables for many-to-many relationships
paper_authors = Table(
    'paper_authors',
    Base.metadata,
    Column('paper_id', UUID(as_uuid=True), ForeignKey('papers.id'), primary_key=True),
    Column('author_id', UUID(as_uuid=True), ForeignKey('authors.id'), primary_key=True),
    Column('order', Integer, default=0),
    Column('is_corresponding', Boolean, default=False)
)

paper_keywords = Table(
    'paper_keywords',
    Base.metadata,
    Column('paper_id', UUID(as_uuid=True), ForeignKey('papers.id'), primary_key=True),
    Column('keyword_id', UUID(as_uuid=True), ForeignKey('keywords.id'), primary_key=True)
)

paper_institutions = Table(
    'paper_institutions',
    Base.metadata,
    Column('paper_id', UUID(as_uuid=True), ForeignKey('papers.id'), primary_key=True),
    Column('institution_id', UUID(as_uuid=True), ForeignKey('institutions.id'), primary_key=True)
)

author_institutions = Table(
    'author_institutions',
    Base.metadata,
    Column('author_id', UUID(as_uuid=True), ForeignKey('authors.id'), primary_key=True),
    Column('institution_id', UUID(as_uuid=True), ForeignKey('institutions.id'), primary_key=True),
    Column('start_date', DateTime),
    Column('end_date', DateTime),
    Column('is_primary', Boolean, default=False)
)


class PublicationType(str, Enum):
    """Types of publications."""
    ARTICLE = "article"
    CONFERENCE_PAPER = "conference_paper"
    THESIS = "thesis"
    PREPRINT = "preprint"
    BOOK = "book"
    BOOK_CHAPTER = "book_chapter"
    TECHNICAL_REPORT = "technical_report"
    DATASET = "dataset"
    SOFTWARE = "software"


class ExperimentStatus(str, Enum):
    """Status of experiments."""
    PLANNING = "planning"
    ACTIVE = "active"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class User(Base):
    """User model for authentication and personalization."""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    
    # Profile information
    orcid_id = Column(String(50), unique=True, index=True)
    institution_id = Column(UUID(as_uuid=True), ForeignKey('institutions.id'))
    department = Column(String(255))
    research_interests = Column(JSON)  # Store as JSON array for SQLite compatibility
    
    # Account status
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    is_admin = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_login = Column(DateTime(timezone=True))
    
    # Relationships
    institution = relationship("Institution", back_populates="users")
    saved_searches = relationship("SavedSearch", back_populates="user")
    collections = relationship("Collection", back_populates="user")


class Institution(Base):
    """Institution/Organization model."""
    __tablename__ = "institutions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(500), nullable=False, index=True)
    short_name = Column(String(100), index=True)
    country = Column(String(100), index=True)
    city = Column(String(100))
    
    # External identifiers
    ror_id = Column(String(50), unique=True, index=True)  # Research Organization Registry
    grid_id = Column(String(50), unique=True, index=True)  # Global Research Identifier Database
    
    # Institution details
    type = Column(String(100))  # university, research_institute, company, etc.
    website = Column(String(500))
    established_year = Column(Integer)
    
    # Additional metadata
    extra_data = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    users = relationship("User", back_populates="institution")
    papers = relationship("Paper", secondary=paper_institutions, back_populates="institutions")
    authors = relationship("Author", secondary=author_institutions, back_populates="institutions")


class Author(Base):
    """Author model."""
    __tablename__ = "authors"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    full_name = Column(String(255), nullable=False, index=True)
    given_name = Column(String(100))
    family_name = Column(String(155))
    
    # External identifiers
    orcid_id = Column(String(50), unique=True, index=True)
    inspire_id = Column(String(50), unique=True, index=True)
    
    # Contact information
    email = Column(String(255), index=True)
    
    # Research information
    research_areas = Column(JSON)  # Store as JSON array for SQLite compatibility
    h_index = Column(Integer)
    citation_count = Column(Integer, default=0)
    
    # Full-text search (using Text for SQLite compatibility)
    search_vector = Column(Text)
    
    # Additional metadata
    extra_data = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    papers = relationship("Paper", secondary=paper_authors, back_populates="authors")
    institutions = relationship("Institution", secondary=author_institutions, back_populates="authors")
    
    # Indexes
    __table_args__ = (
        Index('idx_author_name', 'full_name'),
    )


class Paper(Base):
    """Scientific paper/publication model."""
    __tablename__ = "papers"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(Text, nullable=False, index=True)
    abstract = Column(Text)
    
    # Publication details
    publication_type = Column(String(50), default=PublicationType.ARTICLE)
    journal = Column(String(500))
    volume = Column(String(50))
    issue = Column(String(50))
    pages = Column(String(100))
    publisher = Column(String(255))
    
    # Dates
    publication_date = Column(DateTime)
    submitted_date = Column(DateTime)
    accepted_date = Column(DateTime)
    
    # External identifiers
    doi = Column(String(255), unique=True, index=True)
    arxiv_id = Column(String(50), unique=True, index=True)
    pmid = Column(String(50), unique=True, index=True)  # PubMed ID
    inspire_id = Column(String(50), unique=True, index=True)
    
    # Content
    full_text = Column(Text)
    language = Column(String(10), default='en')
    
    # Metrics
    citation_count = Column(Integer, default=0)
    download_count = Column(Integer, default=0)
    view_count = Column(Integer, default=0)
    
    # URLs and files
    pdf_url = Column(String(1000))
    html_url = Column(String(1000))
    source_url = Column(String(1000))
    
    # Full-text search (using Text for SQLite compatibility)
    search_vector = Column(Text)
    
    # Experiment association
    experiment_id = Column(UUID(as_uuid=True), ForeignKey('experiments.id'))
    
    # Additional fields and metadata
    extra_data = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    authors = relationship("Author", secondary=paper_authors, back_populates="papers")
    keywords = relationship("Keyword", secondary=paper_keywords, back_populates="papers")
    institutions = relationship("Institution", secondary=paper_institutions, back_populates="papers")
    experiment = relationship("Experiment", back_populates="papers")
    citations = relationship("Citation", foreign_keys="Citation.citing_paper_id", back_populates="citing_paper")
    cited_by = relationship("Citation", foreign_keys="Citation.cited_paper_id", back_populates="cited_paper")
    
    # Indexes
    __table_args__ = (
        Index('idx_paper_title', 'title'),
        Index('idx_paper_publication_date', 'publication_date'),
        Index('idx_paper_doi', 'doi'),
    )


class Keyword(Base):
    """Keyword/tag model."""
    __tablename__ = "keywords"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, unique=True, index=True)
    normalized_name = Column(String(255), index=True)  # Lowercase, cleaned version
    category = Column(String(100))  # subject, method, instrument, etc.
    
    # Usage statistics
    usage_count = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    papers = relationship("Paper", secondary=paper_keywords, back_populates="keywords")


class Experiment(Base):
    """CERN experiment model."""
    __tablename__ = "experiments"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, unique=True, index=True)
    full_name = Column(String(500))
    description = Column(Text)
    
    # Experiment details
    status = Column(String(50), default=ExperimentStatus.ACTIVE)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    
    # Location and infrastructure
    location = Column(String(255))
    detector = Column(String(255))
    accelerator = Column(String(255))
    
    # External links
    website = Column(String(500))
    collaboration_size = Column(Integer)
    
    # Additional metadata
    extra_data = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    papers = relationship("Paper", back_populates="experiment")


class Citation(Base):
    """Citation relationship between papers."""
    __tablename__ = "citations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    citing_paper_id = Column(UUID(as_uuid=True), ForeignKey('papers.id'), nullable=False)
    cited_paper_id = Column(UUID(as_uuid=True), ForeignKey('papers.id'), nullable=False)
    
    # Citation context
    context = Column(Text)  # Text around the citation
    citation_type = Column(String(50))  # direct, indirect, etc.
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    citing_paper = relationship("Paper", foreign_keys=[citing_paper_id], back_populates="citations")
    cited_paper = relationship("Paper", foreign_keys=[cited_paper_id], back_populates="cited_by")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('citing_paper_id', 'cited_paper_id', name='unique_citation'),
        Index('idx_citation_citing', 'citing_paper_id'),
        Index('idx_citation_cited', 'cited_paper_id'),
    )


class SavedSearch(Base):
    """User saved searches."""
    __tablename__ = "saved_searches"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Search parameters
    query = Column(JSON, nullable=False)  # Store search parameters as JSON
    filters = Column(JSON)
    
    # Settings
    is_alert = Column(Boolean, default=False)  # Send notifications for new results
    alert_frequency = Column(String(50))  # daily, weekly, monthly
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_run = Column(DateTime(timezone=True))
    
    # Relationships
    user = relationship("User", back_populates="saved_searches")


class Collection(Base):
    """User collections of papers."""
    __tablename__ = "collections"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Collection settings
    is_public = Column(Boolean, default=False)
    is_collaborative = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="collections")
    items = relationship("CollectionItem", back_populates="collection")


class CollectionItem(Base):
    """Items in user collections."""
    __tablename__ = "collection_items"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    collection_id = Column(UUID(as_uuid=True), ForeignKey('collections.id'), nullable=False)
    paper_id = Column(UUID(as_uuid=True), ForeignKey('papers.id'), nullable=False)
    
    # Item metadata
    notes = Column(Text)
    tags = Column(JSON)  # Store as JSON array for SQLite compatibility
    
    # Timestamps
    added_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    collection = relationship("Collection", back_populates="items")
    paper = relationship("Paper")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('collection_id', 'paper_id', name='unique_collection_item'),
    )


class DataSource(Base):
    """Data sources for ingestion tracking."""
    __tablename__ = "data_sources"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, unique=True)
    url = Column(String(1000))
    description = Column(Text)
    
    # Source configuration
    source_type = Column(String(100))  # api, scraper, manual, etc.
    is_active = Column(Boolean, default=True)
    
    # Ingestion statistics
    last_sync = Column(DateTime(timezone=True))
    total_records = Column(Integer, default=0)
    
    # Configuration
    config = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now()) 