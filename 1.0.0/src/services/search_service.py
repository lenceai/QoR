"""
Search service with Elasticsearch integration for CERN Knowledge Explorer.
"""

import json
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import NotFoundError, RequestError

from src.core.logging import get_logger
from src.core.config import settings
from src.core.exceptions import SearchError
from src.data.models import Paper
from src.utils.helpers import normalize_string

logger = get_logger(__name__)


class SearchService:
    """Elasticsearch-based search service."""
    
    def __init__(self):
        self.client: Optional[AsyncElasticsearch] = None
        self.index_prefix = settings.ELASTICSEARCH_INDEX_PREFIX
        self.papers_index = f"{self.index_prefix}_papers"
        self.authors_index = f"{self.index_prefix}_authors"
        self.institutions_index = f"{self.index_prefix}_institutions"
    
    async def initialize(self):
        """Initialize Elasticsearch client and create indices."""
        try:
            self.client = AsyncElasticsearch(
                [settings.ELASTICSEARCH_URL],
                verify_certs=False,
                ssl_show_warn=False
            )
            
            # Test connection
            info = await self.client.info()
            logger.info(f"Connected to Elasticsearch cluster: {info['cluster_name']}")
            
            # Create indices if they don't exist
            await self._create_indices()
            
        except Exception as e:
            logger.error(f"Failed to initialize Elasticsearch: {e}")
            raise SearchError(f"Elasticsearch initialization failed: {e}")
    
    async def close(self):
        """Close Elasticsearch client."""
        if self.client:
            await self.client.close()
    
    async def _create_indices(self):
        """Create Elasticsearch indices with mappings."""
        # Papers index mapping
        papers_mapping = {
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "external_id": {"type": "keyword"},
                    "title": {
                        "type": "text",
                        "analyzer": "standard",
                        "fields": {
                            "keyword": {"type": "keyword"},
                            "suggest": {"type": "completion"}
                        }
                    },
                    "abstract": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "publication_date": {"type": "date"},
                    "journal": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}}
                    },
                    "doi": {"type": "keyword"},
                    "arxiv_id": {"type": "keyword"},
                    "categories": {"type": "keyword"},
                    "keywords": {"type": "keyword"},
                    "citation_count": {"type": "integer"},
                    "download_count": {"type": "integer"},
                    "source": {"type": "keyword"},
                    "authors": {
                        "type": "nested",
                        "properties": {
                            "id": {"type": "keyword"},
                            "name": {
                                "type": "text",
                                "fields": {"keyword": {"type": "keyword"}}
                            },
                            "email": {"type": "keyword"},
                            "orcid": {"type": "keyword"}
                        }
                    },
                    "institutions": {
                        "type": "nested",
                        "properties": {
                            "id": {"type": "keyword"},
                            "name": {
                                "type": "text",
                                "fields": {"keyword": {"type": "keyword"}}
                            },
                            "country": {"type": "keyword"},
                            "type": {"type": "keyword"}
                        }
                    },
                    "created_at": {"type": "date"},
                    "updated_at": {"type": "date"}
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "physics_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": [
                                "lowercase",
                                "stop",
                                "snowball",
                                "physics_synonyms"
                            ]
                        }
                    },
                    "filter": {
                        "physics_synonyms": {
                            "type": "synonym",
                            "synonyms": [
                                "LHC,Large Hadron Collider",
                                "CMS,Compact Muon Solenoid",
                                "ATLAS,A Toroidal LHC ApparatuS",
                                "CERN,European Organization for Nuclear Research"
                            ]
                        }
                    }
                }
            }
        }
        
        # Create papers index
        if not await self.client.indices.exists(index=self.papers_index):
            await self.client.indices.create(
                index=self.papers_index,
                body=papers_mapping
            )
            logger.info(f"Created papers index: {self.papers_index}")
        
        # Authors index mapping
        authors_mapping = {
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "name": {
                        "type": "text",
                        "analyzer": "standard",
                        "fields": {
                            "keyword": {"type": "keyword"},
                            "suggest": {"type": "completion"}
                        }
                    },
                    "email": {"type": "keyword"},
                    "orcid": {"type": "keyword"},
                    "affiliation": {"type": "text"},
                    "paper_count": {"type": "integer"},
                    "citation_count": {"type": "integer"},
                    "h_index": {"type": "integer"},
                    "created_at": {"type": "date"},
                    "updated_at": {"type": "date"}
                }
            }
        }
        
        if not await self.client.indices.exists(index=self.authors_index):
            await self.client.indices.create(
                index=self.authors_index,
                body=authors_mapping
            )
            logger.info(f"Created authors index: {self.authors_index}")
        
        # Institutions index mapping
        institutions_mapping = {
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "name": {
                        "type": "text",
                        "analyzer": "standard",
                        "fields": {
                            "keyword": {"type": "keyword"},
                            "suggest": {"type": "completion"}
                        }
                    },
                    "country": {"type": "keyword"},
                    "type": {"type": "keyword"},
                    "paper_count": {"type": "integer"},
                    "citation_count": {"type": "integer"},
                    "created_at": {"type": "date"},
                    "updated_at": {"type": "date"}
                }
            }
        }
        
        if not await self.client.indices.exists(index=self.institutions_index):
            await self.client.indices.create(
                index=self.institutions_index,
                body=institutions_mapping
            )
            logger.info(f"Created institutions index: {self.institutions_index}")
    
    async def index_paper(self, paper_doc: Dict[str, Any]) -> bool:
        """Index a single paper document."""
        try:
            await self.client.index(
                index=self.papers_index,
                id=paper_doc["id"],
                body=paper_doc
            )
            logger.debug(f"Indexed paper: {paper_doc['id']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to index paper {paper_doc.get('id')}: {e}")
            return False
    
    async def bulk_index_papers(self, papers: List[Dict[str, Any]]) -> Dict[str, int]:
        """Bulk index multiple papers."""
        if not papers:
            return {"indexed": 0, "errors": 0}
        
        # Prepare bulk operations
        bulk_body = []
        for paper in papers:
            bulk_body.extend([
                {"index": {"_index": self.papers_index, "_id": paper["id"]}},
                paper
            ])
        
        try:
            response = await self.client.bulk(body=bulk_body)
            
            # Count successes and errors
            indexed = 0
            errors = 0
            
            for item in response["items"]:
                if "index" in item:
                    if item["index"]["status"] in [200, 201]:
                        indexed += 1
                    else:
                        errors += 1
                        logger.warning(f"Bulk index error: {item['index']}")
            
            logger.info(f"Bulk indexed {indexed} papers, {errors} errors")
            return {"indexed": indexed, "errors": errors}
            
        except Exception as e:
            logger.error(f"Bulk indexing failed: {e}")
            return {"indexed": 0, "errors": len(papers)}
    
    async def search_papers(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        sort: Optional[Dict[str, str]] = None,
        skip: int = 0,
        limit: int = 10,
        highlight: bool = True
    ) -> Dict[str, Any]:
        """Search papers with advanced query capabilities."""
        try:
            # Build Elasticsearch query
            search_body = self._build_papers_query(
                query, filters, sort, skip, limit, highlight
            )
            
            response = await self.client.search(
                index=self.papers_index,
                body=search_body
            )
            
            # Process results
            hits = response["hits"]
            total = hits["total"]["value"] if isinstance(hits["total"], dict) else hits["total"]
            
            results = []
            for hit in hits["hits"]:
                result = hit["_source"]
                result["score"] = hit["_score"]
                
                # Add highlights if available
                if highlight and "highlight" in hit:
                    result["highlights"] = hit["highlight"]
                
                results.append(result)
            
            return {
                "results": results,
                "total": total,
                "query": query,
                "filters": filters or {},
                "skip": skip,
                "limit": limit,
                "max_score": hits.get("max_score", 0)
            }
            
        except Exception as e:
            logger.error(f"Paper search failed: {e}")
            raise SearchError(f"Search failed: {e}")
    
    def _build_papers_query(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
        sort: Optional[Dict[str, str]],
        skip: int,
        limit: int,
        highlight: bool
    ) -> Dict[str, Any]:
        """Build Elasticsearch query for papers."""
        search_body = {
            "from": skip,
            "size": limit,
            "query": {"bool": {"must": [], "filter": []}},
            "sort": []
        }
        
        # Main text query
        if query.strip():
            # Multi-field search with boosting
            multi_match = {
                "multi_match": {
                    "query": query,
                    "fields": [
                        "title^3",
                        "abstract^2",
                        "keywords^2",
                        "authors.name^1.5",
                        "journal"
                    ],
                    "type": "best_fields",
                    "fuzziness": "AUTO"
                }
            }
            search_body["query"]["bool"]["must"].append(multi_match)
        else:
            # Match all if no query
            search_body["query"]["bool"]["must"].append({"match_all": {}})
        
        # Apply filters
        if filters:
            filter_clauses = []
            
            # Source filter
            if filters.get("source"):
                filter_clauses.append({"term": {"source": filters["source"]}})
            
            # Category filter
            if filters.get("categories"):
                categories = filters["categories"] if isinstance(filters["categories"], list) else [filters["categories"]]
                filter_clauses.append({"terms": {"categories": categories}})
            
            # Date range filter
            if filters.get("date_from") or filters.get("date_to"):
                date_range = {"range": {"publication_date": {}}}
                if filters.get("date_from"):
                    date_range["range"]["publication_date"]["gte"] = filters["date_from"]
                if filters.get("date_to"):
                    date_range["range"]["publication_date"]["lte"] = filters["date_to"]
                filter_clauses.append(date_range)
            
            # Citation count range
            if filters.get("min_citations"):
                filter_clauses.append({
                    "range": {"citation_count": {"gte": filters["min_citations"]}}
                })
            
            # Author filter
            if filters.get("author"):
                filter_clauses.append({
                    "nested": {
                        "path": "authors",
                        "query": {
                            "bool": {
                                "should": [
                                    {"match": {"authors.name": filters["author"]}},
                                    {"term": {"authors.email": filters["author"]}}
                                ]
                            }
                        }
                    }
                })
            
            # Institution filter
            if filters.get("institution"):
                filter_clauses.append({
                    "nested": {
                        "path": "institutions",
                        "query": {
                            "match": {"institutions.name": filters["institution"]}
                        }
                    }
                })
            
            search_body["query"]["bool"]["filter"].extend(filter_clauses)
        
        # Apply sorting
        if sort:
            sort_field = sort.get("field", "relevance")
            sort_order = sort.get("order", "desc")
            
            if sort_field == "relevance":
                search_body["sort"].append({"_score": {"order": sort_order}})
            elif sort_field == "date":
                search_body["sort"].append({"publication_date": {"order": sort_order}})
            elif sort_field == "citations":
                search_body["sort"].append({"citation_count": {"order": sort_order}})
            elif sort_field == "title":
                search_body["sort"].append({"title.keyword": {"order": sort_order}})
        else:
            # Default sorting by relevance
            search_body["sort"].append({"_score": {"order": "desc"}})
        
        # Add highlighting
        if highlight:
            search_body["highlight"] = {
                "fields": {
                    "title": {"pre_tags": ["<mark>"], "post_tags": ["</mark>"]},
                    "abstract": {
                        "pre_tags": ["<mark>"], 
                        "post_tags": ["</mark>"],
                        "fragment_size": 150,
                        "number_of_fragments": 3
                    }
                }
            }
        
        return search_body
    
    async def suggest_papers(self, query: str, size: int = 10) -> List[Dict[str, Any]]:
        """Get paper suggestions based on title completion."""
        try:
            suggest_body = {
                "suggest": {
                    "paper_suggest": {
                        "prefix": query,
                        "completion": {
                            "field": "title.suggest",
                            "size": size
                        }
                    }
                }
            }
            
            response = await self.client.search(
                index=self.papers_index,
                body=suggest_body
            )
            
            suggestions = []
            for option in response["suggest"]["paper_suggest"][0]["options"]:
                suggestions.append({
                    "text": option["text"],
                    "score": option["_score"],
                    "source": option["_source"]
                })
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Paper suggestion failed: {e}")
            return []
    
    async def get_aggregations(
        self,
        query: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get aggregated data for faceted search."""
        try:
            # Build base query
            base_query = {"bool": {"must": [], "filter": []}}
            
            if query and query.strip():
                base_query["bool"]["must"].append({
                    "multi_match": {
                        "query": query,
                        "fields": ["title^3", "abstract^2", "keywords^2"]
                    }
                })
            else:
                base_query["bool"]["must"].append({"match_all": {}})
            
            # Apply filters (same as in _build_papers_query)
            if filters:
                # Add filter logic here if needed
                pass
            
            aggs_body = {
                "size": 0,
                "query": base_query,
                "aggs": {
                    "sources": {
                        "terms": {"field": "source", "size": 10}
                    },
                    "categories": {
                        "terms": {"field": "categories", "size": 20}
                    },
                    "years": {
                        "date_histogram": {
                            "field": "publication_date",
                            "calendar_interval": "year",
                            "format": "yyyy"
                        }
                    },
                    "journals": {
                        "terms": {"field": "journal.keyword", "size": 20}
                    },
                    "citation_ranges": {
                        "range": {
                            "field": "citation_count",
                            "ranges": [
                                {"to": 10},
                                {"from": 10, "to": 50},
                                {"from": 50, "to": 100},
                                {"from": 100}
                            ]
                        }
                    }
                }
            }
            
            response = await self.client.search(
                index=self.papers_index,
                body=aggs_body
            )
            
            return response["aggregations"]
            
        except Exception as e:
            logger.error(f"Aggregation query failed: {e}")
            return {}
    
    async def delete_paper(self, paper_id: str) -> bool:
        """Delete a paper from the index."""
        try:
            await self.client.delete(
                index=self.papers_index,
                id=paper_id
            )
            logger.debug(f"Deleted paper from index: {paper_id}")
            return True
            
        except NotFoundError:
            logger.warning(f"Paper not found in index: {paper_id}")
            return True  # Already deleted
        except Exception as e:
            logger.error(f"Failed to delete paper {paper_id}: {e}")
            return False
    
    async def refresh_index(self, index_name: Optional[str] = None):
        """Refresh indices to make changes visible."""
        try:
            if index_name:
                await self.client.indices.refresh(index=index_name)
            else:
                await self.client.indices.refresh(index=f"{self.index_prefix}_*")
            logger.debug("Refreshed search indices")
            
        except Exception as e:
            logger.error(f"Failed to refresh indices: {e}")


# Global search service instance
search_service = SearchService() 