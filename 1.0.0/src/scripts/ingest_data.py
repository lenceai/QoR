#!/usr/bin/env python3
"""
Data ingestion script for CERN Knowledge Explorer.
"""

import asyncio
import sys
import argparse
import yaml
from pathlib import Path
from typing import List, Optional

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import settings
from src.core.logging import setup_logging, get_logger
from src.services.data_sources import DataIngestionService
from src.services.data_persistence import DataPersistenceService, process_ingestion_batch

logger = get_logger(__name__)


async def load_config() -> dict:
    """Load data sources configuration."""
    config_path = Path("data/config/data_sources.yaml")
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)


async def ingest_sample_data():
    """Ingest sample data for testing."""
    logger.info("Starting sample data ingestion...")
    
    config = await load_config()
    ingestion_service = DataIngestionService(config)
    persistence_service = DataPersistenceService()
    
    # Sample queries for different physics areas
    queries = [
        "higgs boson",
        "dark matter",
        "supersymmetry",
        "cms experiment",
        "atlas experiment"
    ]
    
    total_results = {"saved": 0, "duplicates": 0, "errors": 0, "total": 0}
    
    for query in queries:
        logger.info(f"Ingesting data for query: {query}")
        
        try:
            result = await process_ingestion_batch(
                ingestion_service,
                persistence_service,
                query,
                sources=["arxiv"],  # Start with ArXiv only for testing
                max_results=20
            )
            
            # Aggregate results
            for key in total_results:
                total_results[key] += result.get(key, 0)
            
            logger.info(f"Query '{query}' results: {result}")
            
        except Exception as e:
            logger.error(f"Failed to ingest data for query '{query}': {e}")
            continue
    
    logger.info(f"Sample data ingestion completed. Total results: {total_results}")
    return total_results


async def ingest_full_data(
    query: str,
    sources: Optional[List[str]] = None,
    max_results: int = 100
):
    """Ingest full dataset with specified parameters."""
    logger.info(f"Starting full data ingestion: query='{query}', sources={sources}, max_results={max_results}")
    
    config = await load_config()
    ingestion_service = DataIngestionService(config)
    persistence_service = DataPersistenceService()
    
    try:
        result = await process_ingestion_batch(
            ingestion_service,
            persistence_service,
            query,
            sources,
            max_results
        )
        
        logger.info(f"Full data ingestion completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Full data ingestion failed: {e}")
        raise


async def get_ingestion_status():
    """Get current ingestion status and statistics."""
    logger.info("Getting ingestion status...")
    
    persistence_service = DataPersistenceService()
    
    try:
        stats = await persistence_service.get_ingestion_stats()
        
        print("\n=== CERN Knowledge Explorer Ingestion Status ===")
        print(f"Total Papers: {stats['total_papers']}")
        print(f"Total Authors: {stats['total_authors']}")
        print(f"Total Institutions: {stats['total_institutions']}")
        print(f"Recent Papers (30 days): {stats['recent_papers']}")
        print("\nPapers by Source:")
        for source, count in stats['papers_by_source'].items():
            print(f"  {source}: {count}")
        print(f"\nLast Updated: {stats['last_updated']}")
        print("=" * 48)
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get ingestion status: {e}")
        raise


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="CERN Knowledge Explorer Data Ingestion")
    parser.add_argument(
        "command",
        choices=["sample", "full", "status"],
        help="Ingestion command to run"
    )
    parser.add_argument(
        "--query",
        help="Search query for full ingestion"
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["arxiv", "inspire_hep", "cern_document_server"],
        help="Data sources to use"
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=100,
        help="Maximum results per source (default: 100)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("CERN Knowledge Explorer Data Ingestion Started")
    logger.info(f"Command: {args.command}")
    
    try:
        if args.command == "sample":
            await ingest_sample_data()
        elif args.command == "full":
            if not args.query:
                logger.error("Query is required for full ingestion")
                sys.exit(1)
            await ingest_full_data(args.query, args.sources, args.max_results)
        elif args.command == "status":
            await get_ingestion_status()
    
    except KeyboardInterrupt:
        logger.info("Ingestion interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        sys.exit(1)
    
    logger.info("Data ingestion completed successfully")


if __name__ == "__main__":
    asyncio.run(main()) 