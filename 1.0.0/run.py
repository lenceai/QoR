#!/usr/bin/env python3
"""
Main run script for CERN Knowledge Explorer.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def run_server():
    """Run the FastAPI server."""
    import uvicorn
    from src.core.config import settings
    from src.core.logging import setup_logging
    
    setup_logging()
    
    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else settings.WORKERS,
        log_config=None  # We handle logging ourselves
    )


def init_database():
    """Initialize the database."""
    import asyncio
    from src.scripts.init_db import main
    
    asyncio.run(main())


def run_tests():
    """Run the test suite."""
    import pytest
    
    # Run tests from the tests directory
    test_dir = Path(__file__).parent / "tests"
    if test_dir.exists():
        sys.exit(pytest.main([str(test_dir), "-v"]))
    else:
        print("No tests directory found. Creating basic test structure...")
        test_dir.mkdir(exist_ok=True)
        print(f"Created {test_dir}. Please add your tests there.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="CERN Knowledge Explorer")
    parser.add_argument(
        "command",
        choices=["server", "init-db", "test"],
        help="Command to run"
    )
    parser.add_argument(
        "--env",
        default=".env",
        help="Environment file to load (default: .env)"
    )
    
    args = parser.parse_args()
    
    # Load environment file if it exists
    env_file = Path(args.env)
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)
        print(f"Loaded environment from {env_file}")
    
    # Run the specified command
    if args.command == "server":
        print("Starting CERN Knowledge Explorer server...")
        run_server()
    elif args.command == "init-db":
        print("Initializing database...")
        init_database()
    elif args.command == "test":
        print("Running tests...")
        run_tests()


if __name__ == "__main__":
    main() 