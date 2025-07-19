#!/usr/bin/env python3
"""
Simple script to download PDF documents from CERN sites using the API.
"""

import asyncio
import aiohttp
import json
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import settings


async def download_pdf(api_url: str, pdf_url: str, filename: Optional[str] = None) -> dict:
    """Download a PDF using the API."""
    
    # Prepare the request
    payload = {
        "url": pdf_url,
        "filename": filename,
        "save_to_disk": True
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{api_url}/api/v1/documents/download",
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as response:
            result = await response.json()
            return result


async def search_documents(api_url: str, query: str, source: str = "cds", max_results: int = 5) -> list:
    """Search for documents using the API."""
    
    params = {
        "query": query,
        "source": source,
        "max_results": max_results
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{api_url}/api/v1/documents/search",
            params=params
        ) as response:
            results = await response.json()
            return results


async def main():
    """Main function to demonstrate PDF downloading."""
    
    api_url = "http://localhost:8000"
    
    print("=== CERN Knowledge Explorer - PDF Downloader ===\n")
    
    # Example 1: Download a specific PDF
    print("1. Downloading a specific PDF...")
    pdf_url = "https://arxiv.org/pdf/1207.7214.pdf"
    result = await download_pdf(api_url, pdf_url, "higgs_discovery.pdf")
    
    if result.get("success"):
        print(f"✅ Successfully downloaded: {result['filename']}")
        print(f"   File size: {result['file_size']} bytes")
        print(f"   Download URL: {api_url}{result['download_url']}")
    else:
        print(f"❌ Download failed: {result.get('message', 'Unknown error')}")
    
    print()
    
    # Example 2: Search for documents
    print("2. Searching for documents...")
    search_results = await search_documents(api_url, "Higgs boson", "cds", 3)
    
    print(f"Found {len(search_results)} documents:")
    for i, doc in enumerate(search_results, 1):
        print(f"   {i}. {doc['title']}")
        print(f"      Authors: {', '.join(doc['authors'])}")
        print(f"      URL: {doc['url']}")
        if doc.get('pdf_url'):
            print(f"      PDF: {doc['pdf_url']}")
        print()
    
    # Example 3: Download from search results
    if search_results and search_results[0].get('pdf_url'):
        print("3. Downloading first search result...")
        first_doc = search_results[0]
        download_result = await download_pdf(
            api_url, 
            str(first_doc['pdf_url']), 
            f"search_result_{first_doc['title'].replace(' ', '_')}.pdf"
        )
        
        if download_result.get("success"):
            print(f"✅ Successfully downloaded search result: {download_result['filename']}")
        else:
            print(f"❌ Failed to download search result: {download_result.get('message')}")


if __name__ == "__main__":
    asyncio.run(main()) 