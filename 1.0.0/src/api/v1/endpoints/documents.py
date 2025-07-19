"""
Document download endpoints for CERN Knowledge Explorer.
"""

import asyncio
import aiohttp
import aiofiles
from pathlib import Path
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse, urljoin
import re

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, HttpUrl

from src.core.logging import get_logger
from src.core.config import settings

logger = get_logger(__name__)
router = APIRouter()


class DownloadRequest(BaseModel):
    """Request model for document downloads."""
    url: HttpUrl
    filename: Optional[str] = None
    save_to_disk: bool = True


class DownloadResponse(BaseModel):
    """Response model for download operations."""
    success: bool
    message: str
    filename: Optional[str] = None
    file_size: Optional[int] = None
    download_url: Optional[str] = None


class CERNDocumentInfo(BaseModel):
    """Information about a CERN document."""
    title: str
    authors: List[str] = []
    abstract: Optional[str] = None
    publication_date: Optional[str] = None
    document_type: Optional[str] = None
    url: HttpUrl
    pdf_url: Optional[HttpUrl] = None


# CERN document sources
CERN_SOURCES = {
    "cds": "https://cds.cern.ch",
    "inspire": "https://inspirehep.net",
    "zenodo": "https://zenodo.org",
    "arxiv": "https://arxiv.org"
}


async def is_valid_cern_url(url: str) -> bool:
    """Check if URL is from a valid CERN-related source."""
    parsed = urlparse(url)
    # Check if the domain contains any of our CERN sources
    return any(source_domain in parsed.netloc for source_domain in [
        "cds.cern.ch", "inspirehep.net", "zenodo.org", "arxiv.org"
    ])


async def extract_pdf_url(html_content: str, base_url: str) -> Optional[str]:
    """Extract PDF URL from HTML content."""
    # Common patterns for PDF links
    pdf_patterns = [
        r'href=["\']([^"\']*\.pdf[^"\']*)["\']',
        r'href=["\']([^"\']*pdf[^"\']*)["\']',
        r'["\']([^"\']*\.pdf[^"\']*)["\']',
    ]
    
    for pattern in pdf_patterns:
        matches = re.findall(pattern, html_content, re.IGNORECASE)
        for match in matches:
            if match.startswith('http'):
                return match
            elif match.startswith('/'):
                return urljoin(base_url, match)
            else:
                return urljoin(base_url, '/' + match)
    
    return None


async def download_file(url: str, filename: Optional[str] = None) -> Dict[str, Any]:
    """Download a file from URL."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, allow_redirects=True) as response:
                if response.status != 200:
                    raise HTTPException(status_code=response.status, detail=f"Failed to download file: {response.status}")
                
                # Get filename from URL or response headers
                if not filename:
                    content_disposition = response.headers.get('content-disposition', '')
                    if 'filename=' in content_disposition:
                        match = re.search(r'filename=["\']([^"\']*)["\']', content_disposition)
                        if match:
                            filename = match.group(1)
                        else:
                            filename = url.split('/')[-1] or 'document.pdf'
                    else:
                        filename = url.split('/')[-1] or 'document.pdf'
                
                # Create downloads directory
                downloads_dir = Path("output/downloads")
                downloads_dir.mkdir(parents=True, exist_ok=True)
                
                file_path = downloads_dir / (filename or 'document.pdf')
                
                # Save file
                async with aiofiles.open(file_path, 'wb') as f:
                    content = await response.read()
                    await f.write(content)
                
                return {
                    "success": True,
                    "filename": filename,
                    "file_size": len(content),
                    "file_path": str(file_path),
                    "content_type": response.headers.get('content-type', 'application/octet-stream')
                }
                
    except Exception as e:
        logger.error(f"Error downloading file from {url}: {e}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@router.post("/download", response_model=DownloadResponse)
async def download_document(
    request: DownloadRequest,
    background_tasks: BackgroundTasks
) -> DownloadResponse:
    """Download a document from a CERN-related URL."""
    
    # Validate URL
    if not await is_valid_cern_url(str(request.url)):
        raise HTTPException(status_code=400, detail="URL must be from a valid CERN source")
    
    try:
        # Download the file
        result = await download_file(str(request.url), request.filename)
        
        logger.info(
            "Document downloaded successfully",
            url=str(request.url),
            filename=result["filename"],
            file_size=result["file_size"]
        )
        
        return DownloadResponse(
            success=True,
            message="Document downloaded successfully",
            filename=result["filename"],
            file_size=result["file_size"],
            download_url=f"/api/v1/documents/files/{result['filename']}"
        )
        
    except Exception as e:
        logger.error(f"Failed to download document: {e}")
        return DownloadResponse(
            success=False,
            message=f"Download failed: {str(e)}"
        )


@router.get("/files/{filename}")
async def get_downloaded_file(filename: str):
    """Get a downloaded file."""
    file_path = Path("output/downloads") / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/pdf'
    )


@router.get("/search", response_model=List[CERNDocumentInfo])
async def search_cern_documents(
    query: str = Query(..., description="Search query"),
    source: str = Query("cds", description="CERN source (cds, inspire, zenodo, arxiv)"),
    max_results: int = Query(10, description="Maximum number of results")
) -> List[CERNDocumentInfo]:
    """Search for documents in CERN repositories."""
    
    if source not in CERN_SOURCES:
        raise HTTPException(status_code=400, detail=f"Invalid source. Must be one of: {list(CERN_SOURCES.keys())}")
    
    try:
        # This is a simplified search - in a real implementation, you'd use the actual APIs
        # For now, we'll return mock results
        logger.info(f"Searching for documents with query: {query} in source: {source}")
        
        # Mock results - replace with actual API calls
        mock_results = [
            CERNDocumentInfo(
                title=f"Sample Document: {query}",
                authors=["Author 1", "Author 2"],
                abstract="This is a sample abstract for the search query.",
                publication_date="2024-01-01",
                document_type="paper",
                url=HttpUrl(f"{CERN_SOURCES[source]}/record/12345"),
                pdf_url=HttpUrl(f"{CERN_SOURCES[source]}/record/12345/files/document.pdf")
            )
        ]
        
        return mock_results[:max_results]
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/info/{document_id}")
async def get_document_info(document_id: str) -> CERNDocumentInfo:
    """Get information about a specific document."""
    
    try:
        # Mock document info - replace with actual API calls
        return CERNDocumentInfo(
            title=f"Document {document_id}",
            authors=["Author 1", "Author 2", "Author 3"],
            abstract="This is a detailed abstract for the document.",
            publication_date="2024-01-01",
            document_type="paper",
            url=HttpUrl(f"https://cds.cern.ch/record/{document_id}"),
            pdf_url=HttpUrl(f"https://cds.cern.ch/record/{document_id}/files/document.pdf")
        )
        
    except Exception as e:
        logger.error(f"Failed to get document info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document info: {str(e)}")


@router.get("/sources")
async def get_available_sources() -> Dict[str, str]:
    """Get available CERN document sources."""
    return CERN_SOURCES 