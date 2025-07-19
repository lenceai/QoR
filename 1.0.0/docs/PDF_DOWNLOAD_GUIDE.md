# PDF Download Guide for CERN Knowledge Explorer

This guide explains how to download PDF documents from CERN-related sites using the CERN Knowledge Explorer API.

## Overview

The CERN Knowledge Explorer provides a comprehensive API for downloading PDF documents from various CERN-related sources:

- **CDS (CERN Document Server)**: https://cds.cern.ch
- **INSPIRE**: https://inspirehep.net
- **Zenodo**: https://zenodo.org
- **arXiv**: https://arxiv.org

## Quick Start

### 1. Start the API Server

```bash
cd /path/to/your/project
python src/main.py
```

The server will start on `http://localhost:8000`

### 2. Using the Command Line Interface

The easiest way to download PDFs is using the provided CLI tool:

```bash
# Download a specific PDF
python scripts/cli_downloader.py download "https://arxiv.org/pdf/1207.7214.pdf" --filename higgs.pdf

# Search for documents
python scripts/cli_downloader.py search "Higgs boson" --source cds --max-results 5

# Search and download the first result
python scripts/cli_downloader.py search "Higgs boson" --download-first
```

### 3. Using the API Directly

#### Download a PDF

```bash
curl -X POST "http://localhost:8000/api/v1/documents/download" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://arxiv.org/pdf/1207.7214.pdf",
    "filename": "my_document.pdf"
  }'
```

#### Search for Documents

```bash
curl -X GET "http://localhost:8000/api/v1/documents/search?query=Higgs%20boson&source=cds&max_results=5"
```

#### Get Available Sources

```bash
curl -X GET "http://localhost:8000/api/v1/documents/sources"
```

## API Endpoints

### POST /api/v1/documents/download

Download a PDF document from a CERN-related URL.

**Request Body:**
```json
{
  "url": "https://arxiv.org/pdf/1207.7214.pdf",
  "filename": "optional_custom_name.pdf",
  "save_to_disk": true
}
```

**Response:**
```json
{
  "success": true,
  "message": "Document downloaded successfully",
  "filename": "my_document.pdf",
  "file_size": 737631,
  "download_url": "/api/v1/documents/files/my_document.pdf"
}
```

### GET /api/v1/documents/files/{filename}

Download a previously downloaded file.

### GET /api/v1/documents/search

Search for documents in CERN repositories.

**Query Parameters:**
- `query` (required): Search query
- `source` (optional): Source to search (cds, inspire, zenodo, arxiv)
- `max_results` (optional): Maximum number of results (default: 10)

### GET /api/v1/documents/sources

Get available CERN document sources.

## Python Script Examples

### Basic Download Script

```python
import asyncio
import aiohttp

async def download_pdf():
    api_url = "http://localhost:8000"
    pdf_url = "https://arxiv.org/pdf/1207.7214.pdf"
    
    payload = {
        "url": pdf_url,
        "filename": "higgs_discovery.pdf"
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{api_url}/api/v1/documents/download",
            json=payload
        ) as response:
            result = await response.json()
            print(f"Download result: {result}")

asyncio.run(download_pdf())
```

### Search and Download Script

```python
import asyncio
import aiohttp

async def search_and_download():
    api_url = "http://localhost:8000"
    
    # Search for documents
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{api_url}/api/v1/documents/search",
            params={"query": "Higgs boson", "source": "cds", "max_results": 3}
        ) as response:
            results = await response.json()
    
    # Download first result if it has a PDF
    if results and results[0].get('pdf_url'):
        first_doc = results[0]
        payload = {
            "url": str(first_doc['pdf_url']),
            "filename": f"{first_doc['title'].replace(' ', '_')}.pdf"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{api_url}/api/v1/documents/download",
                json=payload
            ) as response:
                result = await response.json()
                print(f"Download result: {result}")

asyncio.run(search_and_download())
```

## File Storage

Downloaded files are stored in the `output/downloads/` directory. The directory structure is:

```
output/
├── downloads/
│   ├── document1.pdf
│   ├── document2.pdf
│   └── ...
├── logs/
└── ...
```

## Error Handling

The API returns appropriate HTTP status codes and error messages:

- `400 Bad Request`: Invalid URL or parameters
- `404 Not Found`: File not found
- `500 Internal Server Error`: Server error during download

Example error response:
```json
{
  "error": {
    "type": "HTTPException",
    "message": "URL must be from a valid CERN source",
    "status_code": 400
  }
}
```

## Supported URL Formats

The API supports various URL formats from CERN sources:

- Direct PDF URLs: `https://arxiv.org/pdf/1207.7214.pdf`
- CDS record URLs: `https://cds.cern.ch/record/123456`
- INSPIRE URLs: `https://inspirehep.net/literature/123456`
- Zenodo URLs: `https://zenodo.org/record/123456`

## Rate Limiting

The API includes rate limiting to prevent abuse. By default:
- 100 requests per minute
- 20 requests per burst

## Security Considerations

- Only URLs from approved CERN sources are accepted
- Downloaded files are stored locally and can be accessed via the API
- Consider implementing authentication for production use

## Troubleshooting

### Common Issues

1. **"URL must be from a valid CERN source"**
   - Ensure the URL is from one of the supported sources
   - Check that the domain matches exactly

2. **"Download failed"**
   - Check if the URL is accessible
   - Verify the file exists at the specified URL
   - Check server logs for detailed error information

3. **File not found when trying to download**
   - Ensure the file was successfully downloaded first
   - Check the `output/downloads/` directory

### Debug Mode

Enable debug logging by setting the environment variable:
```bash
export DEBUG=true
```

## Advanced Usage

### Batch Downloads

For downloading multiple files, consider using the background task functionality:

```python
import asyncio
from scripts.download_pdf import download_pdf

async def batch_download(urls):
    tasks = []
    for url in urls:
        task = download_pdf("http://localhost:8000", url)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results

urls = [
    "https://arxiv.org/pdf/1207.7214.pdf",
    "https://arxiv.org/pdf/1207.7215.pdf",
    "https://arxiv.org/pdf/1207.7216.pdf"
]

results = asyncio.run(batch_download(urls))
```

### Custom Filename Patterns

You can implement custom filename patterns based on document metadata:

```python
def generate_filename(doc_info):
    """Generate filename from document information."""
    title = doc_info['title'].replace(' ', '_')[:50]
    date = doc_info.get('publication_date', 'unknown')
    return f"{date}_{title}.pdf"
```

## Contributing

To extend the PDF download functionality:

1. Add new CERN sources to the `CERN_SOURCES` dictionary
2. Implement source-specific search APIs
3. Add new validation rules for URLs
4. Extend the document information model

## Support

For issues and questions:
1. Check the server logs in `output/logs/`
2. Verify the API is running and accessible
3. Test with a simple URL first
4. Check the health endpoint: `GET /health` 