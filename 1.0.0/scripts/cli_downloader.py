#!/usr/bin/env python3
"""
Command-line interface for downloading PDF documents from CERN sites.
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.download_pdf import download_pdf, search_documents


async def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Download PDF documents from CERN sites",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download a specific PDF
  python cli_downloader.py download "https://arxiv.org/pdf/1207.7214.pdf" --filename higgs.pdf
  
  # Search for documents
  python cli_downloader.py search "Higgs boson" --source cds --max-results 5
  
  # Download from search results
  python cli_downloader.py search "Higgs boson" --download-first
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download a specific PDF')
    download_parser.add_argument('url', help='URL of the PDF to download')
    download_parser.add_argument('--filename', '-f', help='Custom filename for the downloaded file')
    download_parser.add_argument('--api-url', default='http://localhost:8000', help='API base URL')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search for documents')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--source', '-s', default='cds', 
                              choices=['cds', 'inspire', 'zenodo', 'arxiv'],
                              help='CERN source to search')
    search_parser.add_argument('--max-results', '-m', type=int, default=10,
                              help='Maximum number of results')
    search_parser.add_argument('--download-first', '-d', action='store_true',
                              help='Download the first search result')
    search_parser.add_argument('--api-url', default='http://localhost:8000', help='API base URL')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'download':
            print(f"Downloading PDF from: {args.url}")
            result = await download_pdf(args.api_url, args.url, args.filename)
            
            if result.get("success"):
                print(f"✅ Successfully downloaded: {result['filename']}")
                print(f"   File size: {result['file_size']} bytes")
                print(f"   Download URL: {args.api_url}{result['download_url']}")
            else:
                print(f"❌ Download failed: {result.get('message', 'Unknown error')}")
                sys.exit(1)
        
        elif args.command == 'search':
            print(f"Searching for: '{args.query}' in {args.source}")
            results = await search_documents(args.api_url, args.query, args.source, args.max_results)
            
            print(f"\nFound {len(results)} documents:")
            for i, doc in enumerate(results, 1):
                print(f"\n{i}. {doc['title']}")
                print(f"   Authors: {', '.join(doc['authors'])}")
                if doc.get('abstract'):
                    print(f"   Abstract: {doc['abstract'][:100]}...")
                print(f"   URL: {doc['url']}")
                if doc.get('pdf_url'):
                    print(f"   PDF: {doc['pdf_url']}")
            
            if args.download_first and results:
                print(f"\nDownloading first result...")
                first_doc = results[0]
                if first_doc.get('pdf_url'):
                    download_result = await download_pdf(
                        args.api_url,
                        str(first_doc['pdf_url']),
                        f"search_result_{first_doc['title'].replace(' ', '_')}.pdf"
                    )
                    
                    if download_result.get("success"):
                        print(f"✅ Successfully downloaded: {download_result['filename']}")
                    else:
                        print(f"❌ Failed to download: {download_result.get('message')}")
                else:
                    print("❌ First result has no PDF URL")
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 