"""
PDF Downloader (v2.0.1)

Small utility to scrape and download recent CERN Courier magazine PDFs.

Usage:
  - Default (downloads up to 5 PDFs):
      python pdf_downloader.py

  - Specify a different limit:
      python pdf_downloader.py --limit 10

Notes:
  - PDFs are saved under `../../data/pdfs` relative to this file by default.
  - Pass `--help` to see CLI options.
"""

import os
import argparse
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

class PDFDownloader:
    """
    A class to download PDF files from the CERN Courier magazine website.
    """
    def __init__(self, base_url, save_path, limit=5):
        self.base_url = base_url
        self.save_path = save_path
        self.limit = limit
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def get_pdf_links(self):
        """
        Scrapes the magazine page to find links to PDF files.
        """
        try:
            response = requests.get(self.base_url)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch {self.base_url}: {e}")
            return []

        soup = BeautifulSoup(response.content, 'html.parser')
        
        links_found = set()
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if href.endswith('.pdf') and 'digitaledition' in href:
                full_url = urljoin(self.base_url, href)
                links_found.add(full_url)
        
        return sorted(list(links_found), reverse=True)[:self.limit]

    def download_pdf(self, url):
        """
        Downloads a single PDF file.
        """
        filename = os.path.join(self.save_path, url.split('/')[-1])
        if os.path.exists(filename):
            print(f"File already exists: {filename}")
            return
            
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Successfully downloaded {filename}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {url}: {e}")

    def download_all(self):
        """
        Downloads all found PDF files.
        """
        pdf_links = self.get_pdf_links()
        if not pdf_links:
            print("No PDF links found.")
            return

        print(f"Found {len(pdf_links)} PDF links. Downloading up to {self.limit} PDFs...")
        for link in pdf_links:
            self.download_pdf(link)

def _positive_int(value):
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("limit must be an integer")
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("limit must be a positive integer")
    return ivalue


def main():
    parser = argparse.ArgumentParser(
        description='Download PDF files from the CERN Courier magazine website.'
    )
    parser.add_argument(
        '--limit', '-l',
        type=_positive_int,
        default=5,
        help='Maximum number of PDFs to download (default: 5)'
    )

    args = parser.parse_args()

    base_url = 'https://cerncourier.com/p/magazine/'
    save_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'pdfs')

    downloader = PDFDownloader(base_url=base_url, save_path=save_dir, limit=args.limit)
    downloader.download_all()


if __name__ == '__main__':
    main()
