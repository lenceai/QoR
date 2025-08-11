import os
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter

class PDFVectorizer:
    """
    Handles text extraction and chunking from PDF files.
    """
    def __init__(self, pdf_directory, chunk_size=1024, chunk_overlap=128):
        self.pdf_directory = pdf_directory
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

    def extract_text_from_pdf(self, pdf_path):
        """
        Extracts raw text from a single PDF file.
        """
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
                return text
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
            return ""

    def get_text_chunks(self, text):
        """
        Splits a raw text into manageable chunks.
        """
        return self.text_splitter.split_text(text)

    def process_all_pdfs(self):
        """
        Processes all PDFs in the specified directory, returning a list of text chunks.
        """
        all_chunks = []
        pdf_files = [f for f in os.listdir(self.pdf_directory) if f.endswith('.pdf')]
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.pdf_directory, pdf_file)
            print(f"Processing {pdf_file}...")
            raw_text = self.extract_text_from_pdf(pdf_path)
            if raw_text:
                chunks = self.get_text_chunks(raw_text)
                all_chunks.extend(chunks)
                print(f"  Extracted {len(chunks)} chunks.")
        
        return all_chunks

if __name__ == '__main__':
    PDF_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'pdfs')
    
    vectorizer = PDFVectorizer(pdf_directory=PDF_DIR)
    chunks = vectorizer.process_all_pdfs()
    
    print(f"\nTotal chunks processed: {len(chunks)}")
    if chunks:
        print("Sample chunk:")
        print(f"'{chunks[0][:200]}...'")
