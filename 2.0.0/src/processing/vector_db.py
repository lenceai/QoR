import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import pickle
import torch

class VectorDB:
    """
    Manages the creation and querying of a FAISS vector database.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2', index_path='cern_explorer.index', metadata_path='cern_explorer_metadata.pkl'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device for embeddings: {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.metadata = []
        self.dimension = self.model.get_sentence_embedding_dimension()

    def build_index(self, chunks):
        """
        Builds a FAISS index from a list of text chunks.
        """
        print("Encoding text chunks into vectors...")
        embeddings = self.model.encode(chunks, convert_to_tensor=False, show_progress_bar=True)
        
        print(f"Creating FAISS index with dimension {self.dimension}...")
        # Create CPU index (FAISS-CPU doesn't support GPU operations)
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        
        self.metadata = chunks
        
        print(f"Index built successfully with {self.index.ntotal} vectors.")

    def save_index(self):
        """
        Saves the FAISS index and metadata to disk.
        """
        if self.index is not None:
            print(f"Saving index to {self.index_path}")
            faiss.write_index(self.index, self.index_path)
            
            print(f"Saving metadata to {self.metadata_path}")
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
        else:
            print("No index to save.")

    def load_index(self):
        """
        Loads the FAISS index and metadata from disk.
        """
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            print(f"Loading index from {self.index_path}")
            self.index = faiss.read_index(self.index_path)
            
            print(f"Loading metadata from {self.metadata_path}")
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            
            print(f"Index and metadata loaded successfully. Index contains {self.index.ntotal} vectors.")
        else:
            print("Index files not found. Please build the index first.")

    def search(self, query, k=5):
        """
        Performs a similarity search on the index.
        """
        if self.index is None:
            print("Index not loaded. Please load or build an index first.")
            return []
            
        query_vector = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_vector).astype('float32'), k)
        
        results = [(self.metadata[i], distances[0][j]) for j, i in enumerate(indices[0])]
        return results

if __name__ == '__main__':
    from vectorizer import PDFVectorizer

    PDF_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'pdfs')
    
    print("--- STAGE 1: TEXT EXTRACTION AND CHUNKING (CPU) ---")
    print("This may take several minutes depending on the number and size of PDFs...")
    # Step 1: Process PDFs to get text chunks
    vectorizer = PDFVectorizer(pdf_directory=PDF_DIR)
    chunks = vectorizer.process_all_pdfs()

    if chunks:
        print(f"\n--- STAGE 2: VECTORIZATION AND INDEXING ---")
        # Step 2: Build and save the vector database
        db = VectorDB()
        db.build_index(chunks)
        db.save_index()
        
        # Step 3: Load the database and perform a search
        db_loaded = VectorDB()
        db_loaded.load_index()
        
        if db_loaded.index:
            query = "quantum mechanics"
            print(f"\nPerforming search for: '{query}'")
            results = db_loaded.search(query)
            
            print("\nTop 5 search results:")
            for i, (result, score) in enumerate(results):
                print(f"{i+1}. Score: {score:.4f}")
                print(f"   Context: '{result[:150]}...'")
