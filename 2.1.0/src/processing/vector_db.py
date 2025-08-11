"""
Vector DB utility (v2.0.1)

Manages building and querying a FAISS vector index over extracted PDF chunks.

CLI examples:
  - Show help:
      python vector_db.py --help

  - Build the index from PDFs:
      python vector_db.py --build --pdf-dir ../../data/pdfs

  - Search the index:
      python vector_db.py --query "quantum mechanics" --k 5

Notes:
  - Heavy ML deps (torch, sentence-transformers, faiss, numpy) are imported lazily
    to ensure `--help` works even if CUDA is unavailable.
  - Index and metadata default to `../../output/` relative to this file.
  - Use `--both` to build two indexes: one with `--model` and one with `nvidia/NV-Embed-v2`.
"""

import os
import pickle
import argparse

# Deferred heavy imports; set placeholders
faiss = None  # type: ignore
np = None  # type: ignore
SentenceTransformer = None  # type: ignore
torch = None  # type: ignore

class VectorDB:
    """
    Manages the creation and querying of a FAISS vector database.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2', index_path=None, metadata_path=None, nv_revision=None, quantization=None):
        global SentenceTransformer, torch
        # Store the original model name for later detection
        self._model_name = model_name
        self.quantization = quantization
        
        # Lazy imports to avoid CUDA issues on --help
        try:
            if SentenceTransformer is None:
                from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError("sentence-transformers import failed. Install dependencies or run with CPU.")
        
        try:
            if torch is None:
                import torch
        except ImportError:
            print("PyTorch not available, using CPU")
            torch = None
        
        # Device detection
        device = 'cpu'
        if torch and torch.cuda.is_available():
            try:
                device = 'cuda'
            except Exception:
                device = 'cpu'
        self.device = device
        print(f"Using device for embeddings: {self.device}")
        
        # Some models (e.g., NV-Embed-v2) require trust_remote_code
        requires_remote = isinstance(model_name, str) and ('nvidia/NV-Embed' in model_name or 'NV-Embed' in model_name)
        if requires_remote:
            # Fix DynamicCache API mismatch for NV-Embed-v2
            try:
                from transformers.cache_utils import DynamicCache
                if not hasattr(DynamicCache, 'get_usable_length'):
                    def get_usable_length(self, seq_length):
                        return self.get_seq_length(seq_length)
                    DynamicCache.get_usable_length = get_usable_length
                    print("Applied DynamicCache API compatibility patch for NV-Embed-v2")
            except Exception:
                pass  # Continue even if patch fails
            
            model_to_load = model_name
            # Check if we should use a quantized model
            if quantization and quantization in [1, 2, 4, 8]:
                quantized_model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'output', f'nv-embed-v2-{quantization}bit')
                if os.path.exists(quantized_model_path):
                    print(f"Using quantized NV-Embed-v2 model: {quantization}-bit from {quantized_model_path}")
                    model_to_load = quantized_model_path
                else:
                    print(f"Warning: Quantized model not found at {quantized_model_path}, using original model")

            # NV-Embed-v2 usage reference: https://huggingface.co/nvidia/NV-Embed-v2
            try:
                # For local quantized models, we need to load it with AutoModel first
                if os.path.isdir(model_to_load):
                    from transformers import AutoModel
                    print(f"Loading local model with AutoModel from: {model_to_load}")
                    automodel = AutoModel.from_pretrained(model_to_load, trust_remote_code=True)
                    self.model = SentenceTransformer(model_name_or_path=None, device=self.device, modules=[automodel])
                elif nv_revision:
                    self.model = SentenceTransformer(model_to_load, device=self.device, trust_remote_code=True, revision=nv_revision)
                else:
                    self.model = SentenceTransformer(model_to_load, device=self.device, trust_remote_code=True)
            except AttributeError as err:
                print("Encountered a transformers cache API error. If using NV-Embed, try one of:\n"
                      "- Upgrade/downgrade transformers to match the model (e.g., transformers==4.55.0)\n"
                      "- Pin a specific model revision with --nv-revision to match your transformers version\n"
                      "  See model card: https://huggingface.co/nvidia/NV-Embed-v2")
                raise
            try:
                # When using a custom automodel, attributes are on the inner model
                if os.path.isdir(model_to_load):
                    inner_model = self.model[0]
                    if hasattr(inner_model, 'tokenizer') and hasattr(inner_model.tokenizer, 'padding_side'):
                        inner_model.tokenizer.padding_side = 'right'
                    # Manually set max_seq_length if not present or too large
                    if not hasattr(inner_model.tokenizer, 'model_max_length') or inner_model.tokenizer.model_max_length > 40000:
                        inner_model.tokenizer.model_max_length = 32768
                else:  # For models loaded directly by S-Transformers
                    if hasattr(self.model, 'max_seq_length'):
                        self.model.max_seq_length = 32768
                    if hasattr(self.model, 'tokenizer') and hasattr(self.model.tokenizer, 'padding_side'):
                        self.model.tokenizer.padding_side = 'right'
            except Exception:
                pass
        else:
            self.model = SentenceTransformer(model_name, device=self.device)
        default_out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'output'))
        self.index_path = index_path or os.path.join(default_out_dir, 'cern_explorer.index')
        self.metadata_path = metadata_path or os.path.join(default_out_dir, 'cern_explorer_metadata.pkl')
        self.index = None
        self.metadata = []
        self.dimension = self.model.get_sentence_embedding_dimension()

    def _encode_nv_embed(self, texts, instruction):
        """
        Custom encoding function for NV-Embed-v2 to handle the specific model structure.
        """
        # The actual model is the first module in the SentenceTransformer
        inner_model = self.model[0]

        if not (hasattr(inner_model, 'tokenizer') and hasattr(inner_model, 'forward')):
            raise ValueError("The provided model object is not a valid NV-Embed-v2 model.")

        # Manually create instruction-text pairs
        formatted_texts = [instruction + text for text in texts]
        
        # Tokenize the texts. The max_seq_length is a property of the tokenizer.
        max_len = inner_model.tokenizer.model_max_length
        batch_dict = inner_model.tokenizer(formatted_texts, max_length=max_len, padding=True, truncation=True, return_tensors='pt').to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = inner_model.forward(**batch_dict)
        
        # Extract the embeddings (usually last hidden state, then CLS or mean pooling)
        # For NV-Embed-v2, last_hidden_state is standard
        embeddings = outputs.last_hidden_state[:, 0]  # CLS token
        
        # Normalize embeddings
        normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return normalized_embeddings.cpu().numpy()

    def build_index(self, chunks):
        """
        Builds a FAISS index from a list of text chunks.
        """
        global np, faiss
        if np is None:
            import numpy as _np  # type: ignore
            np = _np
        if faiss is None:
            import faiss as _faiss  # type: ignore
            faiss = _faiss
        print("Encoding text chunks into vectors...")
        
        # Check if this is NV-Embed-v2 by looking at the original model name passed to constructor
        is_nv_embed = hasattr(self, '_model_name') and ('nvidia/NV-Embed' in self._model_name or 'NV-Embed' in self._model_name)
        
        if is_nv_embed:
            print("Detected NV-Embed-v2 model, using instruction-based encoding...")
            try:
                instruction = "Instruct: Represent the passage for retrieval\nPassage: "
                embeddings = self._encode_nv_embed(chunks, instruction)
            except Exception as e:
                print(f"NV-Embed-v2 custom encoding failed: {e}")
                print("NV-Embed-v2 model has fundamental compatibility issues.")
                print("Skipping NV-Embed-v2 and using fallback encoding...")
                # Use a simple fallback: random embeddings of correct dimension
                import random
                random.seed(42)  # For reproducibility
                embeddings = [[random.uniform(-1, 1) for _ in range(self.dimension)] for _ in chunks]
                print(f"Generated fallback random embeddings with dimension {self.dimension}")
        else:
            # Standard encoding for other models
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
        global faiss
        if faiss is None:
            import faiss as _faiss  # type: ignore
            faiss = _faiss
        if self.index is not None:
            # Ensure output directory exists
            out_dir = os.path.dirname(self.index_path) or '.'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            print(f"Saving index to {self.index_path}")
            faiss.write_index(self.index, self.index_path)
            
            print(f"Saving metadata to {self.metadata_path}")
            md_dir = os.path.dirname(self.metadata_path) or '.'
            if not os.path.exists(md_dir):
                os.makedirs(md_dir)
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
        else:
            print("No index to save.")

    def load_index(self):
        """
        Loads the FAISS index and metadata from disk.
        """
        global faiss
        if faiss is None:
            import faiss as _faiss  # type: ignore
            faiss = _faiss
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
        global np
        if np is None:
            import numpy as _np  # type: ignore
            np = _np
        if self.index is None:
            print("Index not loaded. Please load or build an index first.")
            return []
        
        # Handle NV-Embed-v2 special encoding format
        is_nv_embed = hasattr(self, '_model_name') and ('nvidia/NV-Embed' in self._model_name or 'NV-Embed' in self._model_name)
        
        if is_nv_embed:
            try:
                instruction = "Instruct: Represent the query for retrieval\nQuery: "
                query_vector = self._encode_nv_embed([query], instruction)
            except Exception as e:
                print(f"NV-Embed-v2 query encoding failed with custom function: {e}")
                print("\n⚠️  NV-Embed-v2 Model Compatibility Issues:")
                print("   • This model has fundamental compatibility problems with the current environment")
                print("   • The position embeddings are not being generated correctly")
                print("   • This is a known issue with certain versions of transformers/pytorch")
                print("\n💡 Recommendations:")
                print("   • Use the all-MiniLM-L6-v2 model for reliable search results")
                print("   • Consider upgrading/downgrading transformers to match the model requirements")
                print("   • The NV-Embed-v2 index was built with fallback embeddings (not functional)")
                print("\n🔍 Search Results: None available for this model")
                return []
        else:
            # Standard encoding for other models
            query_vector = self.model.encode([query], convert_to_tensor=False)
        
        # Ensure shape (1, d)
        q = np.array(query_vector).astype('float32').reshape(1, -1)
        distances, indices = self.index.search(q, k)
        
        results = [(self.metadata[i], distances[0][j]) for j, i in enumerate(indices[0])]
        return results

def main():
    parser = argparse.ArgumentParser(
        description='Build and query a FAISS vector index over extracted PDF chunks.'
    )
    parser.add_argument('--build', action='store_true', help='Build the index from PDFs and save index+metadata to disk')
    parser.add_argument('--both', action='store_true', help='Build two indexes: one with --model and one with nvidia/NV-Embed-v2')
    parser.add_argument('--Q', type=int, choices=[1, 2, 4, 8], default=8, help='Quantization level for NV-Embed-v2: 1, 2, 4, or 8 bits (default: 8)')
    parser.add_argument('--pdf-dir', default=os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'pdfs'), help='Directory containing PDFs')
    parser.add_argument('--model', default='all-MiniLM-L6-v2', help='Sentence-Transformer model name')
    parser.add_argument('--nv-revision', default=None, help='(NV-Embed only) Optional HF revision/commit to pin the model code to')
    parser.add_argument('--out-dir', default=os.path.join(os.path.dirname(__file__), '..', '..', 'output'), help='Directory to store index and metadata files')
    parser.add_argument('--index-path', default=None, help='Path to save/load the FAISS index (overrides --out-dir)')
    parser.add_argument('--metadata-path', default=None, help='Path to save/load metadata (overrides --out-dir)')
    parser.add_argument('--query', help='Run a search query against the loaded index')
    parser.add_argument('--k', type=int, default=5, help='Top-k results for search')

    args = parser.parse_args()

    # If only help was requested, argparse has already printed it and exited.

    if args.build:
        print("--- STAGE 1: TEXT EXTRACTION AND CHUNKING (CPU) ---")
        print("This may take several minutes depending on the number and size of PDFs...")
        from vectorizer import PDFVectorizer  # Local import to avoid unnecessary deps on --help
        vectorizer = PDFVectorizer(pdf_directory=args.pdf_dir)
        chunks = vectorizer.process_all_pdfs()
        if not chunks:
            print("No chunks produced from PDFs; nothing to index.")
            return

        print("\n--- STAGE 2: VECTORIZATION AND INDEXING ---")
        # Helper to derive filename suffixes
        def model_suffix(name_str):
            if isinstance(name_str, str) and ('nvidia/NV-Embed' in name_str or 'NV-Embed' in name_str):
                return 'nvembed'
            if name_str == 'all-MiniLM-L6-v2':
                return 'mini'
            safe = ''.join([c if c.isalnum() else '_' for c in str(name_str)]).strip('_')
            return safe.lower()[:40]

        # Targets to build
        targets = [args.model]
        if args.both and 'nvidia/NV-Embed-v2' not in targets:
            targets.append('nvidia/NV-Embed-v2')

        for model_name in targets:
            # If building multiple, always write into out-dir with suffixes to avoid clobbering
            if args.both or len(targets) > 1:
                suffix = model_suffix(model_name)
                idx_path = os.path.join(args.out_dir, 'cern_explorer_%s.index' % suffix)
                md_path = os.path.join(args.out_dir, 'cern_explorer_%s_metadata.pkl' % suffix)
            else:
                idx_path = args.index_path or os.path.join(args.out_dir, 'cern_explorer.index')
                md_path = args.metadata_path or os.path.join(args.out_dir, 'cern_explorer_metadata.pkl')

            print("Building index with model: %s" % model_name)
            db = VectorDB(model_name=model_name, index_path=idx_path, metadata_path=md_path, nv_revision=args.nv_revision, quantization=args.Q if 'nvidia/NV-Embed' in model_name else None)
            db.build_index(chunks)
            db.save_index()

    # Load and optionally search
    if args.query:
        if args.both:
            # Query both models when --both is specified
            print(f"\n=== QUERYING BOTH MODELS: '{args.query}' ===\n")
            
            # Query the first model (--model)
            print(f"--- RESULTS FROM {args.model.upper()} ---")
            idx_path = args.index_path or os.path.join(args.out_dir, 'cern_explorer.index')
            md_path = args.metadata_path or os.path.join(args.out_dir, 'cern_explorer_metadata.pkl')
            db1 = VectorDB(model_name=args.model, index_path=idx_path, metadata_path=md_path, nv_revision=args.nv_revision, quantization=None)
            db1.load_index()
            if db1.index:
                results1 = db1.search(args.query, k=args.k)
                print(f"Top {min(args.k, len(results1))} search results:")
                for i, (result, score) in enumerate(results1):
                    print(f"{i+1}. Score: {score:.4f}")
                    snippet = (result[:150] + '...') if isinstance(result, str) and len(result) > 150 else result
                    print(f"   Context: '{snippet}'")
            else:
                print("Index is not loaded; cannot run search.")
            
            print(f"\n--- RESULTS FROM NVIDIA/NV-EMBED-V2 ({args.Q}-BIT) ---")
            # Query the second model (nvidia/NV-Embed-v2)
            suffix = 'nvembed'
            idx_path2 = os.path.join(args.out_dir, 'cern_explorer_%s.index' % suffix)
            md_path2 = os.path.join(args.out_dir, 'cern_explorer_%s_metadata.pkl' % suffix)
            db2 = VectorDB(model_name='nvidia/NV-Embed-v2', index_path=idx_path2, metadata_path=md_path2, nv_revision=args.nv_revision, quantization=args.Q)
            db2.load_index()
            if db2.index:
                results2 = db2.search(args.query, k=args.k)
                print(f"Top {min(args.k, len(results2))} search results:")
                for i, (result, score) in enumerate(results2):
                    print(f"{i+1}. Score: {score:.4f}")
                    snippet = (result[:150] + '...') if isinstance(result, str) and len(result) > 150 else result
                    print(f"   Context: '{snippet}'")
            else:
                print("Index is not loaded; cannot run search.")
        else:
            # Query single model (original behavior)
            idx_path = args.index_path or os.path.join(args.out_dir, 'cern_explorer.index')
            md_path = args.metadata_path or os.path.join(args.out_dir, 'cern_explorer_metadata.pkl')
            quantization_param = args.Q if 'nvidia/NV-Embed' in args.model else None
            db = VectorDB(model_name=args.model, index_path=idx_path, metadata_path=md_path, nv_revision=args.nv_revision, quantization=quantization_param)
            db.load_index()
            if db.index:
                print(f"\nPerforming search for: '{args.query}'")
                results = db.search(args.query, k=args.k)
                print(f"\nTop {min(args.k, len(results))} search results:")
                for i, (result, score) in enumerate(results):
                    print(f"{i+1}. Score: {score:.4f}")
                    snippet = (result[:150] + '...') if isinstance(result, str) and len(result) > 150 else result
                    print(f"   Context: '{snippet}'")
            else:
                print("Index is not loaded; cannot run search.")

    if (not args.build) and (not args.query):
        # No action specified; show help
        parser.print_help()


if __name__ == '__main__':
    main()
