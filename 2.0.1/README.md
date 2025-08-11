# CERN Knowledge Explorer v2.0.1

Small utilities for ingesting CERN Courier PDFs and building a FAISS vector database for semantic search.

## What’s new in 2.0.1
- `pdf_downloader.py` now has a CLI with `--limit` and `--help`.
- `vector_db.py` now has a CLI with `--build`, `--query`, `--k`, `--out-dir`, and lazy imports so `--help` works without CUDA.
- All index/metadata outputs default to `output/`.

## Requirements
Install with pinned versions below. Choose the GPU stack only if you have an NVIDIA GPU and a matching CUDA runtime. Otherwise, use the CPU-only stack.

### Base (shared)
- `requests>=2.31.0,<3.0.0`
- `beautifulsoup4>=4.12.0,<5.0.0`
- `PyPDF2>=3.0.0,<4.0.0`
- `langchain>=0.2.0,<0.3.0`
- `numpy>=1.24.0,<2.0.0`

### ML/NLP stack (pinned)
- `torch==2.7.0`
- `transformers==4.55.0`
- `accelerate==1.10.0`
- `peft==0.17.0`
- `trl==0.20.0`
- `sentence-transformers==5.1.0`
- `faiss-cpu==1.11.0.post1` (use `faiss-gpu` only if you know what you’re doing)

### Install (CPU-only)
```bash
pip install -r requirements.txt
```

### Install (GPU, NVIDIA, Linux/macOS/WSL2)
Pick ONE CUDA wheels index (uncomment in `requirements.txt`):
- CUDA 12.6: `--extra-index-url https://download.pytorch.org/whl/cu126`
- CUDA 11.8: `--extra-index-url https://download.pytorch.org/whl/cu118`

Then:
```bash
pip install -r requirements.txt
```

If you run into CUDA/NCCL errors, consider CPU-only: install `faiss-cpu` and CPU wheels for PyTorch.

## Quick start

### 1) Download latest PDFs
```bash
python src/ingestion/pdf_downloader.py --limit 10
```
PDFs are saved under `data/pdfs/`.

### 2) Build the vector index
```bash
python src/processing/vector_db.py --build --pdf-dir ./data/pdfs --out-dir ./output
```
This writes `cern_explorer.index` and `cern_explorer_metadata.pkl` to `output/`.

### 3) Query the index
```bash
python src/processing/vector_db.py --query "quantum mechanics" --k 5 --out-dir ./output
```

### 4) Optional: LoRA fine-tune the 24B model (QLoRA 4-bit)
Prereq: ensure `./output/cern_explorer_metadata.pkl` exists (built in step 2).

Run a memory-safe default training (24 GB GPU works):
```bash
python src/training/mistral24b_unsloth_4bit_finetune.py --train
```

Defaults in v2.0.1 when using `--train`:
- max sequence length: 1024 (`--max-seq-len`)
- LoRA target modules: `q_proj,v_proj` (`--lora-target-modules`)
- LoRA rank/alpha: 8 / 16 (`--lora-r`, `--lora-alpha`)
- Sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` automatically
- Saves adapter to `./output/adapters/mistral24b_lora`

Tips:
- Reduce OOM risk by lowering `--max-seq-len` further, or switch to a smaller base model.
- 2-bit GGUF files (e.g., `...IQ2_XXS.gguf`) are inference-only and not trainable via this script.

## Notes
- To change the number of PDFs, pass `--limit` to `pdf_downloader.py`.
- To change the output location for the index/metadata, pass `--out-dir` (default is `output/`).
- If GPU is not available or CUDA is misconfigured, the system will fallback to CPU when possible.
 - QLoRA training uses the 4-bit Unsloth HF checkpoint: `unsloth/Mistral-Small-3.2-24B-Instruct-2506-unsloth-bnb-4bit`.
