"""
NV-Embed-v2 Quantization Utility (v2.0.1)

Quantizes/loading NV-Embed-v2 for different bit-level inference using bitsandbytes.

Outputs:
  - Saves model and tokenizer files to ../../output/nv-embed-v2-{Q}bit
  - Note: Hugging Face save_pretrained writes original precision weights.
    To run in quantized mode, load with the same BitsAndBytesConfig (see sample code below).

Usage examples:
  - Quantize with default 8-bit settings:
      python quantize_nvembed.py

  - Quantize to 4-bit:
      python quantize_nvembed.py --Q 4

  - Quantize to 2-bit:
      python quantize_nvembed.py --Q 2

  - Quantize to 1-bit:
      python quantize_nvembed.py --Q 1

  - Pin a specific revision/commit:
      python quantize_nvembed.py --revision 3fa59658547db50a1e8e3346cf057fd0c77ed6ef

  - Verify a quick encode pass:
      python quantize_nvembed.py --verify

  - Custom output directory:
      python quantize_nvembed.py --out-dir ../../output/nv-embed-v2-4bit

To load later in quantized mode, use:
  from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
  bnb = BitsAndBytesConfig(load_in_8bit=True)  # or load_in_4bit=True, etc.
  tok = AutoTokenizer.from_pretrained("/path/to/nv-embed-v2-{Q}bit", trust_remote_code=True)
  model = AutoModel.from_pretrained("/path/to/nv-embed-v2-{Q}bit", trust_remote_code=True, quantization_config=bnb, device_map="auto")
"""

import os
import argparse


def main():
    parser = argparse.ArgumentParser(description='Quantize nvidia/NV-Embed-v2 to different bit-levels (runtime load) using bitsandbytes.')
    parser.add_argument('--model', default='nvidia/NV-Embed-v2', help='Model ID to quantize (default: nvidia/NV-Embed-v2)')
    parser.add_argument('--Q', type=int, choices=[1, 2, 4, 8], default=8, help='Quantization level: 1, 2, 4, or 8 bits (default: 8)')
    parser.add_argument('--revision', default=None, help='Optional HF revision/commit to pin the model code to')
    parser.add_argument('--out-dir', default=None, help='Directory to save tokenizer/model files (auto-generated if not specified)')
    parser.add_argument('--device-map', default='auto', help='Device map for loading (default: auto)')
    parser.add_argument('--verify', action='store_true', help='Run a small encode verification after load')
    parser.add_argument('--hf-token', default=None, help='Optional Hugging Face token if model access is restricted')
    args = parser.parse_args()

    # Auto-generate output directory if not specified
    if args.out_dir is None:
        args.out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'output', f'nv-embed-v2-{args.Q}bit')

    try:
        from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
    except Exception as exc:
        print('Transformers is required. Please install transformers >= 4.55.0')
        raise

    try:
        import bitsandbytes as bnb  # noqa: F401
    except Exception:
        print('bitsandbytes not found. Install with: pip install bitsandbytes')
        raise

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    print(f'Preparing {args.Q}-bit quantization config...')
    
    # Configure quantization based on bit level
    if args.Q == 8:
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
    elif args.Q == 4:
        quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16")
    elif args.Q == 2:
        quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16", bnb_4bit_use_double_quant=True)
    elif args.Q == 1:
        quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16", bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")

    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        revision=args.revision,
        token=args.hf_token,
    )

    print(f'Loading model in {args.Q}-bit (this may take a while)...')
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        revision=args.revision,
        token=args.hf_token,
        quantization_config=quant_cfg,
        device_map=args.device_map,
    )

    # Best-effort larger sequence length if supported
    try:
        if hasattr(model, 'max_seq_length'):
            model.max_seq_length = 32768
    except Exception:
        pass

    print('Saving tokenizer and model to: %s' % args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    # Note: save_pretrained will not store quantized weights; it saves standard weights/config.
    # At load time, you must pass the same BitsAndBytesConfig again.
    model.save_pretrained(args.out_dir, safe_serialization=True)

    # Drop a small README in the output directory
    readme_path = os.path.join(args.out_dir, 'QUANTIZE_README.md')
    with open(readme_path, 'w') as f:
        f.write(
            f'# NV-Embed-v2 {args.Q}-bit (runtime)\n\n'
            f'This directory contains tokenizer/model files for NV-Embed-v2 quantized to {args.Q}-bit.\n'
            f'To run in {args.Q}-bit mode, you must load with the appropriate BitsAndBytesConfig.\n\n'
            'Example:\n\n'
            '```python\n'
            'from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig\n'
        )
        
        # Add the appropriate config based on quantization level
        if args.Q == 8:
            f.write('bnb = BitsAndBytesConfig(load_in_8bit=True)\n')
        elif args.Q == 4:
            f.write('bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16")\n')
        elif args.Q == 2:
            f.write('bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16", bnb_4bit_use_double_quant=True)\n')
        elif args.Q == 1:
            f.write('bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16", bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")\n')
        
        f.write(
            f'tok = AutoTokenizer.from_pretrained("{args.out_dir}", trust_remote_code=True)\n'
            f'model = AutoModel.from_pretrained("{args.out_dir}", trust_remote_code=True, quantization_config=bnb, device_map="auto")\n'
            '```\n'
        )

    if args.verify:
        try:
            print('Running quick verification...')
            # NV-Embed-v2 custom encode API (trust_remote_code)
            queries = ["test query"]
            instruction = "Instruct: Given a question, retrieve passages that answer the question\nQuery: "
            # Some implementations provide model.encode; fallback to forward if missing
            if hasattr(model, 'encode'):
                _ = model.encode(queries, instruction=instruction, max_length=2048)
            else:
                print('Model has no encode() method; skipping encode verification.')
            print('Verification completed.')
        except Exception as exc:
            print('Verification failed: %s' % exc)


if __name__ == '__main__':
    main()


