"""
PDF Chatbot (v2.0.1)

Small CLI chatbot to answer questions about the ingested CERN PDFs using a
RAG pipeline:
- Retrieve top-k relevant chunks from the FAISS index built by `vector_db.py`
- Feed context + question to the 4-bit Unsloth Mistral 24B with your LoRA adapter

Usage:
- One-off question:
  python pdf_chatbot.py --once "What is the LHC luminosity?"

- Interactive chat:
  python pdf_chatbot.py

Defaults assume you have built the index (output/cern_explorer.index, metadata.pkl)
and trained/saved the LoRA adapter to output/adapters/mistral24b_lora.

Notes:
- This script loads a 4-bit quantized HF model (Unsloth) and applies LoRA adapter.
- 2-bit GGUF files are inference-only and not used here.
"""

import os
import sys
import argparse
import pickle
from typing import List, Dict, Any


def _abs_path(path_from_here):
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, path_from_here))


def parse_args():
    parser = argparse.ArgumentParser(description="Chatbot over CERN PDFs using FAISS + LoRA (v2.0.1)")
    # RAG / index
    parser.add_argument("--index-path", default=_abs_path("../../output/cern_explorer.index"), help="Path to FAISS index")
    parser.add_argument("--metadata-path", default=_abs_path("../../output/cern_explorer_metadata.pkl"), help="Path to metadata (text chunks)")
    parser.add_argument("--top-k", type=int, default=4, help="Top-k chunks to retrieve")

    # Model / LoRA
    parser.add_argument("--model-id", default="unsloth/Mistral-Small-3.2-24B-Instruct-2506-unsloth-bnb-4bit", help="HF base model repo ID")
    parser.add_argument("--adapter-dir", default=_abs_path("../../output/adapters/mistral24b_lora"), help="Path to LoRA adapter directory")
    parser.add_argument("--cache-dir", default=_abs_path("../../output/models/mistral24b_4bit"), help="HF cache dir for base model")

    # Generation
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling")
    parser.add_argument("--max-input-tokens", type=int, default=4096, help="Max input tokens for prompt (history will be truncated)")
    parser.add_argument("--max-history-turns", type=int, default=8, help="Max user/assistant turns to keep in history before truncation")
    parser.add_argument("--max-context-chars", type=int, default=2000, help="Max characters from retrieved chunks to include per turn")

    # System prompt
    parser.add_argument("--system", default=(
        "You are a helpful assistant. Answer the user's question using only the provided context. "
        "If the answer is not in the context, say you don't know."
    ), help="Optional system prompt override")

    # Run mode
    parser.add_argument("--once", default=None, help="Ask a single question and exit")

    return parser.parse_args()


def load_index_and_metadata(index_path: str, metadata_path: str):
    # Lazy import to avoid mandatory FAISS dependency on --help
    try:
        import faiss  # type: ignore
    except Exception:
        print("faiss is required. Install faiss-cpu or faiss-gpu.")
        raise

    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError("Index or metadata not found. Build via vector_db.py --build first.")

    print("Loading FAISS index from %s" % index_path)
    index = faiss.read_index(index_path)  # type: ignore
    print("Loading metadata from %s" % metadata_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    if not isinstance(metadata, list) or not metadata:
        raise ValueError("Metadata is empty or invalid.")
    return index, metadata


def retrieve_top_k(index, metadata: List[str], query: str, k: int):
    # Lazy import numpy
    try:
        import numpy as np  # type: ignore
    except Exception:
        print("numpy is required.")
        raise
    # Simple embedding with all-MiniLM to encode the query only
    # Reuse the same model name used for index build by default
    from sentence_transformers import SentenceTransformer  # type: ignore
    model = SentenceTransformer("all-MiniLM-L6-v2")
    q_vec = model.encode([query], convert_to_tensor=False)
    q = np.array(q_vec).astype("float32")
    distances, indices = index.search(q, k)
    hits = []
    for j, i in enumerate(indices[0]):
        if 0 <= i < len(metadata):
            hits.append((metadata[i], float(distances[0][j])))
    return hits


def build_model_and_tokenizer(model_id: str, cache_dir: str):
    from transformers import (
        AutoTokenizer,
        AutoConfig,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        Mistral3ForConditionalGeneration,
    )  # type: ignore
    import torch  # type: ignore

    # Prefer bf16 when available
    def _bf16_available():
        try:
            if torch.cuda.is_available():
                major, _ = torch.cuda.get_device_capability(0)
                return major >= 8
            return False
        except Exception:
            return False

    compute_dtype = torch.bfloat16 if _bf16_available() else torch.float16

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    print("Loading tokenizer: %s" % model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading 4-bit model: %s" % model_id)
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True, cache_dir=cache_dir)
    try:
        cfg_dict = cfg.to_dict()
    except Exception:
        cfg_dict = {}
    use_mistral3_cg = False
    try:
        archs = getattr(cfg, "architectures", None)
        if archs:
            for _name in archs:
                if str(_name) == "Mistral3ForConditionalGeneration":
                    use_mistral3_cg = True
                    break
        if not use_mistral3_cg:
            mt = str(getattr(cfg, "model_type", "")).lower()
            if mt == "mistral3":
                use_mistral3_cg = True
    except Exception:
        pass

    pass_bnb = "quantization_config" not in cfg_dict

    if use_mistral3_cg:
        if pass_bnb:
            model = Mistral3ForConditionalGeneration.from_pretrained(
                model_id,
                device_map={"": 0},
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                quantization_config=bnb_cfg,
                torch_dtype=compute_dtype,
                cache_dir=cache_dir,
            )
        else:
            model = Mistral3ForConditionalGeneration.from_pretrained(
                model_id,
                device_map={"": 0},
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                torch_dtype=compute_dtype,
                cache_dir=cache_dir,
            )
    else:
        if pass_bnb:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map={"": 0},
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                quantization_config=bnb_cfg,
                torch_dtype=compute_dtype,
                cache_dir=cache_dir,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map={"": 0},
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                torch_dtype=compute_dtype,
                cache_dir=cache_dir,
            )

    # Align config flags
    try:
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = True  # ok for inference
        # Remove unknown loss_type if present
        if hasattr(model.config, "loss_type"):
            try:
                delattr(model.config, "loss_type")
            except Exception:
                try:
                    model.config.__dict__.pop("loss_type", None)
                except Exception:
                    pass
        text_cfg = getattr(model.config, "text_config", None)
        if text_cfg is not None and hasattr(text_cfg, "loss_type"):
            try:
                delattr(text_cfg, "loss_type")
            except Exception:
                try:
                    text_cfg.__dict__.pop("loss_type", None)
                except Exception:
                    pass
    except Exception:
        pass

    return model, tokenizer


def load_with_lora(model, adapter_dir: str):
    from peft import PeftModel  # type: ignore
    print("Loading LoRA adapter from %s" % adapter_dir)
    model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    try:
        model.eval()
    except Exception:
        pass
    return model


def _format_retrieved(retrieved: List[str], max_chars: int) -> str:
    # Concatenate bullet items up to max_chars
    pieces: List[str] = []
    total = 0
    for c in retrieved:
        if not isinstance(c, str):
            continue
        item = "- %s" % c.strip()
        if total + len(item) > max_chars and pieces:
            break
        pieces.append(item)
        total += len(item)
    return "\n\n".join(pieces)


def _apply_chat_template(tokenizer, messages: List[Dict[str, Any]]) -> str:
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Fallback for tokenizers without chat templates
        parts: List[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.append("<s>[SYSTEM]\n%s\n[/SYSTEM]" % content)
            elif role == "assistant":
                parts.append("[ASSISTANT]\n%s\n[/ASSISTANT]" % content)
            else:
                parts.append("[USER]\n%s\n[/USER]" % content)
        parts.append("[ASSISTANT]")
        return "\n".join(parts)


def _truncate_history_to_fit(
    tokenizer,
    system_msg: str,
    history: List[Dict[str, Any]],
    user_content: str,
    max_input_tokens: int,
    max_history_turns: int,
) -> List[Dict[str, Any]]:
    # Trim by number of turns first (defensive)
    trimmed = history[-(max(0, max_history_turns) * 2):] if history else []
    # Then drop oldest until under token limit
    while True:
        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_msg}] + trimmed + [{"role": "user", "content": user_content}]
        prompt = _apply_chat_template(tokenizer, messages)
        try:
            num_tokens = len(tokenizer(prompt, return_tensors="pt").input_ids[0])
        except Exception:
            # Fallback: approximate by characters/4
            num_tokens = max(1, len(prompt) // 4)
        if num_tokens <= max_input_tokens or not trimmed:
            return trimmed
        # Drop the oldest pair (user+assistant) if available, else drop first
        if len(trimmed) >= 2:
            trimmed = trimmed[2:]
        else:
            trimmed = trimmed[1:]


def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, top_p: float):
    import torch  # type: ignore
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text


def run_once(args):
    index, metadata = load_index_and_metadata(args.index_path, args.metadata_path)
    # Retrieved contexts
    hits = retrieve_top_k(index, metadata, args.once, args.top_k)
    retrieved = [c for c, _ in hits]

    # Model + tokenizer + LoRA
    model, tokenizer = build_model_and_tokenizer(args.model_id, args.cache_dir)
    model = load_with_lora(model, args.adapter_dir)

    context_text = _format_retrieved(retrieved, args.max_context_chars)
    system_msg = args.system
    user_content = "Context:\n%s\n\nQuestion: %s" % (context_text, args.once)
    messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_content}]
    prompt = _apply_chat_template(tokenizer, messages)
    text = generate_answer(model, tokenizer, prompt, args.max_new_tokens, args.temperature, args.top_p)
    print("\n=== Answer ===\n%s\n" % text)


def run_chat(args):
    index, metadata = load_index_and_metadata(args.index_path, args.metadata_path)
    model, tokenizer = build_model_and_tokenizer(args.model_id, args.cache_dir)
    model = load_with_lora(model, args.adapter_dir)

    print("\nPDF Chatbot ready. Type your question (or 'exit' to quit).\n")
    history: List[Dict[str, Any]] = []
    system_msg = args.system
    while True:
        try:
            question = input("You: ").strip()
        except EOFError:
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit", ":q", "q"):
            break
        hits = retrieve_top_k(index, metadata, question, args.top_k)
        retrieved = [c for c, _ in hits]
        context_text = _format_retrieved(retrieved, args.max_context_chars)
        user_content = "Context:\n%s\n\nQuestion: %s" % (context_text, question)

        # Truncate history to fit token budget
        trimmed_history = _truncate_history_to_fit(
            tokenizer,
            system_msg,
            history,
            user_content,
            args.max_input_tokens,
            args.max_history_turns,
        )
        messages: List[Dict[str, Any]] = ([{"role": "system", "content": system_msg}] + trimmed_history + [{"role": "user", "content": user_content}])
        prompt = _apply_chat_template(tokenizer, messages)
        answer = generate_answer(model, tokenizer, prompt, args.max_new_tokens, args.temperature, args.top_p)
        print("\nAssistant: %s\n" % answer)
        # Update full history (not just trimmed), then optionally cap by turns
        history.extend([{"role": "user", "content": user_content}, {"role": "assistant", "content": answer}])
        if args.max_history_turns > 0 and len(history) > args.max_history_turns * 2:
            history = history[-args.max_history_turns * 2:]


def main():
    # Ensure expandable CUDA segments for reduced fragmentation (optional for inference)
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") is None:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    args = parse_args()

    if args.once:
        run_once(args)
    else:
        run_chat(args)


if __name__ == "__main__":
    main()


