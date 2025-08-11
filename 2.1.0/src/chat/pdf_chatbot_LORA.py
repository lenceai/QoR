"""
PDF Chatbot (LoRA-only, v2.0.1)

Description:
- Simple CLI chatbot that uses only the 4-bit Unsloth Mistral 24B base model
  with your saved LoRA adapter. No RAG, FAISS, or index lookup is used.

Usage:
- One-off question:
  python pdf_chatbot_LORA.py --once "Summarize the main themes."

- Interactive chat:
  python pdf_chatbot_LORA.py

Defaults:
- Base model: unsloth/Mistral-Small-3.2-24B-Instruct-2506-unsloth-bnb-4bit
- LoRA adapter: ../../output/adapters/mistral24b_lora
- Cache dir: ../../output/models/mistral24b_4bit

Notes:
- 2-bit GGUF models are not used here; this script relies on the HF 4-bit checkpoint.
"""

import os
import argparse
from typing import List, Dict, Any


def _abs_path(path_from_here):
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, path_from_here))


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA-only PDF chatbot (v2.0.1): no RAG/index")

    # Model / LoRA paths
    parser.add_argument("--model-id", default="unsloth/Mistral-Small-3.2-24B-Instruct-2506-unsloth-bnb-4bit", help="HF base model repo ID")
    parser.add_argument("--adapter-dir", default=_abs_path("../../output/adapters/mistral24b_lora"), help="Path to LoRA adapter directory")
    parser.add_argument("--cache-dir", default=_abs_path("../../output/models/mistral24b_4bit"), help="HF cache dir for base model")

    # Generation settings
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling")
    parser.add_argument("--max-input-tokens", type=int, default=4096, help="Max input tokens for prompt (history will be truncated)")
    parser.add_argument("--max-history-turns", type=int, default=16, help="Max user/assistant turns to keep in history before truncation")

    # Run mode
    parser.add_argument("--once", default=None, help="Ask a single question and exit")
    parser.add_argument("--system", default="You are a helpful assistant specializing in CERN, and a PHD in Particle Physics.", help="Optional system prompt override")

    return parser.parse_args()


def _bf16_available():
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(0)
            return major >= 8
        return False
    except Exception:
        return False


def build_model_and_tokenizer(model_id: str, cache_dir: str):
    from transformers import (
        AutoTokenizer,
        AutoConfig,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        Mistral3ForConditionalGeneration,
    )  # type: ignore
    import torch  # type: ignore

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

    # Clean up config for inference
    try:
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = True
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


def messages_to_prompt(tokenizer, messages: List[Dict[str, Any]]):
    # Prefer tokenizer chat template if available
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Minimal fallback formatting
        parts = []
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


def generate_answer(model, tokenizer, messages: List[Dict[str, Any]], max_new_tokens: int, temperature: float, top_p: float):
    import torch  # type: ignore
    prompt = messages_to_prompt(tokenizer, messages)
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
    model, tokenizer = build_model_and_tokenizer(args.model_id, args.cache_dir)
    model = load_with_lora(model, args.adapter_dir)

    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.once},
    ]
    answer = generate_answer(model, tokenizer, messages, args.max_new_tokens, args.temperature, args.top_p)
    print("\n=== Answer ===\n%s\n" % answer)


def _truncate_history_to_fit(tokenizer, system_msg: str, history: List[Dict[str, Any]], new_user: Dict[str, Any], max_input_tokens: int) -> List[Dict[str, Any]]:
    # Start with full history; caller will enforce max-history-turns afterwards
    trimmed = history[:] if history else []
    while True:
        messages = ([{"role": "system", "content": system_msg}] + trimmed + [new_user])
        prompt = messages_to_prompt(tokenizer, messages)
        try:
            num_tokens = len(tokenizer(prompt, return_tensors="pt").input_ids[0])
        except Exception:
            num_tokens = max(1, len(prompt) // 4)
        if num_tokens <= max_input_tokens or not trimmed:
            return trimmed
        if len(trimmed) >= 2:
            trimmed = trimmed[2:]
        else:
            trimmed = trimmed[1:]


def run_chat(args):
    model, tokenizer = build_model_and_tokenizer(args.model_id, args.cache_dir)
    model = load_with_lora(model, args.adapter_dir)

    print("\nLoRA-only Chatbot ready. Type your question (or 'exit' to quit).\n")
    history: List[Dict[str, Any]] = []
    system_msg = args.system
    while True:
        try:
            user_inp = input("You: ").strip()
        except EOFError:
            break
        if not user_inp:
            continue
        if user_inp.lower() in ("exit", "quit", ":q", "q"):
            break
        new_user = {"role": "user", "content": user_inp}
        trimmed = _truncate_history_to_fit(tokenizer, system_msg, history, new_user, args.max_input_tokens)
        messages = ([{"role": "system", "content": system_msg}] + trimmed + [new_user])
        answer = generate_answer(model, tokenizer, messages, args.max_new_tokens, args.temperature, args.top_p)
        print("\nAssistant: %s\n" % answer)
        history.extend([new_user, {"role": "assistant", "content": answer}])
        if args.max_history_turns > 0 and len(history) > args.max_history_turns * 2:
            history = history[-args.max_history_turns * 2:]


def main():
    # Memory fragmentation guard (optional for inference)
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") is None:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    args = parse_args()
    if args.once:
        run_once(args)
    else:
        run_chat(args)


if __name__ == "__main__":
    main()


