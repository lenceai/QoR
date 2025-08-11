"""
Mistral 24B (Unsloth 4-bit) – LoRA Fine-Tuning with Multi-GPU FSDP (v2.1.0)

This script mirrors mistral24b_unsloth_4bit_finetune.py, but enables multi-GPU
training via PyTorch FSDP through Hugging Face Accelerate integration.

Notes:
- Requires PyTorch >= 2.7.1 and accelerate >= 1.10.0.
- Launch with accelerate for single-node or multi-node.

Examples:
- Single node (2 GPUs):
  accelerate launch --config_file accelerate_fsdp.yaml \
    src/training/mistral24b_unsloth_4bit_finetune_multi_gpu.py --train \
    --epochs 2 --save-steps 100 --lora-target-modules "q_proj,v_proj,o_proj"

- Multi-node (2 machines, same network):
  On both boxes set matching envs, then run on each box (with proper machine_rank):
    accelerate launch --num_processes 2 --num_machines 2 --machine_rank 0 \\
      --main_process_ip <MASTER_IP> --main_process_port 29500 \\
      src/training/mistral24b_unsloth_4bit_finetune_multi_gpu.py --train \\
      --epochs 2 --save-steps 100 --lora-target-modules "q_proj,v_proj,o_proj"

    accelerate launch --num_processes 2 --num_machines 2 --machine_rank 1 \\
      --main_process_ip <MASTER_IP> --main_process_port 29500 \\
      src/training/mistral24b_unsloth_4bit_finetune_multi_gpu.py --train \\
      --epochs 2 --save-steps 100 --lora-target-modules "q_proj,v_proj,o_proj"

You can also create an accelerate config via `accelerate config` and pass
`--fsdp "full_shard auto_wrap"` when prompted, or provide a YAML config.
"""

import os
import sys
import argparse
import pickle
from typing import List


def _abs_path(path_from_here):
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, path_from_here))


def parse_args():
    parser = argparse.ArgumentParser(description="Unsloth 4-bit Mistral 24B LoRA fine-tune with FSDP (multi-GPU)")

    # Model / cache
    parser.add_argument("--model-id", default="unsloth/Mistral-Small-3.2-24B-Instruct-2506-unsloth-bnb-4bit", help="HF model repo ID")
    parser.add_argument("--cache-dir", default=_abs_path("../../output/models/mistral24b_4bit"), help="Where to cache model files")
    parser.add_argument("--download-only", action="store_true", help="Only download model snapshots; do not initialize the model")

    # Data
    parser.add_argument("--metadata-path", default=_abs_path("../../output/cern_explorer_metadata.pkl"), help="Pickle of text chunks from vector_db build")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit number of chunks used for training (0 = no limit)")

    # Training / LoRA
    parser.add_argument("--train", action="store_true", help="Launch LoRA fine-tuning")
    parser.add_argument("--output-dir", default=_abs_path("../../output/adapters/mistral24b_lora"), help="Where to save LoRA adapter")
    parser.add_argument("--micro-batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--grad-accum-steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--save-steps", type=int, default=200, help="Save every N steps")
    parser.add_argument("--logging-steps", type=int, default=20, help="Log every N steps")
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="Warmup ratio")
    parser.add_argument("--max-seq-len", type=int, default=1024, help="Sequence length for tokenization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # LoRA specifics
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--lora-target-modules", default="q_proj,v_proj", help="Comma-separated target modules for LoRA")

    # FSDP knobs
    parser.add_argument("--fsdp-auto-wrap-policy", default="transformer_layer", choices=["transformer_layer", "size_based"], help="Auto wrap policy")
    parser.add_argument("--fsdp-sharding-strategy", default="full_shard", choices=["full_shard", "shard_grad_op", "zero2"], help="Sharding strategy")
    parser.add_argument("--fsdp-mixed-precision", default="bf16", choices=["bf16", "fp16"], help="FSDP mixed precision for compute")
    parser.add_argument("--fsdp-state-dtype", default="fp32", choices=["fp32", "bf16"], help="Param/state dtype")
    parser.add_argument("--fsdp-gradient-checkpointing", action="store_true", help="Enable activation checkpointing under FSDP")

    return parser.parse_args()


def download_snapshot(model_id: str, cache_dir: str) -> str:
    from huggingface_hub import snapshot_download  # type: ignore
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Downloading snapshot for {model_id} to {cache_dir}")
    local_dir = snapshot_download(repo_id=model_id, cache_dir=cache_dir, local_dir=cache_dir, local_dir_use_symlinks=False)
    print(f"Snapshot downloaded at: {local_dir}")
    return local_dir


def detect_bf16() -> bool:
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(0)
            return major >= 8
        return False
    except Exception:
        return False


def load_pdf_chunks(metadata_path: str, max_samples: int = 0) -> List[str]:
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found at {metadata_path}. Build it via vector_db first.")
    with open(metadata_path, "rb") as f:
        chunks = pickle.load(f)
    chunks = [c for c in chunks if isinstance(c, str) and c.strip()]
    if max_samples and max_samples > 0:
        chunks = chunks[:max_samples]
    print(f"Loaded {len(chunks)} text chunks from metadata.")
    return chunks


def build_model_and_tokenizer(model_id: str, cache_dir: str):
    from transformers import (
        AutoTokenizer,
        AutoConfig,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        Mistral3ForConditionalGeneration,
    )  # type: ignore
    import torch  # type: ignore

    bf16 = detect_bf16()
    compute_dtype = torch.bfloat16 if bf16 else torch.float16

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading 4-bit model (FSDP-ready): {model_id}")
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

    common_kwargs = dict(
        device_map=None,  # None for FSDP; let accelerate handle placement
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        torch_dtype=compute_dtype,
        cache_dir=cache_dir,
    )

    if use_mistral3_cg:
        if pass_bnb:
            model = Mistral3ForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=bnb_cfg,
                **common_kwargs,
            )
        else:
            model = Mistral3ForConditionalGeneration.from_pretrained(
                model_id,
                **common_kwargs,
            )
    else:
        if pass_bnb:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_cfg,
                **common_kwargs,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **common_kwargs,
            )

    # Enable caching for inference; disable for training later in args
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    return model, tokenizer


def wrap_with_lora(model, lora_r: int, lora_alpha: int, lora_dropout: float, target_modules: List[str]):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # type: ignore

    print("Preparing model for k-bit training and applying LoRA (FSDP)...")
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    try:
        model.print_trainable_parameters()
    except Exception:
        pass
    return model


def train_lora(
    model_id: str,
    cache_dir: str,
    metadata_path: str,
    output_dir: str,
    micro_bs: int,
    grad_accum: int,
    epochs: int,
    lr: float,
    save_steps: int,
    logging_steps: int,
    warmup_ratio: float,
    max_seq_len: int,
    seed: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules_csv: str,
    max_samples: int,
    fsdp_auto_wrap_policy: str,
    fsdp_sharding_strategy: str,
    fsdp_mixed_precision: str,
    fsdp_state_dtype: str,
    fsdp_gradient_checkpointing: bool,
):
    import torch  # type: ignore
    from transformers import (
        DataCollatorForLanguageModeling,
        TrainingArguments,
        Trainer,
        set_seed,
    )  # type: ignore

    model, tokenizer = build_model_and_tokenizer(model_id, cache_dir)

    texts = load_pdf_chunks(metadata_path, max_samples=max_samples)
    class ChunkTextDataset(object):
        def __init__(self, texts: List[str], tokenizer, max_length: int):
            self.samples = []
            for t in texts:
                t2 = t.strip()
                if not t2:
                    continue
                encoded = tokenizer(t2, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
                self.samples.append({k: v.squeeze(0) for k, v in encoded.items()})
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            return dict(self.samples[idx])

    dataset = ChunkTextDataset(texts, tokenizer, max_seq_len)

    target_modules = [m.strip() for m in lora_target_modules_csv.split(",") if m.strip()]
    model = wrap_with_lora(model, lora_r, lora_alpha, lora_dropout, target_modules)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    os.makedirs(output_dir, exist_ok=True)
    bf16 = detect_bf16()
    fp16 = not bf16
    set_seed(seed)

    # FSDP configs
    def to_mixed_precision(mp: str):
        return "bf16" if mp == "bf16" else "fp16"
    fsdp_config = {
        "fsdp": fsdp_sharding_strategy,
        "fsdp_auto_wrap_policy": fsdp_auto_wrap_policy,
        "fsdp_transformer_layer_cls_to_wrap": "MistralDecoderLayer,DecoderLayer,TransformerLayer",
        "bf16": bf16 and (fsdp_mixed_precision == "bf16"),
        "fp16": (not bf16) or (fsdp_mixed_precision == "fp16"),
    }

    # Accelerate picks up these via env/config. We pass via TrainingArguments
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=micro_bs,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        learning_rate=lr,
        logging_steps=logging_steps,
        logging_strategy="steps",
        logging_first_step=True,
        disable_tqdm=False,
        log_level="info",
        save_steps=save_steps,
        save_total_limit=2,
        warmup_ratio=warmup_ratio,
        bf16=bf16,
        fp16=fp16,
        prediction_loss_only=True,
        gradient_checkpointing=fsdp_gradient_checkpointing,
        optim="adamw_bnb_8bit",
        report_to=[],
        seed=seed,
        fsdp=fsdp_config["fsdp"],
        fsdp_auto_wrap_policy=fsdp_config["fsdp_auto_wrap_policy"],
        fsdp_transformer_layer_cls_to_wrap=fsdp_config["fsdp_transformer_layer_cls_to_wrap"],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
    )

    print("Starting FSDP training...")
    trainer.train()

    print(f"Saving LoRA adapter to {output_dir}")
    model.save_pretrained(output_dir)


def main():
    args = parse_args()

    if args.download_only:
        download_snapshot(args.model_id, args.cache_dir)
        return

    if args.train:
        try:
            import torch  # type: ignore
            if not torch.cuda.is_available():
                print("CUDA/GPU not available. FSDP requires CUDA GPUs.")
                sys.exit(1)
        except Exception:
            print("PyTorch not available; cannot train.")
            sys.exit(1)

        # Encourage expandable segments for fragmentation mitigation
        if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") is None:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        train_lora(
            model_id=args.model_id,
            cache_dir=args.cache_dir,
            metadata_path=args.metadata_path,
            output_dir=args.output_dir,
            micro_bs=args.micro_batch_size,
            grad_accum=args.grad_accum_steps,
            epochs=args.epochs,
            lr=args.lr,
            save_steps=args.save_steps,
            logging_steps=args.logging_steps,
            warmup_ratio=args.warmup_ratio,
            max_seq_len=args.max_seq_len,
            seed=args.seed,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules_csv=args.lora_target_modules,
            max_samples=args.max_samples,
            fsdp_auto_wrap_policy=args.fsdp_auto_wrap_policy,
            fsdp_sharding_strategy=args.fsdp_sharding_strategy,
            fsdp_mixed_precision=args.fsdp_mixed_precision,
            fsdp_state_dtype=args.fsdp_state_dtype,
            fsdp_gradient_checkpointing=args.fsdp_gradient_checkpointing,
        )
        return

    print("Nothing to do. Use --download-only to fetch the model, or --train to run FSDP LoRA fine-tuning.")


if __name__ == "__main__":
    main()


