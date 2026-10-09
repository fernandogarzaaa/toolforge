"""Model / tokenizer loading shared by training, evaluation and export (needs the ``train`` extra)."""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any

from toolforge.config import ModelConfig

DTYPES = {"float32": "float32", "bfloat16": "bfloat16", "float16": "float16"}


def resolve_model_dir(name_or_path: str, revision: str | None = None) -> str:
    """Local directory for a model id (downloads it via huggingface_hub if needed)."""
    if Path(name_or_path).is_dir():
        return name_or_path
    from huggingface_hub import snapshot_download

    return snapshot_download(
        name_or_path,
        revision=revision,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "LICENSE"],
    )


def _read_state_dict_no_mmap(model_dir: Path) -> dict[str, Any]:
    """Read all safetensors shards into memory without mmap.

    Memory-mapping weights that live on a network / FUSE file system can kill the process with
    SIGBUS when pages are evicted, so weights are read as plain bytes and deserialised.
    """
    from safetensors.torch import load as st_load

    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files in {model_dir}")
    state: dict[str, Any] = {}
    for f in files:
        state.update(st_load(f.read_bytes()))
        gc.collect()
    return state


def load_model(cfg: ModelConfig, path: str | None = None, trainable: bool = False) -> Any:
    """Load a causal LM per ``cfg`` (optionally from ``path`` instead of ``cfg.name_or_path``)."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    if cfg.num_threads:
        torch.set_num_threads(cfg.num_threads)
    dtype = getattr(torch, DTYPES[cfg.dtype])
    model_dir = resolve_model_dir(path or cfg.name_or_path, None if path else cfg.revision)

    if cfg.quantization == "4bit":  # QLoRA path: CUDA + bitsandbytes only (untested on this box)
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(model_dir, quantization_config=bnb, device_map="auto")
        if trainable:
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.gradient_checkpointing)
        return model

    hf_cfg = AutoConfig.from_pretrained(model_dir)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(hf_cfg, dtype=dtype)
    state = _read_state_dict_no_mmap(Path(model_dir))
    for k in list(state):
        state[k] = state[k].to(dtype)
    if getattr(hf_cfg, "tie_word_embeddings", False) and "lm_head.weight" not in state:
        state["lm_head.weight"] = state["model.embed_tokens.weight"]
    missing, _unexpected = model.load_state_dict(state, strict=False, assign=True)
    del state
    gc.collect()
    if getattr(hf_cfg, "tie_word_embeddings", False):
        model.tie_weights()
    missing = [m for m in missing if not m.endswith("rotary_emb.inv_freq")]
    if missing:
        raise RuntimeError(f"weights missing from checkpoint: {missing[:5]}")
    # buffers created on the meta device (e.g. rotary inv_freq) must be materialised
    _materialize_meta_buffers(model, hf_cfg)
    model.eval()
    if trainable and cfg.gradient_checkpointing:
        # non-reentrant checkpointing does not need input grads, so frozen lower layers stay
        # out of the backward pass
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def _materialize_meta_buffers(model: Any, hf_cfg: Any) -> None:
    import torch

    for module in model.modules():
        for name, buf in list(module.named_buffers(recurse=False)):
            if buf.device.type == "meta":
                if hasattr(module, "rope_init_fn") or "inv_freq" in name:
                    fresh = type(module)(config=hf_cfg)
                    for bname, b in fresh.named_buffers(recurse=False):
                        module.register_buffer(bname, b, persistent=False)
                    break
                module.register_buffer(name, torch.zeros(buf.shape, dtype=buf.dtype), persistent=False)


def load_tokenizer(name_or_path: str, revision: str | None = None) -> Any:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(resolve_model_dir(name_or_path, revision))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def apply_adapters(model: Any, adapters: list[str]) -> Any:
    """Apply LoRA adapters in order, merging each into the weights."""
    from peft import PeftModel

    for path in adapters:
        if not Path(path).exists():
            raise FileNotFoundError(f"adapter not found: {path}")
        model = PeftModel.from_pretrained(model, path)
        model = model.merge_and_unload()
        gc.collect()
    return model


def default_threads() -> int:
    return max(1, (os.cpu_count() or 2))
