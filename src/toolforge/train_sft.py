"""`toolforge sft`: supervised fine-tuning with LoRA (or QLoRA on GPU) via PEFT + TRL.

Examples are rendered with :mod:`toolforge.formatting` and tokenised here (prompt tokens are
masked out of the loss), so the exact bytes the model is trained on are the bytes it is
evaluated on. Over-long examples are dropped, never truncated, so no target is cut mid-call.
Training resumes from the newest ``checkpoint-*`` in ``output_dir`` when ``resume`` is true.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from toolforge.config import RunConfig, TrainConfig
from toolforge.data.build import read_jsonl
from toolforge.formatting import IM_END, render_messages, render_target


def build_text_pairs(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    """(prompt, completion) strings for SFT; the completion ends with ``<|im_end|>``."""
    return [
        {
            "prompt": render_messages(r["messages"], r["tools"], add_generation_prompt=True),
            "completion": render_target(r["target"]) + IM_END,
        }
        for r in records
    ]


def tokenize_pairs(
    tokenizer: Any, pairs: list[dict[str, str]], max_length: int
) -> tuple[list[dict[str, list[int]]], int]:
    """Tokenise prompt and completion separately; drop rows longer than ``max_length``."""
    rows: list[dict[str, list[int]]] = []
    dropped = 0
    for p in pairs:
        prompt_ids = tokenizer(p["prompt"], add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(p["completion"], add_special_tokens=False)["input_ids"]
        ids = prompt_ids + completion_ids
        if len(ids) > max_length:
            dropped += 1
            continue
        rows.append({"input_ids": ids, "completion_mask": [0] * len(prompt_ids) + [1] * len(completion_ids)})
    return rows, dropped


def last_checkpoint(output_dir: str | Path) -> str | None:
    ckpts = [p for p in Path(output_dir).glob("checkpoint-*") if p.is_dir() and p.name.split("-")[-1].isdigit()]
    if not ckpts:
        return None
    return str(max(ckpts, key=lambda p: int(p.name.split("-")[-1])))


def completion_loss(model: Any, inputs: dict[str, Any], num_items_in_batch: Any = None) -> Any:
    """Causal-LM loss on supervised tokens, computing logits only where they are needed.

    Positions before the earliest supervised token in the batch never reach the LM head
    (``logits_to_keep``). The loss is identical to the full computation: a sum of token
    cross-entropies divided by ``num_items_in_batch`` (all supervised tokens in the accumulated
    batch) or, when that is not given, by the supervised tokens in this batch.
    """
    import torch.nn.functional as F

    labels = inputs["labels"]
    mask = labels != -100
    length = labels.shape[1]
    if bool(mask.any()):
        first = int(mask.float().argmax(dim=1)[mask.any(dim=1)].min())
    else:
        first = length - 1
    first = max(first, 1)
    keep = length - first + 1
    out = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        logits_to_keep=keep,
        use_cache=False,
    )
    logits = out.logits[:, :-1, :]
    targets = labels[:, length - keep + 1 :]
    loss = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100, reduction="sum"
    )
    denom = num_items_in_batch if num_items_in_batch is not None else mask.sum().clamp(min=1)
    return loss / denom


def lora_config(tc: TrainConfig) -> Any:
    from peft import LoraConfig

    return LoraConfig(
        r=tc.lora.r,
        lora_alpha=tc.lora.alpha,
        lora_dropout=tc.lora.dropout,
        target_modules=tc.lora.target_modules,
        layers_to_transform=tc.lora.layers_to_transform,
        task_type="CAUSAL_LM",
    )


def adapt_args(config_cls: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Map our trainer kwargs onto whatever this transformers/TRL version accepts.

    transformers 5 folded ``warmup_ratio`` into ``warmup_steps`` (a float < 1 is a ratio).
    """
    import dataclasses

    names = {f.name for f in dataclasses.fields(config_cls)}
    out = dict(kwargs)
    if "warmup_ratio" in out and "warmup_ratio" not in names:
        ratio = out.pop("warmup_ratio")
        out.setdefault("warmup_steps", ratio)
    if "use_cpu" in out and "use_cpu" not in names:
        out.pop("use_cpu")
    return out


def common_args(cfg: RunConfig, tc: TrainConfig) -> dict[str, Any]:
    """Trainer arguments shared by SFT and DPO."""
    import torch

    use_bf16 = cfg.model.dtype == "bfloat16"
    use_fp16 = cfg.model.dtype == "float16" and torch.cuda.is_available()
    return {
        "output_dir": tc.output_dir,
        "per_device_train_batch_size": tc.batch_size,
        "gradient_accumulation_steps": tc.grad_accum,
        "learning_rate": tc.lr,
        "num_train_epochs": tc.epochs,
        "max_steps": tc.max_steps,
        "warmup_ratio": tc.warmup_ratio,
        "weight_decay": tc.weight_decay,
        "lr_scheduler_type": tc.lr_scheduler,
        "logging_steps": tc.logging_steps,
        "save_strategy": "steps",
        "save_steps": tc.save_steps,
        "save_total_limit": tc.save_total_limit,
        "seed": tc.seed,
        "bf16": use_bf16,
        "fp16": use_fp16,
        "report_to": "none",
        "gradient_checkpointing": cfg.model.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "dataloader_num_workers": 0,
        "remove_unused_columns": False,
        "use_cpu": not torch.cuda.is_available(),
    }


def run_sft(cfg: RunConfig) -> dict[str, Any]:
    from datasets import Dataset
    from peft import get_peft_model
    from trl import SFTConfig as TRLSFTConfig
    from trl import SFTTrainer

    from toolforge.modeling import load_model, load_tokenizer
    from toolforge.tracking import Tracker, hf_callback

    tc = cfg.sft
    out = Path(tc.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(tc.train_file)
    if tc.num_examples is not None:
        records = records[: tc.num_examples]
    tokenizer = load_tokenizer(cfg.model.name_or_path, cfg.model.revision)
    rows, dropped = tokenize_pairs(tokenizer, build_text_pairs(records), tc.max_length)
    if not rows:
        raise RuntimeError("no training rows left after length filtering")
    dataset = Dataset.from_list(rows)
    n_tokens = sum(len(r["input_ids"]) for r in rows)

    model = load_model(cfg.model, trainable=True)
    model = get_peft_model(model, lora_config(tc))
    args = TRLSFTConfig(
        **adapt_args(
            TRLSFTConfig,
            {
                **common_args(cfg, tc),
                "max_length": tc.max_length,
                "completion_only_loss": True,
                "packing": False,
                "eos_token": IM_END if IM_END in tokenizer.get_vocab() else None,
            },
        )
    )
    tracker = Tracker(cfg.tracking, f"{cfg.run_name}-sft", out)
    tracker.log_params(
        {
            "stage": "sft",
            "model": cfg.model.model_dump(),
            "sft": tc.model_dump(),
            "n_rows": len(rows),
            "n_dropped_too_long": dropped,
            "n_tokens": n_tokens,
        }
    )

    class _Trainer(SFTTrainer):  # type: ignore[misc]
        def compute_loss(
            self, model: Any, inputs: Any, return_outputs: bool = False, num_items_in_batch: Any = None
        ) -> Any:
            if not tc.completion_logits_only:
                return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
            loss = completion_loss(model, inputs, num_items_in_batch)
            return (loss, None) if return_outputs else loss

    trainer = _Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[hf_callback(tracker)],
    )
    resume = last_checkpoint(out) if tc.resume else None
    if resume:
        print(f"[toolforge] resuming SFT from {resume}")
    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=resume)
    elapsed = time.time() - t0
    adapter_dir = out / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    summary = {
        "stage": "sft",
        "examples": len(rows),
        "dropped_too_long": dropped,
        "tokens": n_tokens,
        "global_steps": trainer.state.global_step,
        "train_loss": float(result.training_loss),
        "wall_time_s_this_session": round(elapsed, 1),
        "resumed_from": resume,
        "adapter": str(adapter_dir),
        "dtype": cfg.model.dtype,
        "quantization": cfg.model.quantization,
    }
    _append_summary(out / "train_summary.json", summary)
    tracker.log_metrics({"final_train_loss": summary["train_loss"], "wall_time_s": elapsed})
    tracker.close()
    return summary


def _append_summary(path: Path, summary: dict[str, Any]) -> None:
    """Keep one entry per training session so resumed runs report their total wall time."""
    sessions: list[dict[str, Any]] = []
    if path.exists():
        prev = json.loads(path.read_text())
        sessions = prev.get("sessions", [])
    sessions.append(summary)
    total = sum(s.get("wall_time_s_this_session", 0) for s in sessions)
    path.write_text(
        json.dumps({"latest": summary, "sessions": sessions, "total_wall_time_s": round(total, 1)}, indent=2)
    )
