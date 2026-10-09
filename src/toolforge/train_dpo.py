"""`toolforge dpo`: mine preference pairs from the SFT model's own mistakes, then run DPO.

1. **Sample**: the SFT model (base + SFT adapter, merged) completes each DPO prompt
   ``samples_per_prompt`` times (first greedy, the rest sampled). Results are appended to
   ``generations_file`` so an interrupted run resumes where it stopped.
2. **Pair**: :func:`toolforge.pairs.build_pairs` keeps prompts where some completion is wrong
   (chosen = gold, rejected = wrong completion) and, if enabled, adds verified synthetic
   corruptions for prompts the model already gets right.
3. **Train**: a fresh LoRA adapter on top of the merged SFT weights, trained with TRL's
   DPOTrainer. With PEFT, the reference model is the same network with the adapter disabled,
   i.e. exactly the SFT model, so no second copy is held in memory.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from toolforge.config import RunConfig
from toolforge.data.build import read_jsonl
from toolforge.formatting import IM_END, render_messages
from toolforge.pairs import build_pairs


def _load_generations(path: Path) -> dict[str, list[str]]:
    gens: dict[str, list[str]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                gens[row["id"]] = row["completions"]
    return gens


def sample_generations(cfg: RunConfig, records: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Sample SFT-model completions for each prompt (resumable)."""
    from toolforge.generate import HFGenerator
    from toolforge.modeling import apply_adapters, load_model, load_tokenizer

    dc = cfg.dpo
    path = Path(dc.generations_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    gens = _load_generations(path)
    todo = [r for r in records if r["id"] not in gens]
    if not todo:
        return gens
    print(f"[toolforge] sampling {len(todo)} prompts x {dc.samples_per_prompt} ({len(gens)} already done)")
    tok = load_tokenizer(cfg.model.name_or_path, cfg.model.revision)
    model = apply_adapters(load_model(cfg.model), [dc.sft_adapter])
    gen = HFGenerator(model, tok)
    bs = cfg.eval.batch_size
    with path.open("a", encoding="utf-8") as fh:
        for i in range(0, len(todo), bs):
            batch = todo[i : i + bs]
            prompts = [render_messages(r["messages"], r["tools"], add_generation_prompt=True) for r in batch]
            per_prompt: list[list[str]] = [[] for _ in batch]
            for s in range(dc.samples_per_prompt):
                temp = 0.0 if s == 0 else dc.temperature
                outs = gen.generate(prompts, dc.max_new_tokens, temperature=temp, seed=dc.seed + s + i)
                for j, o in enumerate(outs):
                    per_prompt[j].append(o)
            for r, comps in zip(batch, per_prompt):
                gens[r["id"]] = comps
                fh.write(json.dumps({"id": r["id"], "completions": comps}, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[toolforge] sampled {min(i + bs, len(todo))}/{len(todo)}", flush=True)
    del model
    return gens


def make_pairs(cfg: RunConfig) -> list[dict[str, Any]]:
    dc = cfg.dpo
    records = read_jsonl(dc.prompts_file)
    if dc.num_prompts is not None:
        records = records[: dc.num_prompts]
    gens = sample_generations(cfg, records)
    pairs = build_pairs(records, gens, synthetic=dc.synthetic_negatives, seed=dc.seed)
    rows = [p.to_dict() for p in pairs]
    Path(dc.pairs_file).parent.mkdir(parents=True, exist_ok=True)
    with Path(dc.pairs_file).open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    greedy_wrong = sum(1 for p in pairs if p.source.startswith("model:"))
    stats = {
        "prompts": len(records),
        "pairs": len(rows),
        "sources": dict(Counter(p.source.split(":")[0] + ":" + p.source.split(":")[1] for p in pairs)),
        "prompts_with_model_error": greedy_wrong,
    }
    Path(dc.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(dc.output_dir) / "pair_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[toolforge] pairs: {json.dumps(stats)}")
    return rows


def run_dpo(cfg: RunConfig) -> dict[str, Any]:
    from datasets import Dataset
    from peft import get_peft_model
    from trl import DPOConfig as TRLDPOConfig
    from trl import DPOTrainer

    from toolforge.compat import patch_trl_logprob
    from toolforge.modeling import apply_adapters, load_model, load_tokenizer
    from toolforge.tracking import Tracker, hf_callback
    from toolforge.train_sft import _append_summary, adapt_args, common_args, last_checkpoint, lora_config

    if patch_trl_logprob():
        print("[toolforge] Triton unavailable: using the pure-PyTorch log-prob path for DPO")
    dc = cfg.dpo
    out = Path(dc.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs_path = Path(dc.pairs_file)
    rows = read_jsonl(pairs_path) if pairs_path.exists() and dc.resume else make_pairs(cfg)
    tok = load_tokenizer(cfg.model.name_or_path, cfg.model.revision)
    kept: list[dict[str, str]] = []
    for r in rows:
        n_prompt = len(tok(r["prompt"], add_special_tokens=False)["input_ids"])
        n_long = max(len(tok(r[k] + IM_END, add_special_tokens=False)["input_ids"]) for k in ("chosen", "rejected"))
        if n_prompt <= dc.max_prompt_length and n_prompt + n_long <= dc.max_length:
            kept.append({"prompt": r["prompt"], "chosen": r["chosen"] + IM_END, "rejected": r["rejected"] + IM_END})
    if not kept:
        raise RuntimeError("no DPO pairs left after length filtering")
    dataset = Dataset.from_list(kept)

    model = apply_adapters(load_model(cfg.model, trainable=False), [dc.sft_adapter])
    if cfg.model.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, lora_config(dc))
    args = TRLDPOConfig(
        **adapt_args(TRLDPOConfig, {**common_args(cfg, dc), "beta": dc.beta, "max_length": dc.max_length})
    )
    tracker = Tracker(cfg.tracking, f"{cfg.run_name}-dpo", out)
    tracker.log_params({"stage": "dpo", "model": cfg.model.model_dump(), "dpo": dc.model_dump(), "n_pairs": len(kept)})
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=dataset,
        processing_class=tok,
        callbacks=[hf_callback(tracker)],
    )
    resume = last_checkpoint(out) if dc.resume else None
    if resume:
        print(f"[toolforge] resuming DPO from {resume}")
    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=resume)
    elapsed = time.time() - t0
    adapter_dir = out / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    summary = {
        "stage": "dpo",
        "pairs": len(kept),
        "pairs_dropped_too_long": len(rows) - len(kept),
        "global_steps": trainer.state.global_step,
        "train_loss": float(result.training_loss),
        "wall_time_s_this_session": round(elapsed, 1),
        "resumed_from": resume,
        "adapter": str(adapter_dir),
    }
    _append_summary(out / "train_summary.json", summary)
    tracker.log_metrics({"final_train_loss": summary["train_loss"], "wall_time_s": elapsed})
    tracker.close()
    return summary
