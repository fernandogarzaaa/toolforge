"""`toolforge eval`: BFCL-style evaluation of base vs SFT vs SFT+DPO.

Per model, raw generations are appended to ``<out_dir>/predictions/<model>.jsonl`` (resumable),
then scored with :mod:`toolforge.scorer`. The report has per-category accuracy with percentile
bootstrap CIs, overall accuracy (macro over categories), call/no-call relevance detection, and a
paired bootstrap CI for each model's difference against the first model (the base).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from toolforge.config import EvalModelSpec, RunConfig
from toolforge.data.build import read_jsonl
from toolforge.formatting import parse_tool_calls, render_messages
from toolforge.scorer import NO_CALL_CATEGORIES, score
from toolforge.stats import bootstrap_ci, paired_bootstrap_diff


def bucket(rec: dict[str, Any]) -> str:
    src = "bfcl" if rec["source"].startswith("bfcl") else "glaive"
    return f"{src}/{rec['category']}"


def load_eval_records(files: list[str], limit_per_category: int | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)
    for f in files:
        for rec in read_jsonl(f):
            b = bucket(rec)
            if limit_per_category is not None and counts[b] >= limit_per_category:
                continue
            counts[b] += 1
            out.append(rec)
    return out


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _make_generator(cfg: RunConfig, spec: EvalModelSpec) -> Any:
    if spec.backend == "openai":
        from toolforge.generate import OpenAICompletionsGenerator

        if not spec.endpoint:
            raise ValueError(f"model {spec.name!r}: backend=openai needs an endpoint")
        return OpenAICompletionsGenerator(spec.endpoint, spec.model_id or "toolforge")
    from toolforge.generate import HFGenerator
    from toolforge.modeling import apply_adapters, load_model, load_tokenizer

    base = spec.base or cfg.model.name_or_path
    tok = load_tokenizer(cfg.model.name_or_path, cfg.model.revision)
    model = load_model(cfg.model, path=spec.base)
    model = apply_adapters(model, spec.adapters)
    print(f"[toolforge] eval model {spec.name}: base={base} adapters={spec.adapters}")
    return HFGenerator(model, tok)


def predict(cfg: RunConfig, spec: EvalModelSpec, records: list[dict[str, Any]]) -> dict[str, str]:
    """Generate (or resume) raw outputs for every record; returns id -> output."""
    ec = cfg.eval
    path = Path(ec.out_dir) / "predictions" / f"{slug(spec.name)}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["id"]] = row["output"]
    todo = [r for r in records if r["id"] not in done]
    if not todo:
        return done
    gen = _make_generator(cfg, spec)
    # sort by prompt length so batches pad little
    prompts = {r["id"]: render_messages(r["messages"], r["tools"], add_generation_prompt=True) for r in todo}
    todo.sort(key=lambda r: len(prompts[r["id"]]))
    with path.open("a", encoding="utf-8") as fh:
        for i in range(0, len(todo), ec.batch_size):
            batch = todo[i : i + ec.batch_size]
            outs = gen.generate([prompts[r["id"]] for r in batch], ec.max_new_tokens, temperature=0.0)
            for r, o in zip(batch, outs):
                done[r["id"]] = o
                fh.write(json.dumps({"id": r["id"], "output": o}, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[toolforge] {spec.name}: {len(done)}/{len(records)}", flush=True)
    return done


def score_records(records: list[dict[str, Any]], outputs: dict[str, str]) -> list[dict[str, Any]]:
    rows = []
    for rec in records:
        parsed = parse_tool_calls(outputs[rec["id"]])
        verdict = score(parsed, rec["gold"], rec["category"])
        rows.append(
            {
                "id": rec["id"],
                "bucket": bucket(rec),
                "correct": verdict.correct,
                "reason": verdict.reason,
                "should_call": rec["category"] not in NO_CALL_CATEGORIES,
                "did_call": parsed.attempted_call,
                "format": parsed.format,
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]], n_boot: int, seed: int) -> dict[str, Any]:
    by_bucket: dict[str, list[float]] = defaultdict(list)
    reasons: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        by_bucket[r["bucket"]].append(1.0 if r["correct"] else 0.0)
        if not r["correct"]:
            reasons[r["bucket"]][r["reason"].split(":")[0]] += 1
    cats = {b: bootstrap_ci(v, n_boot, seed=seed).__dict__ for b, v in sorted(by_bucket.items())}
    macro = sum(c["mean"] for c in cats.values()) / len(cats) if cats else float("nan")
    tp = sum(1 for r in rows if r["should_call"] and r["did_call"])
    fp = sum(1 for r in rows if not r["should_call"] and r["did_call"])
    fn = sum(1 for r in rows if r["should_call"] and not r["did_call"])
    tn = sum(1 for r in rows if not r["should_call"] and not r["did_call"])
    micro = bootstrap_ci([1.0 if r["correct"] else 0.0 for r in rows], n_boot, seed=seed)
    return {
        "n": len(rows),
        "categories": cats,
        "macro_accuracy": macro,
        "micro_accuracy": micro.__dict__,
        "relevance": {
            "call_recall": tp / (tp + fn) if tp + fn else float("nan"),
            "no_call_accuracy": tn / (tn + fp) if tn + fp else float("nan"),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        },
        "error_reasons": {k: dict(v) for k, v in reasons.items()},
    }


def run_eval(cfg: RunConfig, only: list[str] | None = None) -> dict[str, Any]:
    ec = cfg.eval
    records = load_eval_records(ec.files, ec.limit_per_category)
    out = Path(ec.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"n_examples": len(records), "models": {}, "paired_vs_first": {}}
    scored: dict[str, list[dict[str, Any]]] = {}
    for spec in ec.models:
        if only and spec.name not in only:
            # not regenerated this time: include it in the report if its predictions are complete
            outputs = _read_preds(out, spec)
        else:
            outputs = predict(cfg, spec, records)
        if not outputs or any(r["id"] not in outputs for r in records):
            print(f"[toolforge] {spec.name}: predictions incomplete, skipped in report")
            continue
        rows = score_records(records, outputs)
        scored[spec.name] = rows
        report["models"][spec.name] = summarize(rows, ec.n_bootstrap, ec.seed)
        (out / "scored").mkdir(exist_ok=True)
        with (out / "scored" / f"{slug(spec.name)}.jsonl").open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
    names = list(scored)
    if names:
        first = names[0]
        for name in names[1:]:
            report["paired_vs_first"][f"{name} - {first}"] = paired_bootstrap_diff(
                [1.0 if r["correct"] else 0.0 for r in scored[first]],
                [1.0 if r["correct"] else 0.0 for r in scored[name]],
                ec.n_bootstrap,
                seed=ec.seed,
            ).__dict__
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    (out / "RESULTS.md").write_text(render_markdown(report))
    print(render_markdown(report))
    return report


def _read_preds(out: Path, spec: EvalModelSpec) -> dict[str, str]:
    path = out / "predictions" / f"{slug(spec.name)}.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r["id"]: r["output"] for r in rows}


def render_markdown(report: dict[str, Any]) -> str:
    models = report["models"]
    if not models:
        return "No completed model evaluations.\n"
    names = list(models)
    buckets = sorted({b for m in models.values() for b in m["categories"]})
    head = "| Category | n | " + " | ".join(names) + " |"
    sep = "|---|---|" + "---|" * len(names)
    lines = [f"Accuracy %, 95% bootstrap CI in brackets. {report['n_examples']} examples.", "", head, sep]
    for b in buckets:
        n = next(m["categories"][b]["n"] for m in models.values() if b in m["categories"])
        cells = []
        for name in names:
            c = models[name]["categories"].get(b)
            cells.append(f"{c['mean'] * 100:.1f} [{c['low'] * 100:.1f}, {c['high'] * 100:.1f}]" if c else "-")
        lines.append(f"| {b} | {n} | " + " | ".join(cells) + " |")
    lines.append(
        "| **macro avg** | | " + " | ".join(f"**{models[n]['macro_accuracy'] * 100:.1f}**" for n in names) + " |"
    )
    lines.append(
        "| relevance: call recall | | "
        + " | ".join(f"{models[n]['relevance']['call_recall'] * 100:.1f}" for n in names)
        + " |"
    )
    lines.append(
        "| relevance: no-call acc. | | "
        + " | ".join(f"{models[n]['relevance']['no_call_accuracy'] * 100:.1f}" for n in names)
        + " |"
    )
    if report.get("paired_vs_first"):
        lines += ["", "Paired bootstrap difference in micro accuracy (percentage points, 95% CI):", ""]
        for k, v in report["paired_vs_first"].items():
            lines.append(f"- {k}: {v['mean'] * 100:+.1f} [{v['low'] * 100:+.1f}, {v['high'] * 100:+.1f}]")
    return "\n".join(lines) + "\n"
