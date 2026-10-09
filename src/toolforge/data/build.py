"""`toolforge data`: download, parse, validate, dedup, split, decontaminate, subsample, card."""

from __future__ import annotations

import json
import random
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from toolforge.config import DataConfig
from toolforge.data import bfcl, glaive
from toolforge.data.clean import Decontaminator, dedup, validate_record
from toolforge.formatting import render_messages, render_target
from toolforge.scorer import gold_from_calls

GLAIVE_FILE = "glaive-function-calling-v2.json"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    tmp.replace(dest)


def fetch_raw(cfg: DataConfig) -> None:
    """Download pinned raw files unless already present (idempotent, resumable per file)."""
    raw = Path(cfg.raw_dir)
    glaive_path = raw / GLAIVE_FILE
    if not glaive_path.exists():
        url = f"https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2/resolve/{cfg.glaive_revision}/{GLAIVE_FILE}"
        print(f"downloading {url}")
        _download(url, glaive_path)
    base = f"https://raw.githubusercontent.com/ShishirPatil/gorilla/{cfg.bfcl_commit}/berkeley-function-call-leaderboard/bfcl_eval/data"
    for cat in cfg.bfcl_categories:
        stem = bfcl.CATEGORY_FILES[cat]
        for name, url in (
            (f"{stem}.json", f"{base}/{stem}.json"),
            (f"ans_{stem}.json", f"{base}/possible_answer/{stem}.json"),
        ):
            if cat == "irrelevance" and name.startswith("ans_"):
                continue
            dest = raw / "bfcl" / name
            if not dest.exists():
                print(f"downloading {url}")
                _download(url, dest)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _with_gold(rec: dict[str, Any]) -> dict[str, Any]:
    rec = dict(rec)
    if rec["target"]["type"] == "call":
        rec["gold"] = gold_from_calls(rec["target"]["calls"])
    else:
        rec["gold"] = []
    return rec


def _stratified_take(
    rng: random.Random, pools: dict[str, list[dict[str, Any]]], quotas: dict[str, int]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cat, n in quotas.items():
        pool = list(pools.get(cat, []))
        rng.shuffle(pool)
        out.extend(pool[:n])
    return out


def build(cfg: DataConfig) -> dict[str, Any]:
    """Run the full data stage and return its statistics (also written to ``stats.json``)."""
    fetch_raw(cfg)
    raw = Path(cfg.raw_dir)
    out = Path(cfg.out_dir)
    rng = random.Random(cfg.seed)
    stats: dict[str, Any] = {"config": cfg.model_dump()}

    rows = json.loads((raw / GLAIVE_FILE).read_text(encoding="utf-8"))
    stats["glaive_raw_rows"] = len(rows)
    parsed: list[dict[str, Any]] = []
    parse_fail = 0
    no_decision = 0
    for i, row in enumerate(rows):
        try:
            rec = glaive.to_record(row, i)
        except glaive.GlaiveParseError:
            parse_fail += 1
            continue
        if rec is None:
            no_decision += 1
            continue
        parsed.append(rec)
    stats["glaive_parse_errors"] = parse_fail
    stats["glaive_no_decision"] = no_decision

    valid: list[dict[str, Any]] = []
    invalid_reasons: Counter[str] = Counter()
    for rec in parsed:
        errs = validate_record(rec)
        if errs:
            invalid_reasons[errs[0].split("'")[0].strip()] += 1
            continue
        if (
            len(render_messages(rec["messages"], rec["tools"], True)) + len(render_target(rec["target"]))
            > cfg.max_chars
        ):
            invalid_reasons["too long"] += 1
            continue
        valid.append(rec)
    stats["schema_invalid"] = dict(invalid_reasons.most_common())
    stats["valid"] = len(valid)

    deduped, n_dup = dedup(valid)
    stats["duplicates_removed"] = n_dup

    # held-out glaive eval split (by category) before any training subsample
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for rec in deduped:
        by_cat.setdefault(rec["category"], []).append(rec)
    stats["category_counts_after_dedup"] = {k: len(v) for k, v in sorted(by_cat.items())}
    eval_glaive: list[dict[str, Any]] = []
    eval_ids: set[str] = set()
    for cat, n in cfg.glaive_eval_per_category.items():
        pool = list(by_cat.get(cat, []))
        rng.shuffle(pool)
        picked = pool[:n]
        eval_glaive.extend(_with_gold(r) for r in picked)
        eval_ids |= {r["id"] for r in picked}
    remaining = [r for r in deduped if r["id"] not in eval_ids]

    eval_bfcl: list[dict[str, Any]] = []
    for cat in cfg.bfcl_categories:
        recs = bfcl.load_category(raw / "bfcl", cat)
        n = cfg.bfcl_eval_per_category.get(cat, len(recs))
        idx = sorted(random.Random(cfg.seed + 1).sample(range(len(recs)), min(n, len(recs))))
        eval_bfcl.extend(recs[i] for i in idx)

    # decontaminate the training pool against *all* BFCL questions in the used categories (not
    # just the sampled ones) and the glaive held-out split
    all_bfcl = [r for cat in cfg.bfcl_categories for r in bfcl.load_category(raw / "bfcl", cat)]
    deco = Decontaminator(eval_glaive + all_bfcl, n=cfg.decontam_ngram)
    clean_pool, n_contam = deco.filter(remaining)
    stats["decontaminated_removed"] = n_contam

    pools: dict[str, list[dict[str, Any]]] = {}
    for rec in clean_pool:
        pools.setdefault(rec["category"], []).append(rec)
    train = _stratified_take(rng, pools, cfg.train_per_category)
    rng.shuffle(train)
    train = [_with_gold(r) for r in train]
    stats["train_size"] = len(train)
    stats["train_categories"] = dict(Counter(r["category"] for r in train))
    stats["eval_glaive_size"] = len(eval_glaive)
    stats["eval_glaive_categories"] = dict(Counter(r["category"] for r in eval_glaive))
    stats["eval_bfcl_size"] = len(eval_bfcl)
    stats["eval_bfcl_categories"] = dict(Counter(r["category"] for r in eval_bfcl))
    stats["clean_pool_size"] = len(clean_pool)

    # DPO prompt pool: disjoint from the SFT subset
    train_ids = {r["id"] for r in train}
    dpo_pool = [r for r in clean_pool if r["id"] not in train_ids and r["category"] != "clarify"]
    rng.shuffle(dpo_pool)
    dpo_prompts = [_with_gold(r) for r in dpo_pool[: cfg.dpo_prompts]]
    stats["dpo_prompts"] = len(dpo_prompts)

    _write_jsonl(out / "train.jsonl", train)
    _write_jsonl(out / "dpo_prompts.jsonl", dpo_prompts)
    _write_jsonl(out / "eval_glaive.jsonl", eval_glaive)
    _write_jsonl(out / "eval_bfcl.jsonl", eval_bfcl)
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    write_data_card(stats, Path(cfg.card_path))
    return stats


def write_data_card(stats: dict[str, Any], path: Path) -> None:
    cfg = stats["config"]
    lines = [
        "# Data card: toolforge function-calling SFT set",
        "",
        "Generated by `toolforge data`. Numbers below are from the run that produced the committed",
        "`data/processed/stats.json`; re-running with the same config reproduces them.",
        "",
        "## Sources",
        "",
        f"- **Training**: [glaiveai/glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) "
        f"(Apache-2.0), revision `{cfg['glaive_revision']}`. {stats['glaive_raw_rows']:,} raw conversations.",
        f"- **Evaluation**: a held-out glaive split plus [BFCL v4](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) "
        f"(Apache-2.0) categories {', '.join(cfg['bfcl_categories'])}, gorilla commit `{cfg['bfcl_commit']}`.",
        "",
        "## Processing",
        "",
        "1. **Parse** each conversation into a *decision example*: tools + first user turn -> first assistant decision "
        '(a tool call, or a text reply). Filler like "Sure, let me check" directly followed by a call is collapsed into the call.',
        f"   Parse errors: {stats['glaive_parse_errors']:,}; conversations without a usable decision: {stats['glaive_no_decision']:,}.",
        "2. **Schema validation**: every tool must be a valid object schema (name, typed properties, declared required params); "
        "every gold call must name an available tool, pass required/unknown-argument checks, and match declared types and enums. "
        f"Rendered length capped at {cfg['max_chars']:,} characters.",
        f"   Kept {stats['valid']:,}; dropped: {json.dumps(stats['schema_invalid'])}.",
        f"3. **Dedup** on normalised user text + tool-name set: {stats['duplicates_removed']:,} removed.",
        f"   Categories after dedup: {json.dumps(stats['category_counts_after_dedup'])}.",
        f"4. **Held-out glaive eval split** taken before training sampling: {json.dumps(stats['eval_glaive_categories'])}.",
        f"5. **Decontamination** against the glaive eval split and *all* questions of the used BFCL categories: exact normalised "
        f"match or any shared {cfg['decontam_ngram']}-word n-gram. Removed {stats['decontaminated_removed']:,}.",
        f"6. **Stratified subsample** for CPU training: {json.dumps(stats['train_categories'])} (total {stats['train_size']:,}); "
        f"{stats['dpo_prompts']:,} further disjoint prompts reserved for DPO pair mining.",
        "",
        "## Categories",
        "",
        "- `simple`: one tool available, one call expected.",
        "- `multiple`: several tools available, exactly one call expected.",
        "- `irrelevance`: tools available but none fits; the right answer is a text reply and no call.",
        "- `clarify` (train only): the assistant asks for missing information before calling.",
        "",
        "## Known issues and limitations",
        "",
        "- glaive v2 is synthetic (generated by an LLM); tools are simple and repetitive (many `calculate_*`, `get_*`), so "
        "the held-out glaive split is in-distribution. BFCL is the out-of-distribution check.",
        "- glaive has no parallel calls, so `parallel` is only evaluated (BFCL), never trained.",
        "- Gold arguments are a single acceptable value on the glaive split (BFCL provides several); this makes the glaive "
        "split stricter than BFCL for free-text arguments.",
        "- Text replies in glaive are not scored for content, only for the absence of a call.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
