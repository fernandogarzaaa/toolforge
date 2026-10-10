"""Paired analysis of results/scored/*.jsonl -> results/paired_analysis.json (+ markdown on stdout).

Run from the repo root after `toolforge eval`. Compares base / sft / sft+dpo pairwise on the same items.

Uses the same stats as `toolforge eval` (toolforge.stats, 2,000 resamples, seed 0).
"""

from __future__ import annotations

import json
from pathlib import Path

from toolforge.stats import bootstrap_ci, paired_bootstrap_diff

N_BOOT, SEED = 2000, 0
names = {"base": "base", "sft": "sft", "sft+dpo": "sft_dpo"}
rows = {
    k: {json.loads(line)["id"]: json.loads(line) for line in Path(f"results/scored/{v}.jsonl").read_text().splitlines()}
    for k, v in names.items()
}
ids = list(rows["base"])  # file order = eval record order, so CIs match `toolforge eval` exactly
assert all(sorted(r) == sorted(ids) for r in rows.values()), "scored files cover different ids"
corr = {k: [1.0 if rows[k][i]["correct"] else 0.0 for i in ids] for k in names}
buckets = sorted({rows["base"][i]["bucket"] for i in ids})

out: dict = {"n": len(ids), "micro": {}, "pairs": {}}
for k in names:
    out["micro"][k] = bootstrap_ci(corr[k], N_BOOT, seed=SEED).__dict__
for a, b in [("base", "sft"), ("base", "sft+dpo"), ("sft", "sft+dpo")]:
    d = paired_bootstrap_diff(corr[a], corr[b], N_BOOT, seed=SEED).__dict__
    wins = sum(1 for x, y in zip(corr[a], corr[b]) if y > x)
    losses = sum(1 for x, y in zip(corr[a], corr[b]) if y < x)
    per_bucket = {}
    for bk in buckets:
        idx = [j for j, i in enumerate(ids) if rows["base"][i]["bucket"] == bk]
        per_bucket[bk] = paired_bootstrap_diff(
            [corr[a][j] for j in idx], [corr[b][j] for j in idx], N_BOOT, seed=SEED
        ).__dict__
    out["pairs"][f"{b} - {a}"] = {"overall": d, "fixed": wins, "broken": losses, "by_category": per_bucket}

Path("results/paired_analysis.json").write_text(json.dumps(out, indent=2) + "\n")


def pp(x: float) -> str:
    return f"{x * 100:+.1f}"


lines = [
    "| Comparison | Δ micro acc. (pp) [95% CI] | fixed | broken | "
    + " | ".join(b.split("/")[-1] for b in buckets)
    + " |"
]
lines.append("|---|---|---|---|" + "---|" * len(buckets))
for k, v in out["pairs"].items():
    o = v["overall"]
    cells = [
        f"{pp(v['by_category'][b]['mean'])} [{pp(v['by_category'][b]['low'])}, {pp(v['by_category'][b]['high'])}]"
        for b in buckets
    ]
    lines.append(
        f"| {k} | {pp(o['mean'])} [{pp(o['low'])}, {pp(o['high'])}] | {v['fixed']} | {v['broken']} | "
        + " | ".join(cells)
        + " |"
    )
micro = " · ".join(
    f"{k} {m['mean'] * 100:.1f} [{m['low'] * 100:.1f}, {m['high'] * 100:.1f}]" for k, m in out["micro"].items()
)
print("Micro accuracy % [95% CI]: " + micro + "\n")
print("\n".join(lines))
