"""Load Berkeley Function Calling Leaderboard (BFCL v4) categories as toolforge eval records.

Files are JSON-lines from the gorilla repository (``bfcl_eval/data``), pinned to a commit in the
data config. Only single-turn AST categories are used: simple_python, multiple, parallel and
irrelevance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from toolforge.schema import normalize_tool

# BFCL file stem -> toolforge category
CATEGORY_FILES = {
    "simple": "BFCL_v4_simple_python",
    "multiple": "BFCL_v4_multiple",
    "parallel": "BFCL_v4_parallel",
    "irrelevance": "BFCL_v4_irrelevance",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_category(raw_dir: Path, category: str) -> list[dict[str, Any]]:
    stem = CATEGORY_FILES[category]
    questions = _read_jsonl(raw_dir / f"{stem}.json")
    answers: dict[str, Any] = {}
    ans_path = raw_dir / f"ans_{stem}.json"
    if category != "irrelevance":
        answers = {a["id"]: a["ground_truth"] for a in _read_jsonl(ans_path)}
    records = []
    for q in questions:
        turns = q["question"][0]
        messages = [{"role": m["role"], "content": m["content"]} for m in turns]
        records.append(
            {
                "id": f"bfcl-{q['id']}",
                "source": "bfcl-v4",
                "category": category,
                "tools": [normalize_tool(f) for f in q["function"]],
                "messages": messages,
                "gold": answers.get(q["id"], []),
            }
        )
    return records
