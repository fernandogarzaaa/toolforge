"""Pure data-cleaning steps: record validation, deduplication and decontamination."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from typing import Any

from toolforge.schema import validate_call, validate_tool

_WORD_RE = re.compile(r"[a-z0-9]+")


def validate_record(rec: dict[str, Any]) -> list[str]:
    """All problems with a training record; empty means it may be used for training."""
    errors: list[str] = []
    tools = rec.get("tools")
    if not isinstance(tools, list) or not tools:
        return ["no tools"]
    names = [t.get("name") if isinstance(t, dict) else None for t in tools]
    if len(set(names)) != len(names):
        errors.append("duplicate tool names")
    for t in tools:
        errors.extend(validate_tool(t))
    if errors:
        return errors  # calls cannot be checked against broken tool schemas
    messages = rec.get("messages")
    if not isinstance(messages, list) or not messages:
        errors.append("no messages")
    elif messages[-1].get("role") not in ("user", "tool"):
        errors.append("context must end with a user or tool turn")
    elif any(not isinstance(m.get("content"), str) or not m["content"].strip() for m in messages):
        errors.append("empty message")
    target = rec.get("target")
    if not isinstance(target, dict):
        return errors + ["no target"]
    if target.get("type") == "call":
        calls = target.get("calls")
        if not isinstance(calls, list) or not calls:
            errors.append("call target without calls")
        else:
            for c in calls:
                errors.extend(validate_call(c.get("name", ""), c.get("arguments"), tools))
    elif target.get("type") == "text":
        content = target.get("content")
        if not isinstance(content, str) or not content.strip():
            errors.append("empty text target")
        elif "<functioncall>" in content or "<tool_call>" in content:
            errors.append("text target contains a call marker")
    else:
        errors.append(f"unknown target type {target.get('type')!r}")
    return errors


def normalize_text(text: str) -> str:
    """Lower-case alphanumeric tokens joined by single spaces (for dedup / overlap checks)."""
    return " ".join(_WORD_RE.findall(text.lower()))


def user_text(rec: dict[str, Any]) -> str:
    return " ".join(m["content"] for m in rec.get("messages", []) if m.get("role") == "user")


def dedup_key(rec: dict[str, Any]) -> str:
    """Records with the same normalised user text and the same tool names are duplicates."""
    tool_names = sorted(t.get("name", "") for t in rec.get("tools", []))
    payload = json.dumps([normalize_text(user_text(rec)), tool_names])
    return hashlib.sha1(payload.encode()).hexdigest()


def dedup(records: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Keep the first record per :func:`dedup_key`; returns (kept, n_dropped)."""
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    dropped = 0
    for rec in records:
        key = dedup_key(rec)
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        kept.append(rec)
    return kept, dropped


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    toks = normalize_text(text).split()
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


class Decontaminator:
    """Flags training records that overlap the evaluation sets.

    A record is contaminated if its normalised user text equals an eval query, or shares any
    word ``n``-gram with an eval query (queries shorter than ``n`` words only match exactly).
    """

    def __init__(self, eval_records: Iterable[dict[str, Any]], n: int = 8) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n
        self.exact: set[str] = set()
        self.grams: set[tuple[str, ...]] = set()
        for rec in eval_records:
            text = user_text(rec)
            self.exact.add(normalize_text(text))
            self.grams |= ngrams(text, n)

    def is_contaminated(self, rec: dict[str, Any]) -> bool:
        text = user_text(rec)
        if normalize_text(text) in self.exact:
            return True
        return not self.grams.isdisjoint(ngrams(text, self.n))

    def filter(self, records: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
        kept: list[dict[str, Any]] = []
        dropped = 0
        for rec in records:
            if self.is_contaminated(rec):
                dropped += 1
            else:
                kept.append(rec)
        return kept, dropped
