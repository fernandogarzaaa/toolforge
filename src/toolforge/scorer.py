"""BFCL-style AST scorer.

A gold answer is a list of *possible answers*, one per expected call, in BFCL's format::

    [{"func_name": {"param": [acceptable_value, ...], ...}}]

``""`` in a parameter's acceptable list means the parameter may be omitted. A model call matches
when the function name is right, every required parameter is present, no parameter outside the
possible answer is given, and every value is acceptable after BFCL's normalisation (strings are
compared case-insensitively ignoring spaces and common punctuation, ints are accepted for floats,
lists are compared element-wise, dicts recursively).

Categories:

* ``simple`` / ``multiple``: exactly one call, matching the single possible answer.
* ``parallel``: the set of calls must match the possible answers one-to-one, in any order.
* ``irrelevance``: correct iff the model makes no call attempt at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from toolforge.formatting import ParseResult, ToolCall

CALL_CATEGORIES = frozenset({"simple", "multiple", "parallel"})
NO_CALL_CATEGORIES = frozenset({"irrelevance"})
_STRIP_RE = re.compile(r"[ ,./\-_*^]")


@dataclass(frozen=True)
class Verdict:
    correct: bool
    reason: str = "ok"


def normalize_string(s: str) -> str:
    """BFCL string normalisation: lower-case, drop spaces and ``,./-_*^``."""
    return _STRIP_RE.sub("", s).lower()


def value_matches(value: Any, acceptable: Any) -> bool:
    """Does a single model value equal one acceptable gold value (with BFCL normalisation)?"""
    if isinstance(acceptable, bool) or isinstance(value, bool):
        return isinstance(value, bool) and isinstance(acceptable, bool) and value == acceptable
    if isinstance(acceptable, (int, float)):
        if not isinstance(value, (int, float)):
            return False
        if isinstance(acceptable, int) and isinstance(value, float) and not value.is_integer():
            return False
        return abs(float(value) - float(acceptable)) <= 1e-9 * max(1.0, abs(float(acceptable)))
    if isinstance(acceptable, str):
        return isinstance(value, str) and normalize_string(value) == normalize_string(acceptable)
    if isinstance(acceptable, list):
        if not isinstance(value, list) or len(value) != len(acceptable):
            return False
        return all(value_matches(v, a) for v, a in zip(value, acceptable))
    if isinstance(acceptable, dict):
        # nested possible-answer dict: {key: [acceptable, ...]}
        if not isinstance(value, dict):
            return False
        return params_match(value, acceptable).correct
    if acceptable is None:
        return value is None
    return bool(value == acceptable)


def params_match(arguments: dict[str, Any], possible: dict[str, list[Any]]) -> Verdict:
    """Match a call's arguments against a possible-answer parameter map."""
    for key in arguments:
        if key not in possible:
            return Verdict(False, f"unexpected_param:{key}")
    for key, options in possible.items():
        options = options if isinstance(options, list) else [options]
        if key not in arguments:
            if "" in options:
                continue
            return Verdict(False, f"missing_param:{key}")
        value = arguments[key]
        if not any(value_matches(value, opt) for opt in options if not (opt == "" and value != "")):
            return Verdict(False, f"value_mismatch:{key}")
    return Verdict(True)


def call_matches(call: ToolCall, possible_call: dict[str, dict[str, list[Any]]]) -> Verdict:
    """Match one model call against one possible answer ``{func_name: {param: [values]}}``."""
    if len(possible_call) != 1:
        raise ValueError("a possible answer must name exactly one function")
    ((gold_name, gold_params),) = possible_call.items()
    if call.name != gold_name:
        return Verdict(False, "wrong_function")
    return params_match(call.arguments, gold_params)


def _perfect_matching(calls: list[ToolCall], gold: list[dict[str, Any]]) -> bool:
    """Is there a one-to-one assignment of calls to gold answers (order-insensitive)?"""
    used = [False] * len(gold)

    def assign(i: int) -> bool:
        if i == len(calls):
            return True
        for j, g in enumerate(gold):
            if not used[j] and call_matches(calls[i], g).correct:
                used[j] = True
                if assign(i + 1):
                    return True
                used[j] = False
        return False

    return assign(0)


def score(parsed: ParseResult, gold: list[dict[str, Any]], category: str) -> Verdict:
    """Score one parsed model output against its gold answer for a BFCL-style category."""
    if category in NO_CALL_CATEGORIES:
        return Verdict(False, "spurious_call") if parsed.attempted_call else Verdict(True)
    if category not in CALL_CATEGORIES:
        raise ValueError(f"unknown category: {category!r}")
    if not parsed.attempted_call:
        return Verdict(False, "no_call")
    if parsed.errors:
        return Verdict(False, "malformed_call")
    if len(parsed.calls) != len(gold):
        return Verdict(False, "wrong_call_count")
    if category == "parallel":
        return Verdict(True) if _perfect_matching(parsed.calls, gold) else Verdict(False, "parallel_mismatch")
    return call_matches(parsed.calls[0], gold[0])


def gold_from_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn concrete gold calls (``{"name", "arguments"}``) into single-option possible answers."""
    return [{c["name"]: {k: [v] for k, v in c["arguments"].items()}} for c in calls]
