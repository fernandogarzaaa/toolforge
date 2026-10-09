"""Build DPO preference pairs from the SFT model's own mistakes.

For each training prompt we have the gold target and one or more completions sampled from the SFT
model. ``chosen`` is the gold completion; ``rejected`` is a completion the AST scorer marks wrong.
Well-formed-but-wrong calls (wrong value, wrong function, missing argument) are preferred as
rejections because they are the most informative contrast. When the model got a prompt right and
``synthetic=True``, a controlled corruption of the gold call is used instead (dropped required
argument, perturbed value, wrong function, spurious or missing call). Every pair is checked: the
chosen side must score correct and the rejected side must score incorrect.
"""

from __future__ import annotations

import copy
import random
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from toolforge.formatting import IM_END, ToolCall, parse_tool_calls, render_messages, render_target, render_tool_calls
from toolforge.scorer import score

# lower rank = more informative rejection
REASON_RANK = {
    "value_mismatch": 0,
    "missing_param": 1,
    "unexpected_param": 2,
    "wrong_function": 3,
    "parallel_mismatch": 4,
    "wrong_call_count": 5,
    "spurious_call": 6,
    "no_call": 7,
    "malformed_call": 8,
}
NO_CALL_TEXT = "I'm sorry, but I can't help with that using the available functions."


@dataclass(frozen=True)
class Pair:
    id: str
    prompt: str
    chosen: str
    rejected: str
    source: str  # "model:<reason>" or "synthetic:<kind>"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def clean_completion(text: str) -> str:
    """Strip the end-of-turn marker and surrounding whitespace from a generated completion."""
    return text.split(IM_END)[0].strip()


def _reason_key(reason: str) -> str:
    return reason.split(":", 1)[0]


def _is_correct(completion: str, example: dict[str, Any]) -> tuple[bool, str]:
    verdict = score(parse_tool_calls(completion), example["gold"], example["category"])
    return verdict.correct, verdict.reason


def _perturb_value(value: Any) -> Any:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return round(value * 2 + 1.0, 3)
    if isinstance(value, str):
        return value + " extra" if value else "unknown"
    if isinstance(value, list):
        return value[:-1] if value else ["unknown"]
    if isinstance(value, dict):
        return {}
    return "unknown"


def synthetic_negatives(example: dict[str, Any]) -> list[tuple[str, str]]:
    """Candidate (kind, completion) corruptions of the gold target, most informative first."""
    out: list[tuple[str, str]] = []
    target = example["target"]
    tools = example.get("tools") or []
    if target["type"] != "call":
        if tools:
            first = tools[0]
            props = first.get("parameters", {}).get("properties", {})
            args = {k: "unknown" for k in first.get("parameters", {}).get("required", []) if k in props}
            out.append(("spurious_call", render_tool_calls([ToolCall(first["name"], args)])))
        return out
    calls = [ToolCall(c["name"], dict(c["arguments"])) for c in target["calls"]]
    first = calls[0]
    tool = next((t for t in tools if t.get("name") == first.name), None)
    required = (tool or {}).get("parameters", {}).get("required", []) or []
    present_required = [r for r in required if r in first.arguments]
    if present_required:
        args = dict(first.arguments)
        del args[present_required[0]]
        out.append(("drop_required", render_tool_calls([ToolCall(first.name, args)] + calls[1:])))
    if first.arguments:
        key = sorted(first.arguments)[0]
        args = copy.deepcopy(first.arguments)
        args[key] = _perturb_value(args[key])
        out.append(("perturb_value", render_tool_calls([ToolCall(first.name, args)] + calls[1:])))
    others = [t["name"] for t in tools if t.get("name") != first.name]
    if others:
        out.append(("wrong_function", render_tool_calls([ToolCall(others[0], first.arguments)] + calls[1:])))
    out.append(("no_call", NO_CALL_TEXT))
    return out


def build_pairs(
    examples: Iterable[dict[str, Any]],
    generations: dict[str, list[str]],
    *,
    synthetic: bool = True,
    max_pairs_per_prompt: int = 1,
    seed: int = 0,
) -> list[Pair]:
    """Build verified DPO pairs. ``generations`` maps example id to raw SFT-model completions."""
    if max_pairs_per_prompt < 1:
        raise ValueError("max_pairs_per_prompt must be >= 1")
    rng = random.Random(seed)
    pairs: list[Pair] = []
    for ex in examples:
        chosen = render_target(ex["target"])
        ok, _ = _is_correct(chosen, ex)
        if not ok:
            continue  # gold does not pass its own scorer: never train towards it
        prompt = render_messages(ex["messages"], ex.get("tools"), add_generation_prompt=True)
        seen: set[str] = {chosen}
        candidates: list[tuple[int, int, str, str]] = []
        for idx, raw in enumerate(generations.get(ex["id"], [])):
            completion = clean_completion(raw)
            if completion in seen:
                continue
            seen.add(completion)
            correct, reason = _is_correct(completion, ex)
            if not correct:
                rank = REASON_RANK.get(_reason_key(reason), len(REASON_RANK))
                candidates.append((rank, idx, completion, f"model:{reason}"))
        candidates.sort(key=lambda c: (c[0], c[1]))
        chosen_rejections = [(c[2], c[3]) for c in candidates[:max_pairs_per_prompt]]
        if synthetic and len(chosen_rejections) < max_pairs_per_prompt:
            synth = [(kind, comp) for kind, comp in synthetic_negatives(ex) if comp not in seen]
            if len(synth) > 1:
                head, tail = synth[:1], synth[1:]
                rng.shuffle(tail)
                synth = head + tail
            for kind, comp in synth:
                if len(chosen_rejections) >= max_pairs_per_prompt:
                    break
                if not _is_correct(comp, ex)[0]:
                    chosen_rejections.append((comp, f"synthetic:{kind}"))
                    seen.add(comp)
        for rejected, source in chosen_rejections:
            pairs.append(Pair(ex["id"], prompt, chosen, rejected, source))
    return pairs
