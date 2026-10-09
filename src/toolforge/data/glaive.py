"""Parse glaive-function-calling-v2 into toolforge records.

Raw rows look like::

    {"system": "SYSTEM: You are a helpful assistant with access to the following functions. ...
               {json schema}\\n\\n{json schema}",
     "chat": "USER: ...\\n\\n\\nASSISTANT: <functioncall> {\\"name\\": ..., \\"arguments\\": '{...}'}
              <|endoftext|>\\n\\n\\nFUNCTION RESPONSE: {...}\\n\\n\\nASSISTANT: ..."}

Each conversation yields at most one *decision* example: the context up to the first assistant
decision and that decision as target (a call, or a text reply). Filler turns such as
"Sure, let me check." immediately followed by a call are collapsed into the call.
"""

from __future__ import annotations

import json
import re
from typing import Any

from toolforge.schema import normalize_tool

_TURN_RE = re.compile(r"(USER|ASSISTANT|FUNCTION RESPONSE):\s?")
_EOT_RE = re.compile(r"<\u200b?\|endoftext\|>")


class GlaiveParseError(ValueError):
    pass


def parse_tools(system: str) -> list[dict[str, Any]]:
    """Extract the JSON function schemas embedded in a glaive system prompt."""
    start = system.find("{")
    if start < 0:
        return []
    decoder = json.JSONDecoder()
    tools: list[dict[str, Any]] = []
    pos = start
    while pos < len(system):
        while pos < len(system) and system[pos] in " \n\t\r":
            pos += 1
        if pos >= len(system):
            break
        try:
            obj, end = decoder.raw_decode(system, pos)
        except json.JSONDecodeError as exc:
            raise GlaiveParseError(f"bad tool json: {exc}") from exc
        if not isinstance(obj, dict):
            raise GlaiveParseError("tool is not an object")
        tools.append(obj)
        pos = end
    return tools


def parse_functioncall(text: str) -> dict[str, Any]:
    """Parse glaive's ``<functioncall> {"name": ..., "arguments": '<json string>'}`` syntax."""
    body = text.split("<functioncall>", 1)[1].strip()
    m = re.match(r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*\'(.*)\'\s*\}\s*$', body, re.DOTALL)
    if m:
        name, raw_args = m.group(1), m.group(2)
    else:
        try:
            obj = json.loads(body)
        except json.JSONDecodeError as exc:
            raise GlaiveParseError(f"unparseable functioncall: {body[:80]}") from exc
        name, raw_args = obj.get("name"), obj.get("arguments", {})
        if isinstance(raw_args, dict):
            return {"name": name, "arguments": raw_args}
    try:
        args = json.loads(raw_args) if raw_args.strip() else {}
    except (json.JSONDecodeError, AttributeError) as exc:
        raise GlaiveParseError(f"bad arguments json: {str(raw_args)[:80]}") from exc
    if not isinstance(args, dict) or not isinstance(name, str):
        raise GlaiveParseError("arguments is not an object")
    return {"name": name, "arguments": args}


def split_turns(chat: str) -> list[tuple[str, str]]:
    """Split a glaive chat string into (speaker, text) turns."""
    parts = _TURN_RE.split(chat)
    turns: list[tuple[str, str]] = []
    # parts = [prefix, speaker, text, speaker, text, ...]
    for i in range(1, len(parts) - 1, 2):
        text = _EOT_RE.sub("", parts[i + 1]).strip()
        turns.append((parts[i], text))
    return turns


def to_record(row: dict[str, Any], idx: int) -> dict[str, Any] | None:
    """Convert one raw glaive row to a decision record, or ``None`` when it has no usable decision."""
    tools = [normalize_tool(t) for t in parse_tools(row.get("system", ""))]
    if not tools:
        return None
    turns = split_turns(row.get("chat", ""))
    if not turns or turns[0][0] != "USER":
        return None
    ever_calls = any(sp == "ASSISTANT" and "<functioncall>" in tx for sp, tx in turns)
    messages: list[dict[str, Any]] = []
    i = 0
    while i < len(turns):
        speaker, text = turns[i]
        if speaker == "USER":
            messages.append({"role": "user", "content": text})
            i += 1
            continue
        if speaker != "ASSISTANT":
            return None  # function response before any decision: malformed
        # collapse "filler text" immediately followed by a call
        if (
            "<functioncall>" not in text
            and i + 1 < len(turns)
            and turns[i + 1][0] == "ASSISTANT"
            and "<functioncall>" in turns[i + 1][1]
        ):
            i += 1
            text = turns[i][1]
        if "<functioncall>" in text:
            call = parse_functioncall(text)
            target: dict[str, Any] = {"type": "call", "calls": [call]}
            category = "simple" if len(tools) == 1 else "multiple"
        else:
            if not text:
                return None
            target = {"type": "text", "content": text}
            # a reply in a conversation that never calls a tool is an irrelevance case; a reply
            # that is later followed by a call (e.g. a clarifying question) is a "clarify" case.
            category = "clarify" if ever_calls else "irrelevance"
        return {
            "id": f"glaive-{idx}",
            "source": "glaive-function-calling-v2",
            "category": category,
            "tools": tools,
            "messages": messages,
            "target": target,
        }
    return None
