"""Prompt rendering and tool-call parsing.

toolforge renders conversations itself instead of relying on ``tokenizer.apply_chat_template``.
That keeps training, DPO pair building and evaluation byte-identical and testable without a
tokenizer. The default template reproduces the Qwen2.5 "hermes-style" tool-calling chat template
(``<tools>`` block in the system turn, ``<tool_call>`` JSON blocks in assistant turns); a parity
test checks it against the real tokenizer template when the model is available locally.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

TOOLS_PREAMBLE = (
    "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
)
TOOLS_POSTAMBLE = (
    "\n</tools>\n\nFor each function call, return a json object with function name and "
    "arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
)

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.DOTALL)


@dataclass(frozen=True)
class ToolCall:
    """A single function call: a name plus a JSON-object of arguments."""

    name: str
    arguments: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps({"name": self.name, "arguments": self.arguments}, ensure_ascii=False)


@dataclass
class ParseResult:
    """What a model output contained.

    ``calls`` are the well-formed calls; ``errors`` lists blocks that looked like a call but did
    not parse. ``attempted_call`` is true if the model tried to call anything at all, which is
    what relevance / irrelevance detection is scored on.
    """

    calls: list[ToolCall] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    content: str = ""
    format: str = "none"  # "tagged" | "bare_json" | "none"

    @property
    def attempted_call(self) -> bool:
        return bool(self.calls) or bool(self.errors)


def tojson(obj: Any) -> str:
    """JSON exactly as the HF chat-template ``tojson`` filter renders it."""
    return json.dumps(obj, ensure_ascii=False)


def as_openai_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Wrap a bare function schema as ``{"type": "function", "function": ...}``."""
    if tool.get("type") == "function" and "function" in tool:
        return tool
    return {"type": "function", "function": tool}


def render_system(tools: list[dict[str, Any]] | None, system: str | None = None) -> str:
    """Render the system turn (including the trailing ``<|im_end|>\\n``)."""
    text = system if system is not None else DEFAULT_SYSTEM
    if tools:
        text += TOOLS_PREAMBLE
        for tool in tools:
            text += "\n" + tojson(as_openai_tool(tool))
        text += TOOLS_POSTAMBLE
    return f"{IM_START}system\n{text}{IM_END}\n"


def render_tool_calls(calls: list[ToolCall]) -> str:
    """Render calls as the assistant would emit them (no turn markers)."""
    parts = []
    for call in calls:
        parts.append(
            '<tool_call>\n{"name": "' + call.name + '", "arguments": ' + tojson(call.arguments) + "}\n</tool_call>"
        )
    return "\n".join(parts)


def render_assistant_body(content: str | None, calls: list[ToolCall] | None) -> str:
    """Body of an assistant turn, i.e. the text a model should generate (without ``<|im_end|>``)."""
    body = content or ""
    if calls:
        rendered = render_tool_calls(calls)
        body = (body + "\n" + rendered) if body else rendered
    return body


def render_messages(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    add_generation_prompt: bool = False,
) -> str:
    """Render a conversation to the model's raw prompt string.

    ``messages`` use OpenAI-style roles: system, user, assistant (with optional ``tool_calls`` as a
    list of ``{"name", "arguments"}``) and tool.
    """
    out: list[str] = []
    msgs = list(messages)
    system = None
    if msgs and msgs[0]["role"] == "system":
        system = msgs.pop(0)["content"]
    out.append(render_system(tools, system))
    i = 0
    while i < len(msgs):
        msg = msgs[i]
        role = msg["role"]
        if role == "user":
            out.append(f"{IM_START}user\n{msg['content']}{IM_END}\n")
        elif role == "assistant":
            calls = [ToolCall(c["name"], c["arguments"]) for c in msg.get("tool_calls") or []]
            out.append(f"{IM_START}assistant\n{render_assistant_body(msg.get('content'), calls)}{IM_END}\n")
        elif role == "tool":
            # consecutive tool messages share one user turn
            chunk = f"{IM_START}user"
            while i < len(msgs) and msgs[i]["role"] == "tool":
                chunk += f"\n<tool_response>\n{msgs[i]['content']}\n</tool_response>"
                i += 1
            out.append(chunk + f"{IM_END}\n")
            continue
        else:
            raise ValueError(f"unknown role: {role!r}")
        i += 1
    if add_generation_prompt:
        out.append(f"{IM_START}assistant\n")
    return "".join(out)


def render_target(target: dict[str, Any]) -> str:
    """Render a gold target (``{"type": "call", "calls": [...]}`` or ``{"type": "text"}``)."""
    if target["type"] == "call":
        calls = [ToolCall(c["name"], c["arguments"]) for c in target["calls"]]
        return render_tool_calls(calls)
    if target["type"] == "text":
        return str(target["content"])
    raise ValueError(f"unknown target type: {target['type']!r}")


def _coerce_call(obj: Any) -> ToolCall:
    if not isinstance(obj, dict):
        raise ValueError("call is not a JSON object")
    name = obj.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("call has no name")
    args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str):
        args = json.loads(args) if args.strip() else {}
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ValueError("arguments is not a JSON object")
    return ToolCall(name, args)


def parse_tool_calls(text: str) -> ParseResult:
    """Extract tool calls from raw model output.

    Accepts ``<tool_call>`` blocks (a missing closing tag on the last block is tolerated, since
    generation often stops at ``<|im_end|>``), and as a fallback an output that is *entirely* a JSON
    object (or list of objects) with a ``name`` key.
    """
    text = text.replace(IM_END, "")
    result = ParseResult()
    blocks = _TOOL_CALL_RE.findall(text)
    if blocks:
        result.format = "tagged"
        for raw in blocks:
            raw = raw.strip()
            try:
                result.calls.append(_coerce_call(json.loads(raw)))
            except (ValueError, TypeError) as exc:
                result.errors.append(f"{exc}: {raw[:80]}")
        result.content = _TOOL_CALL_RE.sub("", text).strip()
        return result
    stripped = text.strip()
    if stripped[:1] in "{[":
        try:
            obj = json.loads(stripped)
        except ValueError:
            obj = None
        items = obj if isinstance(obj, list) else [obj]
        if obj is not None and items and all(isinstance(x, dict) and "name" in x for x in items):
            result.format = "bare_json"
            for item in items:
                try:
                    result.calls.append(_coerce_call(item))
                except (ValueError, TypeError) as exc:
                    result.errors.append(str(exc))
            return result
    result.content = stripped
    return result
