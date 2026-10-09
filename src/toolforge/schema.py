"""Validation of tool (function) schemas and tool calls.

Used by the data stage to drop malformed training examples, and by the DPO pair builder to make
sure a "chosen" completion is a valid call. Schemas are JSON-Schema-like OpenAI function specs;
BFCL's Python-flavoured type names (``dict``, ``float``, ``tuple``, ``any``) are normalised first.
"""

from __future__ import annotations

import copy
import re
from typing import Any

JSON_TYPES = frozenset({"string", "integer", "number", "boolean", "array", "object", "null"})
TYPE_ALIASES = {
    "dict": "object",
    "float": "number",
    "int": "integer",
    "str": "string",
    "bool": "boolean",
    "list": "array",
    "tuple": "array",
}
NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]*$")


def normalize_type(t: Any) -> Any:
    """Map BFCL / Python type names onto JSON-Schema names (``any`` becomes ``None`` = untyped)."""
    if isinstance(t, str):
        if t == "any":
            return None
        return TYPE_ALIASES.get(t, t)
    if isinstance(t, list):
        return [normalize_type(x) for x in t]
    return t


def _normalize_node(node: Any) -> Any:
    if not isinstance(node, dict):
        return node
    out = dict(node)
    if "type" in out:
        t = normalize_type(out["type"])
        if t is None:
            del out["type"]
        else:
            out["type"] = t
    if isinstance(out.get("properties"), dict):
        out["properties"] = {k: _normalize_node(v) for k, v in out["properties"].items()}
    if isinstance(out.get("items"), dict):
        out["items"] = _normalize_node(out["items"])
    return out


def normalize_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Return a normalised deep copy of a function schema (unwrapping ``{"type": "function"}``)."""
    tool = copy.deepcopy(tool)
    if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
        tool = tool["function"]
    params = tool.get("parameters")
    if params is None or params == {}:
        params = {"type": "object", "properties": {}}
    tool["parameters"] = _normalize_node(params)
    if "properties" not in tool["parameters"]:
        tool["parameters"]["properties"] = {}
    return tool


def validate_tool(tool: Any) -> list[str]:
    """Return a list of problems with a (normalised) function schema; empty means valid."""
    errors: list[str] = []
    if not isinstance(tool, dict):
        return ["tool is not an object"]
    name = tool.get("name")
    if not isinstance(name, str) or not NAME_RE.match(name):
        errors.append(f"invalid tool name: {name!r}")
    if not isinstance(tool.get("description", ""), str):
        errors.append("description is not a string")
    params = tool.get("parameters")
    if not isinstance(params, dict):
        return errors + ["parameters is not an object"]
    if params.get("type") != "object":
        errors.append(f"parameters.type must be 'object', got {params.get('type')!r}")
    props = params.get("properties", {})
    if not isinstance(props, dict):
        return errors + ["parameters.properties is not an object"]
    for pname, spec in props.items():
        if not isinstance(spec, dict):
            errors.append(f"property {pname!r} is not an object")
            continue
        ptype = spec.get("type")
        types = ptype if isinstance(ptype, list) else [ptype]
        if ptype is not None and any(not isinstance(t, str) or t not in JSON_TYPES for t in types):
            errors.append(f"property {pname!r} has unknown type {ptype!r}")
        if "enum" in spec and not isinstance(spec["enum"], list):
            errors.append(f"property {pname!r} enum is not a list")
    required = params.get("required", [])
    if not isinstance(required, list):
        errors.append("required is not a list")
    else:
        for r in required:
            if r not in props:
                errors.append(f"required parameter {r!r} is not declared")
    return errors


def check_type(value: Any, expected: Any) -> bool:
    """Does ``value`` conform to a JSON-Schema type (or list of types)? ``None`` accepts anything.

    ``bool`` is not an integer/number here, and an integer is a valid ``number``.
    """
    if expected is None:
        return True
    if isinstance(expected, list):
        return any(check_type(value, t) for t in expected)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "null":
        return value is None
    return False


def validate_call(name: str, arguments: Any, tools: list[dict[str, Any]]) -> list[str]:
    """Check one call against the available (normalised) tools. Empty list means valid."""
    by_name = {t.get("name"): t for t in tools}
    if name not in by_name:
        return [f"unknown function {name!r}"]
    if not isinstance(arguments, dict):
        return ["arguments is not an object"]
    params = by_name[name].get("parameters") or {}
    props: dict[str, Any] = params.get("properties") or {}
    required = params.get("required") or []
    errors: list[str] = []
    if not isinstance(required, list):
        return ["tool schema has a non-list 'required'"]
    for req in required:
        if req not in arguments:
            errors.append(f"missing required argument {req!r}")
    for key, value in arguments.items():
        if key not in props:
            errors.append(f"unexpected argument {key!r}")
            continue
        spec = props[key]
        if not check_type(value, spec.get("type")):
            errors.append(f"argument {key!r} has wrong type (expected {spec.get('type')})")
        elif "enum" in spec and value not in spec["enum"]:
            errors.append(f"argument {key!r} not in enum")
    return errors
