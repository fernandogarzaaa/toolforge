import pytest

from toolforge.schema import check_type, normalize_tool, normalize_type, validate_call, validate_tool


def tool(**props):
    return {
        "name": "f",
        "description": "d",
        "parameters": {
            "type": "object",
            "properties": props,
            "required": [k for k in props if not k.startswith("opt")],
        },
    }


# ------------------------------------------------------------- normalisation


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("dict", "object"),
        ("float", "number"),
        ("int", "integer"),
        ("str", "string"),
        ("bool", "boolean"),
        ("list", "array"),
        ("tuple", "array"),
        ("any", None),
        ("string", "string"),
        (["float", "null"], ["number", "null"]),
        (5, 5),
    ],
)
def test_normalize_type(given, expected):
    assert normalize_type(given) == expected


def test_normalize_tool_bfcl_style():
    raw = {
        "name": "math.hypot",
        "description": "d",
        "parameters": {
            "type": "dict",
            "properties": {
                "x": {"type": "float"},
                "pts": {"type": "array", "items": {"type": "tuple"}},
                "meta": {"type": "dict", "properties": {"k": {"type": "int"}}},
                "anything": {"type": "any"},
            },
            "required": ["x"],
        },
    }
    out = normalize_tool(raw)
    params = out["parameters"]
    assert params["type"] == "object"
    assert params["properties"]["x"] == {"type": "number"}
    assert params["properties"]["pts"]["items"] == {"type": "array"}
    assert params["properties"]["meta"]["properties"]["k"] == {"type": "integer"}
    assert "type" not in params["properties"]["anything"]
    assert raw["parameters"]["type"] == "dict"  # input untouched


def test_normalize_tool_unwraps_openai_wrapper_and_fills_empty_parameters():
    out = normalize_tool({"type": "function", "function": {"name": "f", "parameters": {}}})
    assert out == {"name": "f", "parameters": {"type": "object", "properties": {}}}
    out = normalize_tool({"name": "g"})
    assert out["parameters"] == {"type": "object", "properties": {}}
    out = normalize_tool({"name": "h", "parameters": {"type": "object"}})
    assert out["parameters"]["properties"] == {}


def test_normalize_tool_keeps_non_dict_nodes():
    out = normalize_tool({"name": "f", "parameters": {"type": "object", "properties": {"a": "weird"}}})
    assert out["parameters"]["properties"]["a"] == "weird"


# ------------------------------------------------------------- validate_tool


def test_valid_tool_has_no_errors():
    assert validate_tool(tool(a={"type": "string"}, opt={"type": ["integer", "null"], "enum": [1, None]})) == []
    assert validate_tool({"name": "x.y-z_1", "parameters": {"type": "object", "properties": {}}}) == []
    # untyped property is allowed
    assert validate_tool({"name": "f", "parameters": {"type": "object", "properties": {"a": {}}}}) == []


@pytest.mark.parametrize(
    ("bad", "fragment"),
    [
        ("not a dict", "tool is not an object"),
        ({"name": "1abc", "parameters": {"type": "object", "properties": {}}}, "invalid tool name"),
        ({"name": "has space", "parameters": {"type": "object", "properties": {}}}, "invalid tool name"),
        ({"name": None, "parameters": {"type": "object", "properties": {}}}, "invalid tool name"),
        ({"name": "f", "description": 3, "parameters": {"type": "object", "properties": {}}}, "description"),
        ({"name": "f", "parameters": []}, "parameters is not an object"),
        ({"name": "f", "parameters": {"type": "array", "properties": {}}}, "parameters.type must be 'object'"),
        ({"name": "f", "parameters": {"type": "object", "properties": []}}, "properties is not an object"),
        ({"name": "f", "parameters": {"type": "object", "properties": {"a": 1}}}, "property 'a' is not an object"),
        ({"name": "f", "parameters": {"type": "object", "properties": {"a": {"type": "str"}}}}, "unknown type"),
        (
            {"name": "f", "parameters": {"type": "object", "properties": {"a": {"type": ["string", "date"]}}}},
            "unknown type",
        ),
        (
            {"name": "f", "parameters": {"type": "object", "properties": {"a": {"type": "string", "enum": "x"}}}},
            "enum is not a list",
        ),
        ({"name": "f", "parameters": {"type": "object", "properties": {}, "required": "a"}}, "required is not a list"),
        (
            {"name": "f", "parameters": {"type": "object", "properties": {}, "required": ["a"]}},
            "required parameter 'a' is not declared",
        ),
    ],
)
def test_invalid_tools(bad, fragment):
    errors = validate_tool(bad)
    assert errors, bad
    assert any(fragment in e for e in errors), errors


def test_validate_tool_reports_every_problem():
    bad = {"name": "1x", "parameters": {"type": "array", "properties": {"a": {"type": "nope"}}, "required": ["b"]}}
    assert len(validate_tool(bad)) == 4


def test_validate_tool_continues_after_bad_property():
    bad = {"name": "f", "parameters": {"type": "object", "properties": {"a": 1, "b": {"type": "nope"}}}}
    errors = validate_tool(bad)
    assert len(errors) == 2


# ------------------------------------------------------------- check_type


@pytest.mark.parametrize(
    ("value", "expected", "ok"),
    [
        ("x", "string", True),
        (1, "string", False),
        (1, "integer", True),
        (1.0, "integer", False),
        (True, "integer", False),
        (1, "number", True),
        (1.5, "number", True),
        (False, "number", False),
        ("1", "number", False),
        (True, "boolean", True),
        (0, "boolean", False),
        ([1], "array", True),
        ((1,), "array", False),
        ({}, "object", True),
        ([], "object", False),
        (None, "null", True),
        (0, "null", False),
        ("x", None, True),
        ("x", ["integer", "string"], True),
        (1.5, ["integer", "string"], False),
        ("x", "datetime", False),
    ],
)
def test_check_type(value, expected, ok):
    assert check_type(value, expected) is ok


# ------------------------------------------------------------- validate_call

TOOLS = [
    tool(city={"type": "string"}, days={"type": "integer"}, opt_unit={"type": "string", "enum": ["c", "f"]}),
    {"name": "noargs", "parameters": {"type": "object", "properties": {}}},
]


def test_valid_calls():
    assert validate_call("f", {"city": "Paris", "days": 3}, TOOLS) == []
    assert validate_call("f", {"city": "Paris", "days": 3, "opt_unit": "c"}, TOOLS) == []
    assert validate_call("noargs", {}, TOOLS) == []


def test_unknown_function():
    assert validate_call("g", {}, TOOLS) == ["unknown function 'g'"]


def test_arguments_must_be_object():
    assert validate_call("f", ["Paris"], TOOLS) == ["arguments is not an object"]


def test_missing_required():
    assert validate_call("f", {"city": "Paris"}, TOOLS) == ["missing required argument 'days'"]


def test_unexpected_argument():
    assert validate_call("f", {"city": "P", "days": 1, "zzz": 1}, TOOLS) == ["unexpected argument 'zzz'"]


def test_wrong_type():
    assert validate_call("f", {"city": "P", "days": "3"}, TOOLS) == [
        "argument 'days' has wrong type (expected integer)"
    ]


def test_enum_violation():
    assert validate_call("f", {"city": "P", "days": 1, "opt_unit": "k"}, TOOLS) == ["argument 'opt_unit' not in enum"]


def test_multiple_errors_are_all_reported():
    errors = validate_call("f", {"days": "x", "zzz": 1}, TOOLS)
    assert len(errors) == 3


def test_tool_with_null_required_and_properties():
    t = [{"name": "f", "parameters": {"type": "object", "properties": None, "required": None}}]
    assert validate_call("f", {}, t) == []
    assert validate_call("f", {"a": 1}, t) == ["unexpected argument 'a'"]


def test_tool_with_non_list_required_is_reported_not_crashing():
    t = [{"name": "f", "parameters": {"type": "object", "properties": {}, "required": True}}]
    assert validate_call("f", {}, t) == ["tool schema has a non-list 'required'"]


def test_tool_without_parameters():
    assert validate_call("f", {}, [{"name": "f"}]) == []
    assert validate_call("f", {"a": 1}, [{"name": "f", "parameters": None}]) == ["unexpected argument 'a'"]


def test_unhashable_property_type_is_invalid():
    bad = {"name": "f", "parameters": {"type": "object", "properties": {"a": {"type": {"x": 1}}}}}
    assert any("unknown type" in e for e in validate_tool(bad))
    bad = {"name": "f", "parameters": {"type": "object", "properties": {"a": {"type": ["string", {"x": 1}]}}}}
    assert any("unknown type" in e for e in validate_tool(bad))


def test_normalize_tool_unwraps_only_a_real_openai_wrapper():
    params = {"type": "object", "properties": {"x": {"type": "string"}}}
    wrapped = {"type": "function", "function": {"name": "f", "parameters": params}}
    assert normalize_tool(wrapped)["name"] == "f"
    # "type": "function" without a nested function object is a bare schema, not a wrapper
    bare = {"type": "function", "name": "g", "parameters": params}
    assert normalize_tool(bare)["name"] == "g"
    # a "function" key without "type": "function" is not unwrapped either
    odd = {"name": "h", "function": {"name": "inner"}, "parameters": params}
    assert normalize_tool(odd)["name"] == "h"
