import copy

import pytest

from toolforge.data.clean import (
    Decontaminator,
    dedup,
    dedup_key,
    ngrams,
    normalize_text,
    user_text,
    validate_record,
)

TOOL = {
    "name": "get_weather",
    "description": "d",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}


def rec(text="What's the weather in Paris?", target=None, tools=None, rid="r"):
    return {
        "id": rid,
        "tools": [TOOL] if tools is None else tools,
        "messages": [{"role": "user", "content": text}],
        "target": target or {"type": "call", "calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}]},
    }


# ------------------------------------------------------------- validate_record


def test_valid_call_and_text_records():
    assert validate_record(rec()) == []
    assert validate_record(rec(target={"type": "text", "content": "Sorry, I can't."})) == []


def test_record_with_tool_response_last_is_valid():
    r = rec()
    r["messages"].append({"role": "tool", "content": "{}"})
    assert validate_record(r) == []


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (lambda r: r.update(tools=[]), "no tools"),
        (lambda r: r.update(tools="x"), "no tools"),
        (lambda r: r.update(tools=[TOOL, TOOL]), "duplicate tool names"),
        (
            lambda r: r["tools"].append({"name": "bad name", "parameters": {"type": "object", "properties": {}}}),
            "invalid tool name",
        ),
        (lambda r: r.update(messages=[]), "no messages"),
        (lambda r: r.update(messages="hi"), "no messages"),
        (lambda r: r["messages"].append({"role": "assistant", "content": "x"}), "must end with a user or tool turn"),
        (lambda r: r["messages"].__setitem__(0, {"role": "user", "content": "  "}), "empty message"),
        (lambda r: r["messages"].__setitem__(0, {"role": "user", "content": None}), "empty message"),
        (lambda r: r.update(target=None), "no target"),
        (lambda r: r.update(target={"type": "call", "calls": []}), "call target without calls"),
        (lambda r: r.update(target={"type": "call"}), "call target without calls"),
        (lambda r: r.update(target={"type": "call", "calls": [{"name": "nope", "arguments": {}}]}), "unknown function"),
        (
            lambda r: r.update(target={"type": "call", "calls": [{"name": "get_weather", "arguments": {}}]}),
            "missing required",
        ),
        (lambda r: r.update(target={"type": "call", "calls": [{"arguments": {}}]}), "unknown function"),
        (lambda r: r.update(target={"type": "text", "content": " "}), "empty text target"),
        (lambda r: r.update(target={"type": "text", "content": 3}), "empty text target"),
        (lambda r: r.update(target={"type": "text", "content": "<functioncall> {...}"}), "call marker"),
        (lambda r: r.update(target={"type": "text", "content": "x <tool_call>"}), "call marker"),
        (lambda r: r.update(target={"type": "image"}), "unknown target type"),
    ],
)
def test_invalid_records(mutate, fragment):
    r = copy.deepcopy(rec())
    mutate(r)
    errors = validate_record(r)
    assert any(fragment in e for e in errors), errors


def test_validate_record_collects_tool_and_call_errors():
    r = rec(target={"type": "call", "calls": [{"name": "get_weather", "arguments": {"city": 1, "x": 2}}]})
    errors = validate_record(r)
    assert len(errors) == 2


def test_non_dict_tool_is_reported():
    r = rec(tools=[TOOL, "junk"])
    assert "tool is not an object" in validate_record(r)


# ------------------------------------------------------------- normalisation / dedup


def test_normalize_text():
    assert normalize_text("What's the  WEATHER, in Paris?!") == "what s the weather in paris"
    assert normalize_text("") == ""


def test_user_text_joins_only_user_turns():
    r = rec()
    r["messages"] += [{"role": "assistant", "content": "A"}, {"role": "user", "content": "B"}]
    assert user_text(r) == "What's the weather in Paris? B"


def test_dedup_key_ignores_case_punctuation_and_tool_order():
    other = {"name": "zzz", "parameters": {"type": "object", "properties": {}}}
    a = rec("Weather in PARIS?", tools=[TOOL, other])
    b = rec("weather in paris", tools=[other, TOOL])
    assert dedup_key(a) == dedup_key(b)
    assert dedup_key(a) != dedup_key(rec("weather in rome", tools=[TOOL, other]))
    assert dedup_key(a) != dedup_key(rec("weather in paris", tools=[TOOL]))


def test_dedup_keeps_first():
    a, b, c = rec("x one", rid="a"), rec("X ONE!", rid="b"), rec("x two", rid="c")
    kept, dropped = dedup([a, b, c])
    assert [r["id"] for r in kept] == ["a", "c"]
    assert dropped == 1


def test_dedup_empty():
    assert dedup([]) == ([], 0)


# ------------------------------------------------------------- decontamination


def test_ngrams():
    assert ngrams("a b c d", 2) == {("a", "b"), ("b", "c"), ("c", "d")}
    assert ngrams("a b", 3) == set()
    assert ngrams("a b c", 3) == {("a", "b", "c")}


def test_decontaminator_exact_and_ngram():
    ev = [rec("Find the area of a triangle with a base of 10 units and height of 5 units.")]
    d = Decontaminator(ev, n=8)
    assert d.is_contaminated(rec("find the AREA of a triangle with a base of 10 units and height of 5 units"))
    # shares an 8-gram
    assert d.is_contaminated(rec("Please find the area of a triangle with a base of 10 meters"))
    # shares only a 6-gram
    assert not d.is_contaminated(rec("Please find the area of a triangle using a big base"))
    assert not d.is_contaminated(rec("What's the weather in Paris?"))


def test_decontaminator_short_queries_match_exactly_only():
    d = Decontaminator([rec("hello there")], n=8)
    assert d.is_contaminated(rec("Hello, there!"))
    assert not d.is_contaminated(rec("hello there friend"))


def test_decontaminator_ngram_boundary():
    d = Decontaminator([rec("a b c d e")], n=3)
    assert d.is_contaminated(rec("x c d e y"))
    assert not d.is_contaminated(rec("x c d y e"))


def test_decontaminator_filter_counts():
    d = Decontaminator([rec("same text here")], n=2)
    kept, dropped = d.filter([rec("same text here", rid="1"), rec("other words entirely", rid="2")])
    assert [r["id"] for r in kept] == ["2"]
    assert dropped == 1


def test_decontaminator_rejects_bad_n():
    with pytest.raises(ValueError):
        Decontaminator([], n=0)
    assert Decontaminator([], n=1).n == 1


def test_call_not_checked_when_tool_schema_broken():
    broken = {"name": "get_weather", "parameters": {"type": "object", "properties": {}, "required": True}}
    errors = validate_record(rec(tools=[broken]))
    assert errors == ["required is not a list"]
