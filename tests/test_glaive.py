import json

import pytest

from toolforge.data.glaive import GlaiveParseError, parse_functioncall, parse_tools, split_turns, to_record

TOOL_A = {
    "name": "calculate_tip",
    "description": "Calculate tip",
    "parameters": {
        "type": "object",
        "properties": {"bill": {"type": "number"}, "pct": {"type": "number"}},
        "required": ["bill", "pct"],
    },
}
TOOL_B = {"name": "get_date", "description": "Get the date", "parameters": {}}
EOT = "<\u200b|endoftext|>"


def system(*tools):
    return (
        "SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -\n"
        + "\n\n".join(json.dumps(t, indent=4) for t in tools)
        + "\n"
    )


def test_parse_tools_multiple_and_whitespace():
    assert parse_tools(system(TOOL_A, TOOL_B)) == [TOOL_A, TOOL_B]
    assert parse_tools("SYSTEM: You are a helpful assistant") == []
    assert parse_tools(system(TOOL_A) + "\n\n   ") == [TOOL_A]


def test_parse_tools_errors():
    with pytest.raises(GlaiveParseError, match="bad tool json"):
        parse_tools("SYSTEM: x {not json")
    with pytest.raises(GlaiveParseError, match="not an object"):
        parse_tools('SYSTEM: x {"name": "a"} [1]')


def test_parse_functioncall_glaive_quoted_arguments():
    text = '<functioncall> {"name": "calculate_tip", "arguments": \'{"bill": 50, "pct": 20}\'} '
    assert parse_functioncall(text) == {"name": "calculate_tip", "arguments": {"bill": 50, "pct": 20}}


def test_parse_functioncall_empty_arguments():
    assert parse_functioncall('<functioncall> {"name": "get_date", "arguments": \'\'}') == {
        "name": "get_date",
        "arguments": {},
    }
    assert parse_functioncall('<functioncall> {"name": "get_date", "arguments": \'{}\'}') == {
        "name": "get_date",
        "arguments": {},
    }


def test_parse_functioncall_plain_json_variants():
    assert parse_functioncall('<functioncall> {"name": "f", "arguments": {"a": 1}}') == {
        "name": "f",
        "arguments": {"a": 1},
    }
    assert parse_functioncall('<functioncall> {"name": "f", "arguments": "{\\"a\\": 2}"}') == {
        "name": "f",
        "arguments": {"a": 2},
    }


@pytest.mark.parametrize(
    "text",
    [
        "<functioncall> not json at all",
        '<functioncall> {"name": "f", "arguments": \'{broken\'}',
        '<functioncall> {"name": "f", "arguments": \'[1, 2]\'}',
        '<functioncall> {"name": "f", "arguments": 5}',
        '<functioncall> {"arguments": "{}"}',
    ],
)
def test_parse_functioncall_errors(text):
    with pytest.raises(GlaiveParseError):
        parse_functioncall(text)


def test_split_turns_strips_end_of_text_markers():
    chat = (
        f"USER: Hi there\n\n\nASSISTANT: Hello! {EOT}\n\n\nFUNCTION RESPONSE: {{}}\n\n\nASSISTANT: Bye <|endoftext|>\n"
    )
    assert split_turns(chat) == [
        ("USER", "Hi there"),
        ("ASSISTANT", "Hello!"),
        ("FUNCTION RESPONSE", "{}"),
        ("ASSISTANT", "Bye"),
    ]


def test_split_turns_empty():
    assert split_turns("") == []


def row(chat, *tools):
    return {"system": system(*(tools or (TOOL_A,))), "chat": chat}


CALL = '<functioncall> {"name": "calculate_tip", "arguments": \'{"bill": 50, "pct": 20}\'}'


def test_to_record_simple_call():
    r = to_record(
        row(
            f'USER: Tip on $50 at 20%?\n\n\nASSISTANT: {CALL} {EOT}\n\n\nFUNCTION RESPONSE: {{"tip": 10}}\n\n\nASSISTANT: $10 {EOT}'
        ),
        7,
    )
    assert r == {
        "id": "glaive-7",
        "source": "glaive-function-calling-v2",
        "category": "simple",
        "tools": [{**TOOL_A}],
        "messages": [{"role": "user", "content": "Tip on $50 at 20%?"}],
        "target": {"type": "call", "calls": [{"name": "calculate_tip", "arguments": {"bill": 50, "pct": 20}}]},
    }


def test_to_record_multiple_tools_and_filler_collapsed():
    chat = f"USER: Tip?\n\n\nASSISTANT: Sure, let me calculate. {EOT}\n\n\nASSISTANT: {CALL} {EOT}"
    r = to_record(row(chat, TOOL_A, TOOL_B), 1)
    assert r is not None
    assert r["category"] == "multiple"
    assert r["target"]["type"] == "call"
    # normalised empty parameters
    assert r["tools"][1]["parameters"] == {"type": "object", "properties": {}}


def test_to_record_irrelevance_vs_clarify():
    no_call = to_record(row(f"USER: Book a flight\n\n\nASSISTANT: I can't book flights. {EOT}"), 2)
    assert no_call is not None
    assert no_call["category"] == "irrelevance"
    assert no_call["target"] == {"type": "text", "content": "I can't book flights."}
    clarify = to_record(
        row(f"USER: Tip?\n\n\nASSISTANT: What is the bill? {EOT}\n\n\nUSER: $50, 20%\n\n\nASSISTANT: {CALL} {EOT}"), 3
    )
    assert clarify is not None
    assert clarify["category"] == "clarify"
    assert clarify["messages"] == [{"role": "user", "content": "Tip?"}]


def test_to_record_text_then_user_is_not_collapsed():
    chat = f"USER: Hi\n\n\nASSISTANT: Hello {EOT}\n\n\nUSER: tip\n\n\nASSISTANT: {CALL}"
    r = to_record(row(chat), 4)
    assert r is not None and r["target"]["type"] == "text"


@pytest.mark.parametrize(
    "r",
    [
        {"system": "SYSTEM: You are a helpful assistant", "chat": "USER: hi\n\n\nASSISTANT: hello"},
        {"system": system(TOOL_A), "chat": ""},
        {"system": system(TOOL_A), "chat": "ASSISTANT: hello"},
        {"system": system(TOOL_A), "chat": "USER: hi"},
        {"system": system(TOOL_A), "chat": "USER: hi\n\n\nFUNCTION RESPONSE: {}"},
        {"system": system(TOOL_A), "chat": f"USER: hi\n\n\nASSISTANT: {EOT}"},
    ],
)
def test_to_record_none_cases(r):
    assert to_record(r, 0) is None


def test_to_record_missing_keys():
    assert to_record({}, 0) is None
