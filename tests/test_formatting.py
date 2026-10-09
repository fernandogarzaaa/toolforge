import json

import pytest

from toolforge.formatting import (
    DEFAULT_SYSTEM,
    IM_END,
    ParseResult,
    ToolCall,
    as_openai_tool,
    parse_tool_calls,
    render_assistant_body,
    render_messages,
    render_system,
    render_target,
    render_tool_calls,
    tojson,
)

WEATHER = {
    "name": "get_weather",
    "description": "Get weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}


def test_render_system_without_tools_is_plain_default():
    assert render_system(None) == f"<|im_start|>system\n{DEFAULT_SYSTEM}<|im_end|>\n"
    assert render_system([], "Be brief.") == "<|im_start|>system\nBe brief.<|im_end|>\n"


def test_render_system_with_tools_matches_qwen_layout():
    text = render_system([WEATHER])
    expected = (
        "<|im_start|>system\n"
        + DEFAULT_SYSTEM
        + "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
        + json.dumps({"type": "function", "function": WEATHER}, ensure_ascii=False)
        + "\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
        '<tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n'
        "</tool_call><|im_end|>\n"
    )
    assert text == expected


def test_render_system_lists_each_tool_on_its_own_line():
    other = {"name": "b", "description": "", "parameters": {"type": "object", "properties": {}}}
    text = render_system([WEATHER, other])
    assert '<tools>\n{"type": "function", "function": {"name": "get_weather"' in text
    assert '}}\n{"type": "function", "function": {"name": "b"' in text


def test_as_openai_tool_wraps_once():
    wrapped = as_openai_tool(WEATHER)
    assert wrapped == {"type": "function", "function": WEATHER}
    assert as_openai_tool(wrapped) is wrapped
    # a "type": "function" without "function" key is a bare schema and gets wrapped
    odd = {"type": "function", "name": "x"}
    assert as_openai_tool(odd) == {"type": "function", "function": odd}


def test_tojson_keeps_unicode_and_default_separators():
    assert tojson({"a": "é", "b": [1, 2]}) == '{"a": "é", "b": [1, 2]}'


def test_render_tool_calls_single_and_multiple():
    one = render_tool_calls([ToolCall("f", {"x": 1})])
    assert one == '<tool_call>\n{"name": "f", "arguments": {"x": 1}}\n</tool_call>'
    two = render_tool_calls([ToolCall("f", {}), ToolCall("g", {"y": "z"})])
    assert two == (
        '<tool_call>\n{"name": "f", "arguments": {}}\n</tool_call>\n'
        '<tool_call>\n{"name": "g", "arguments": {"y": "z"}}\n</tool_call>'
    )


def test_render_assistant_body_combinations():
    call = [ToolCall("f", {})]
    assert render_assistant_body("hi", None) == "hi"
    assert render_assistant_body(None, None) == ""
    assert render_assistant_body(None, call) == render_tool_calls(call)
    assert render_assistant_body("Let me check.", call) == "Let me check.\n" + render_tool_calls(call)


def test_render_messages_full_conversation():
    msgs = [
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": None, "tool_calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}]},
        {"role": "tool", "content": '{"temp": 20}'},
        {"role": "tool", "content": '{"temp": 21}'},
        {"role": "assistant", "content": "It is 20C."},
        {"role": "user", "content": "Thanks"},
    ]
    text = render_messages(msgs, [WEATHER], add_generation_prompt=True)
    assert text.startswith(render_system([WEATHER]))
    tail = text[len(render_system([WEATHER])) :]
    assert tail == (
        "<|im_start|>user\nWeather in Paris?<|im_end|>\n"
        '<|im_start|>assistant\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call><|im_end|>\n'
        '<|im_start|>user\n<tool_response>\n{"temp": 20}\n</tool_response>\n<tool_response>\n{"temp": 21}\n</tool_response><|im_end|>\n'
        "<|im_start|>assistant\nIt is 20C.<|im_end|>\n"
        "<|im_start|>user\nThanks<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def test_render_messages_uses_given_system_and_no_generation_prompt():
    text = render_messages([{"role": "system", "content": "S"}, {"role": "user", "content": "u"}])
    assert text == "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nu<|im_end|>\n"


def test_render_messages_does_not_mutate_input():
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "u"}]
    render_messages(msgs)
    assert len(msgs) == 2


def test_render_messages_rejects_unknown_role():
    with pytest.raises(ValueError, match="unknown role"):
        render_messages([{"role": "robot", "content": "x"}])


def test_render_target():
    assert render_target({"type": "text", "content": "No can do"}) == "No can do"
    call = {"type": "call", "calls": [{"name": "f", "arguments": {"a": 1}}]}
    assert render_target(call) == '<tool_call>\n{"name": "f", "arguments": {"a": 1}}\n</tool_call>'
    with pytest.raises(ValueError, match="unknown target type"):
        render_target({"type": "other"})


# ---------------------------------------------------------------- parsing


def test_parse_single_tagged_call():
    res = parse_tool_calls('<tool_call>\n{"name": "f", "arguments": {"a": 1}}\n</tool_call>')
    assert res.calls == [ToolCall("f", {"a": 1})]
    assert res.errors == []
    assert res.format == "tagged"
    assert res.attempted_call is True
    assert res.content == ""


def test_parse_roundtrip_of_rendered_calls():
    calls = [ToolCall("f", {"a": [1, 2]}), ToolCall("g", {"s": "x"})]
    assert parse_tool_calls(render_tool_calls(calls)).calls == calls


def test_parse_keeps_surrounding_text_as_content():
    res = parse_tool_calls('Sure.\n<tool_call>\n{"name": "f", "arguments": {}}\n</tool_call>\nDone')
    assert res.content == "Sure.\n\nDone"
    assert res.calls == [ToolCall("f", {})]


def test_parse_tolerates_missing_closing_tag_and_im_end():
    res = parse_tool_calls('<tool_call>\n{"name": "f", "arguments": {"a": 2}}' + IM_END)
    assert res.calls == [ToolCall("f", {"a": 2})]
    assert res.errors == []


def test_parse_string_arguments_and_parameters_alias():
    res = parse_tool_calls('<tool_call>{"name": "f", "arguments": "{\\"a\\": 1}"}</tool_call>')
    assert res.calls == [ToolCall("f", {"a": 1})]
    res = parse_tool_calls('<tool_call>{"name": "f", "parameters": {"b": 2}}</tool_call>')
    assert res.calls == [ToolCall("f", {"b": 2})]
    res = parse_tool_calls('<tool_call>{"name": "f", "arguments": ""}</tool_call>')
    assert res.calls == [ToolCall("f", {})]
    res = parse_tool_calls('<tool_call>{"name": "f", "arguments": null}</tool_call>')
    assert res.calls == [ToolCall("f", {})]
    res = parse_tool_calls('<tool_call>{"name": "f"}</tool_call>')
    assert res.calls == [ToolCall("f", {})]


@pytest.mark.parametrize(
    "body",
    [
        "{not json}",
        '["f"]',
        '{"arguments": {}}',
        '{"name": "", "arguments": {}}',
        '{"name": 3, "arguments": {}}',
        '{"name": "f", "arguments": [1]}',
        '{"name": "f", "arguments": "[1]"}',
    ],
)
def test_parse_malformed_tagged_call_is_an_attempt_with_error(body):
    res = parse_tool_calls(f"<tool_call>{body}</tool_call>")
    assert res.calls == []
    assert len(res.errors) == 1
    assert res.attempted_call is True
    assert res.format == "tagged"


def test_parse_mixed_good_and_bad_blocks():
    res = parse_tool_calls('<tool_call>{"name": "f", "arguments": {}}</tool_call><tool_call>oops</tool_call>')
    assert res.calls == [ToolCall("f", {})]
    assert len(res.errors) == 1


def test_parse_bare_json_object_and_list():
    res = parse_tool_calls('  {"name": "f", "arguments": {"a": 1}}  ')
    assert res.format == "bare_json"
    assert res.calls == [ToolCall("f", {"a": 1})]
    assert res.content == ""
    res = parse_tool_calls('[{"name": "f", "arguments": {}}, {"name": "g", "arguments": {}}]')
    assert [c.name for c in res.calls] == ["f", "g"]


def test_parse_bare_json_with_bad_arguments_records_error():
    res = parse_tool_calls('{"name": "f", "arguments": [1]}')
    assert res.format == "bare_json"
    assert res.calls == []
    assert res.errors and res.attempted_call


@pytest.mark.parametrize(
    "text",
    [
        "The weather is nice.",
        '{"answer": 42}',
        "[1, 2, 3]",
        "[]",
        "{broken",
        '[{"name": "f"}, 3]',
    ],
)
def test_parse_plain_text_is_not_a_call(text):
    res = parse_tool_calls(text)
    assert res.calls == []
    assert res.errors == []
    assert res.attempted_call is False
    assert res.format == "none"
    assert res.content == text.strip()


def test_parse_result_default_has_no_attempt():
    assert ParseResult().attempted_call is False
    assert ParseResult(errors=["x"]).attempted_call is True


def test_toolcall_to_json():
    assert json.loads(ToolCall("f", {"a": "é"}).to_json()) == {"name": "f", "arguments": {"a": "é"}}
    assert "é" in ToolCall("f", {"a": "é"}).to_json()
