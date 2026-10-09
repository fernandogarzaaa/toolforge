"""The hand-written renderer must produce exactly what the real Qwen2.5 chat template produces.

Needs ``transformers`` and the Qwen2.5 tokenizer locally (set ``TOOLFORGE_PARITY_MODEL`` to a model
directory or hub id). Skipped otherwise.
"""

from __future__ import annotations

import os

import pytest

from toolforge.formatting import render_messages

pytestmark = pytest.mark.parity

MODEL = os.environ.get("TOOLFORGE_PARITY_MODEL")

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get weather – with ünïcode",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
            "required": ["city"],
        },
    },
    {"name": "noop", "description": "", "parameters": {"type": "object", "properties": {}}},
]

CONVERSATIONS = [
    [{"role": "user", "content": "Weather in Paris?"}],
    [{"role": "system", "content": "Custom system."}, {"role": "user", "content": "hi"}],
    [
        {"role": "user", "content": "Weather in Paris and Rome?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"name": "get_weather", "arguments": {"city": "Paris"}},
                {"name": "get_weather", "arguments": {"city": "Rome", "days": 2}},
            ],
        },
        {"role": "tool", "content": '{"t": 20}'},
        {"role": "tool", "content": '{"t": 25}'},
        {"role": "assistant", "content": "20 and 25."},
        {"role": "user", "content": "thanks"},
    ],
]


@pytest.mark.skipif(not MODEL, reason="set TOOLFORGE_PARITY_MODEL to a Qwen2.5 model dir or id")
@pytest.mark.parametrize("tools", [TOOLS, None])
@pytest.mark.parametrize("conv", CONVERSATIONS)
def test_renderer_matches_hf_chat_template(conv, tools):
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    hf_msgs = []
    for m in conv:
        m = dict(m)
        if m.get("tool_calls"):
            m["tool_calls"] = [{"type": "function", "function": c} for c in m["tool_calls"]]
        hf_msgs.append(m)
    hf_tools = [{"type": "function", "function": t} for t in tools] if tools else None
    expected = tok.apply_chat_template(hf_msgs, tools=hf_tools, tokenize=False, add_generation_prompt=True)
    assert render_messages(conv, tools, add_generation_prompt=True) == expected
