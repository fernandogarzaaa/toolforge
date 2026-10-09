import pytest

from toolforge.formatting import IM_END, ToolCall, parse_tool_calls, render_messages, render_tool_calls
from toolforge.pairs import (
    NO_CALL_TEXT,
    Pair,
    _perturb_value,
    build_pairs,
    clean_completion,
    synthetic_negatives,
)
from toolforge.scorer import gold_from_calls, score

WEATHER = {
    "name": "get_weather",
    "description": "",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
        "required": ["city"],
    },
}
NEWS = {
    "name": "get_news",
    "description": "",
    "parameters": {"type": "object", "properties": {"topic": {"type": "string"}}},
}


def call_example(eid="e1", args=None, tools=None):
    args = {"city": "Paris", "days": 2} if args is None else args
    calls = [{"name": "get_weather", "arguments": args}]
    return {
        "id": eid,
        "category": "multiple" if tools and len(tools) > 1 else "simple",
        "tools": tools or [WEATHER],
        "messages": [{"role": "user", "content": "Weather in Paris for 2 days?"}],
        "target": {"type": "call", "calls": calls},
        "gold": gold_from_calls(calls),
    }


def irrelevant_example(eid="i1"):
    return {
        "id": eid,
        "category": "irrelevance",
        "tools": [WEATHER],
        "messages": [{"role": "user", "content": "Tell me a joke"}],
        "target": {"type": "text", "content": "Why did the chicken..."},
        "gold": [],
    }


def gold_text(ex):
    return render_tool_calls([ToolCall(c["name"], c["arguments"]) for c in ex["target"]["calls"]])


def wrong_value(ex):
    return render_tool_calls([ToolCall("get_weather", {"city": "Rome", "days": 2})])


def is_correct(text, ex):
    return score(parse_tool_calls(text), ex["gold"], ex["category"]).correct


def test_clean_completion():
    assert clean_completion("  abc" + IM_END + "junk") == "abc"
    assert clean_completion("abc\n") == "abc"


def test_pair_from_model_mistake():
    ex = call_example()
    pairs = build_pairs([ex], {"e1": [wrong_value(ex) + IM_END]}, synthetic=False)
    assert pairs == [
        Pair(
            id="e1",
            prompt=render_messages(ex["messages"], ex["tools"], add_generation_prompt=True),
            chosen=gold_text(ex),
            rejected=wrong_value(ex),
            source="model:value_mismatch:city",
        )
    ]
    assert pairs[0].to_dict()["source"] == "model:value_mismatch:city"


def test_correct_model_output_gives_no_pair_without_synthetic():
    ex = call_example()
    assert build_pairs([ex], {"e1": [gold_text(ex)]}, synthetic=False) == []


def test_missing_generations_give_no_pair_without_synthetic():
    assert build_pairs([call_example()], {}, synthetic=False) == []


def test_synthetic_fallback_when_model_is_right():
    ex = call_example()
    pairs = build_pairs([ex], {"e1": [gold_text(ex)]}, synthetic=True)
    assert len(pairs) == 1
    assert pairs[0].source == "synthetic:drop_required"
    assert is_correct(pairs[0].chosen, ex)
    assert not is_correct(pairs[0].rejected, ex)


def test_model_mistake_preferred_over_synthetic():
    ex = call_example()
    pairs = build_pairs([ex], {"e1": [wrong_value(ex)]}, synthetic=True)
    assert [p.source for p in pairs] == ["model:value_mismatch:city"]


def test_most_informative_mistake_is_chosen():
    ex = call_example()
    gens = {"e1": ["I cannot help.", "<tool_call>broken</tool_call>", wrong_value(ex)]}
    pairs = build_pairs([ex], gens, synthetic=False)
    assert pairs[0].rejected == wrong_value(ex)
    two = build_pairs([ex], gens, synthetic=False, max_pairs_per_prompt=2)
    assert [p.source for p in two] == ["model:value_mismatch:city", "model:no_call"]
    three = build_pairs([ex], gens, synthetic=False, max_pairs_per_prompt=3)
    assert [p.source.split(":")[1] for p in three] == ["value_mismatch", "no_call", "malformed_call"]


def test_ties_keep_generation_order():
    ex = call_example()
    a = render_tool_calls([ToolCall("get_weather", {"city": "Rome", "days": 2})])
    b = render_tool_calls([ToolCall("get_weather", {"city": "Oslo", "days": 2})])
    assert build_pairs([ex], {"e1": [b, a]}, synthetic=False)[0].rejected == b


def test_duplicate_completions_are_counted_once():
    ex = call_example()
    w = wrong_value(ex)
    pairs = build_pairs([ex], {"e1": [w, w + IM_END, " " + w]}, synthetic=False, max_pairs_per_prompt=3)
    assert len(pairs) == 1


def test_gold_duplicate_completion_never_used_as_rejected():
    ex = call_example()
    pairs = build_pairs([ex], {"e1": [gold_text(ex) + IM_END]}, synthetic=False)
    assert pairs == []


def test_fills_up_with_synthetic_to_max_pairs():
    ex = call_example()
    pairs = build_pairs([ex], {"e1": [wrong_value(ex)]}, synthetic=True, max_pairs_per_prompt=3)
    assert len(pairs) == 3
    assert pairs[0].source.startswith("model:")
    assert all(p.source.startswith("synthetic:") for p in pairs[1:])
    assert len({p.rejected for p in pairs}) == 3
    for p in pairs:
        assert not is_correct(p.rejected, ex)


def test_irrelevance_pair():
    ex = irrelevant_example()
    spurious = render_tool_calls([ToolCall("get_weather", {"city": "x"})])
    pairs = build_pairs([ex], {"i1": [spurious]}, synthetic=False)
    assert pairs[0].chosen == "Why did the chicken..."
    assert pairs[0].source == "model:spurious_call"
    synth = build_pairs([ex], {"i1": ["Sure, here is a joke"]}, synthetic=True)
    assert synth[0].source == "synthetic:spurious_call"
    assert parse_tool_calls(synth[0].rejected).calls[0].name == "get_weather"


def test_gold_that_fails_scorer_is_skipped():
    ex = call_example()
    ex["gold"] = [{"get_weather": {"city": ["Rome"]}}]  # inconsistent with the target
    assert build_pairs([ex], {"e1": [wrong_value(ex)]}, synthetic=True) == []


def test_invalid_max_pairs():
    with pytest.raises(ValueError):
        build_pairs([], {}, max_pairs_per_prompt=0)


def test_deterministic_given_seed():
    exs = [call_example(f"e{i}", tools=[WEATHER, NEWS]) for i in range(5)]
    gens = {e["id"]: [gold_text(e)] for e in exs}
    a = build_pairs(exs, gens, max_pairs_per_prompt=3, seed=7)
    b = build_pairs(exs, gens, max_pairs_per_prompt=3, seed=7)
    assert a == b


def test_seed_changes_order_of_later_synthetic_kinds():
    exs = [call_example(f"e{i}", tools=[WEATHER, NEWS]) for i in range(12)]
    gens = {e["id"]: [gold_text(e)] for e in exs}
    a = [p.source for p in build_pairs(exs, gens, max_pairs_per_prompt=2, seed=1)]
    b = [p.source for p in build_pairs(exs, gens, max_pairs_per_prompt=2, seed=2)]
    assert a != b
    # the first synthetic pick is always the most informative kind
    assert all(s == "synthetic:drop_required" for s in a[::2])


# ------------------------------------------------------------- synthetic negatives


def test_synthetic_negatives_for_call_with_required():
    ex = call_example(tools=[WEATHER, NEWS])
    kinds = synthetic_negatives(ex)
    assert [k for k, _ in kinds] == ["drop_required", "perturb_value", "wrong_function", "no_call"]
    by_kind = dict(kinds)
    dropped = parse_tool_calls(by_kind["drop_required"]).calls[0]
    assert dropped == ToolCall("get_weather", {"days": 2})
    perturbed = parse_tool_calls(by_kind["perturb_value"]).calls[0]
    assert perturbed == ToolCall("get_weather", {"city": "Paris extra", "days": 2})
    wrong = parse_tool_calls(by_kind["wrong_function"]).calls[0]
    assert wrong.name == "get_news"
    assert by_kind["no_call"] == NO_CALL_TEXT


def test_synthetic_negatives_without_required_or_args():
    ex = call_example(args={})
    kinds = [k for k, _ in synthetic_negatives(ex)]
    assert kinds == ["no_call"]


def test_synthetic_negatives_when_required_arg_absent():
    ex = call_example(args={"days": 3})
    kinds = [k for k, _ in synthetic_negatives(ex)]
    assert kinds == ["perturb_value", "no_call"]


def test_synthetic_negatives_keep_other_parallel_calls():
    ex = call_example()
    ex["target"]["calls"].append({"name": "get_weather", "arguments": {"city": "Rome"}})
    for _, text in synthetic_negatives(ex)[:2]:
        calls = parse_tool_calls(text).calls
        assert calls[1] == ToolCall("get_weather", {"city": "Rome"})


def test_synthetic_negative_for_irrelevance_uses_required_args():
    kinds = synthetic_negatives(irrelevant_example())
    assert len(kinds) == 1
    kind, text = kinds[0]
    assert kind == "spurious_call"
    assert parse_tool_calls(text).calls == [ToolCall("get_weather", {"city": "unknown"})]


def test_synthetic_negative_for_irrelevance_without_tools():
    ex = irrelevant_example()
    ex["tools"] = []
    assert synthetic_negatives(ex) == []


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, False),
        (False, True),
        (3, 4),
        (1.5, 4.0),
        ("x", "x extra"),
        ("", "unknown"),
        ([1, 2], [1]),
        ([], ["unknown"]),
        ({"a": 1}, {}),
        (None, "unknown"),
    ],
)
def test_perturb_value(value, expected):
    assert _perturb_value(value) == expected


# ------------------------------------------------- behaviours pinned by mutation testing


def test_rejection_preference_order_is_strict_and_complete():
    from toolforge.pairs import REASON_RANK

    expected = [
        "value_mismatch",
        "missing_param",
        "unexpected_param",
        "wrong_function",
        "parallel_mismatch",
        "wrong_call_count",
        "spurious_call",
        "no_call",
        "malformed_call",
    ]
    assert sorted(REASON_RANK, key=REASON_RANK.__getitem__) == expected
    assert sorted(REASON_RANK.values()) == list(range(len(expected)))


def test_perturb_value_rounds_floats_to_three_places():
    assert _perturb_value(0.1234) == 1.247


def test_wrong_function_negative_keeps_the_other_parallel_calls():
    calls = [{"name": "get_weather", "arguments": {"city": "Paris"}}, {"name": "get_news", "arguments": {"topic": "x"}}]
    ex = {
        "id": "p",
        "category": "parallel",
        "tools": [WEATHER, NEWS],
        "messages": [],
        "target": {"type": "call", "calls": calls},
        "gold": gold_from_calls(calls),
    }
    kinds = dict(synthetic_negatives(ex))
    parsed = parse_tool_calls(kinds["wrong_function"])
    assert [c.name for c in parsed.calls] == ["get_news", "get_news"]
    assert parsed.calls[1].arguments == {"topic": "x"}


def test_zero_max_pairs_is_rejected():
    with pytest.raises(ValueError):
        build_pairs([call_example()], {}, max_pairs_per_prompt=0)


def test_default_seed_is_zero():
    exs = [call_example(f"e{i}", tools=[WEATHER, NEWS]) for i in range(12)]
    gens = {e["id"]: [gold_text(e)] for e in exs}
    assert build_pairs(exs, gens, max_pairs_per_prompt=3) == build_pairs(exs, gens, max_pairs_per_prompt=3, seed=0)
    assert build_pairs(exs, gens, max_pairs_per_prompt=3) != build_pairs(exs, gens, max_pairs_per_prompt=3, seed=1)


def test_ties_keep_generation_order_not_text_order():
    ex = call_example()
    later_but_smaller = render_tool_calls([ToolCall("get_weather", {"city": "Aachen", "days": 2})])
    first = render_tool_calls([ToolCall("get_weather", {"city": "Zurich", "days": 2})])
    assert first > later_but_smaller
    assert build_pairs([ex], {"e1": [first, later_but_smaller]}, synthetic=False)[0].rejected == first


def test_full_example_does_not_consume_randomness_for_the_next():
    # an example already filled with model mistakes must not shuffle synthetic kinds, so later
    # examples get the same synthetic pairs as when built alone
    bad = call_example("e0")
    gens = {"e0": [wrong_value(bad), "I cannot help."]}
    for seed in range(6):
        ok = call_example("e1", tools=[WEATHER, NEWS])
        alone = build_pairs([ok], {"e1": [gold_text(ok)]}, max_pairs_per_prompt=2, seed=seed)
        both = build_pairs([bad, ok], {**gens, "e1": [gold_text(ok)]}, max_pairs_per_prompt=2, seed=seed)
        assert [p for p in both if p.id == "e1"] == alone


def test_only_the_most_informative_synthetic_kind_is_fixed_first():
    ex = call_example(tools=[WEATHER, NEWS])
    seconds = set()
    for seed in range(20):
        pairs = build_pairs([ex], {"e1": [gold_text(ex)]}, max_pairs_per_prompt=2, seed=seed)
        assert pairs[0].source == "synthetic:drop_required"
        seconds.add(pairs[1].source)
    assert len(seconds) > 1  # the remaining kinds are shuffled


def test_every_synthetic_kind_is_used_when_room_allows():
    # the shuffle reorders the kinds after the first one; it must never drop any of them
    ex = call_example(tools=[WEATHER, NEWS])
    kinds = {k for k, _ in synthetic_negatives(ex)}
    assert len(kinds) >= 3
    for seed in range(5):
        pairs = build_pairs([ex], {"e1": [gold_text(ex)]}, max_pairs_per_prompt=10, seed=seed)
        assert {p.source for p in pairs} == {f"synthetic:{k}" for k in kinds}
