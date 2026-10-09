import pytest

from toolforge.formatting import ParseResult, ToolCall, parse_tool_calls
from toolforge.scorer import (
    Verdict,
    call_matches,
    gold_from_calls,
    normalize_string,
    params_match,
    score,
    value_matches,
)


def parsed(*calls, errors=()):
    return ParseResult(calls=[ToolCall(n, a) for n, a in calls], errors=list(errors))


# ------------------------------------------------------------- strings / values


def test_normalize_string_follows_bfcl():
    assert normalize_string("New York, NY") == "newyorkny"
    assert normalize_string("a.b/c-d_e*f^g") == "abcdefg"
    assert normalize_string("San Francisco") == normalize_string("san_francisco")
    assert normalize_string("x!y") == "x!y"


@pytest.mark.parametrize(
    ("value", "acceptable", "ok"),
    [
        ("Paris", "paris", True),
        ("Paris, France", "Paris France", True),
        ("Paris", "London", False),
        (5, "5", False),
        ("5", 5, False),
        (5, 5, True),
        (5, 5.0, True),
        (5.0, 5, True),
        (5.5, 5, False),
        (5.000000001, 5.0, True),
        (5.1, 5.0, False),
        (1e9 + 2.0, 1e9, False),
        (True, True, True),
        (True, 1, False),
        (1, True, False),
        (False, True, False),
        ([1, "A"], [1, "a"], True),
        ([1, 2], [2, 1], False),
        ([1], [1, 2], False),
        ((1, 2), [1, 2], False),
        ({"k": 1}, {"k": [1]}, True),
        ({"k": 2}, {"k": [1]}, False),
        ([1], {"k": [1]}, False),
        (None, None, True),
        (0, None, False),
        ({"a": 1}, {"a": [1]}, True),
    ],
)
def test_value_matches(value, acceptable, ok):
    assert value_matches(value, acceptable) is ok


def test_value_matches_fallback_equality_for_other_types():
    assert value_matches((1, 2), (1, 2)) is True
    assert value_matches((1, 2), (1, 3)) is False


def test_value_matches_relative_tolerance_scales_with_magnitude():
    assert value_matches(1e12 + 1.0, 1e12) is True  # 1e-9 relative
    assert value_matches(1e12 + 10000.0, 1e12) is False
    assert value_matches(1e-12, 0.0) is True  # absolute floor of 1e-9


# ------------------------------------------------------------- params


def test_params_match_ok_and_optional_omitted():
    assert params_match({"a": 1}, {"a": [1], "b": ["", "x"]}) == Verdict(True, "ok")


def test_params_match_unexpected_param():
    assert params_match({"a": 1, "z": 2}, {"a": [1]}) == Verdict(False, "unexpected_param:z")


def test_params_match_missing_required():
    assert params_match({}, {"a": [1]}) == Verdict(False, "missing_param:a")


def test_params_match_value_mismatch():
    assert params_match({"a": 2}, {"a": [1, 3]}) == Verdict(False, "value_mismatch:a")


def test_params_match_any_of_options():
    assert params_match({"a": 3}, {"a": [1, 3]}).correct


def test_params_match_explicit_empty_string_is_allowed_when_listed():
    assert params_match({"b": ""}, {"b": ["", "x"]}).correct
    # an empty string given for a param whose only options are non-empty is wrong
    assert not params_match({"b": ""}, {"b": ["x"]}).correct


def test_params_match_empty_option_does_not_accept_other_values():
    assert params_match({"b": "y"}, {"b": ["", "x"]}) == Verdict(False, "value_mismatch:b")


def test_params_match_scalar_option_is_wrapped():
    assert params_match({"a": 1}, {"a": 1}).correct
    assert not params_match({"a": 2}, {"a": 1}).correct


# ------------------------------------------------------------- calls


def test_call_matches():
    gold = {"f": {"a": [1]}}
    assert call_matches(ToolCall("f", {"a": 1}), gold).correct
    assert call_matches(ToolCall("g", {"a": 1}), gold) == Verdict(False, "wrong_function")
    assert call_matches(ToolCall("f", {"a": 2}), gold) == Verdict(False, "value_mismatch:a")


def test_call_matches_rejects_malformed_gold():
    with pytest.raises(ValueError):
        call_matches(ToolCall("f", {}), {"f": {}, "g": {}})
    with pytest.raises(ValueError):
        call_matches(ToolCall("f", {}), {})


# ------------------------------------------------------------- score by category

GOLD = [{"f": {"a": [1]}}]


def test_simple_correct():
    assert score(parsed(("f", {"a": 1})), GOLD, "simple") == Verdict(True, "ok")


def test_simple_wrong_value():
    assert score(parsed(("f", {"a": 2})), GOLD, "multiple") == Verdict(False, "value_mismatch:a")


def test_call_category_without_call():
    assert score(ParseResult(content="hi"), GOLD, "simple") == Verdict(False, "no_call")


def test_malformed_call_fails_even_with_a_good_one():
    assert score(parsed(("f", {"a": 1}), errors=["bad"]), GOLD, "simple") == Verdict(False, "malformed_call")
    assert score(parsed(errors=["bad"]), GOLD, "simple") == Verdict(False, "malformed_call")


def test_extra_call_is_wrong_count():
    assert score(parsed(("f", {"a": 1}), ("f", {"a": 1})), GOLD, "simple") == Verdict(False, "wrong_call_count")


def test_irrelevance():
    assert score(ParseResult(content="Sorry"), [], "irrelevance") == Verdict(True, "ok")
    assert score(parsed(("f", {})), [], "irrelevance") == Verdict(False, "spurious_call")
    assert score(parsed(errors=["x"]), [], "irrelevance") == Verdict(False, "spurious_call")


def test_unknown_category():
    with pytest.raises(ValueError, match="unknown category"):
        score(ParseResult(), [], "multi_turn")


PAR_GOLD = [{"f": {"city": ["Paris"]}}, {"f": {"city": ["Rome"]}}]


def test_parallel_any_order():
    assert score(parsed(("f", {"city": "Rome"}), ("f", {"city": "Paris"})), PAR_GOLD, "parallel").correct
    assert score(parsed(("f", {"city": "Paris"}), ("f", {"city": "Rome"})), PAR_GOLD, "parallel").correct


def test_parallel_needs_one_to_one_matching():
    res = score(parsed(("f", {"city": "Paris"}), ("f", {"city": "Paris"})), PAR_GOLD, "parallel")
    assert res == Verdict(False, "parallel_mismatch")


def test_parallel_count_mismatch():
    assert score(parsed(("f", {"city": "Paris"})), PAR_GOLD, "parallel") == Verdict(False, "wrong_call_count")


def test_parallel_backtracks_over_greedy_choice():
    gold = [{"f": {"x": [1, 2]}}, {"f": {"x": [1]}}]
    # greedy would give call x=1 to gold[0] and then fail x=2 on gold[1]; backtracking finds x=1->gold[1]
    assert score(parsed(("f", {"x": 1}), ("f", {"x": 2})), gold, "parallel").correct


def test_end_to_end_from_text():
    text = '<tool_call>\n{"name": "f", "arguments": {"a": 1.0}}\n</tool_call>'
    assert score(parse_tool_calls(text), GOLD, "simple").correct


def test_gold_from_calls():
    assert gold_from_calls([{"name": "f", "arguments": {"a": 1, "b": "x"}}]) == [{"f": {"a": [1], "b": ["x"]}}]
    assert gold_from_calls([]) == []
    # a list-valued argument stays one option (a list), not several
    assert gold_from_calls([{"name": "f", "arguments": {"xs": [1, 2]}}]) == [{"f": {"xs": [[1, 2]]}}]
    assert score(
        parsed(("f", {"xs": [1, 2]})), gold_from_calls([{"name": "f", "arguments": {"xs": [1, 2]}}]), "simple"
    ).correct


# ------------------------------------------------- boundaries found by mutation testing


def test_float_tolerance_is_inclusive_and_absolute_below_one():
    # |value - acceptable| <= 1e-9 * max(1, |acceptable|)
    assert value_matches(1e-9, 0.0)  # exactly on the boundary: accepted
    assert not value_matches(1.5e-9, 0.0)  # tolerance does not grow below |acceptable| = 1
    assert value_matches(1000.0 + 1e-7, 1000.0)  # relative above 1
    assert not value_matches(1000.0 + 1e-5, 1000.0)


@pytest.mark.parametrize("possible", [{}, {"f": {}, "g": {}}])
def test_call_matches_rejects_malformed_possible_answer(possible):
    with pytest.raises(ValueError, match="exactly one function"):
        call_matches(ToolCall("f", {}), possible)


def test_perfect_matching_returns_bool():
    from toolforge.scorer import _perfect_matching

    gold = [{"f": {"x": [1]}}, {"f": {"x": [2]}}]
    assert _perfect_matching([ToolCall("f", {"x": 2}), ToolCall("f", {"x": 1})], gold) is True
    assert _perfect_matching([ToolCall("f", {"x": 2}), ToolCall("f", {"x": 2})], gold) is False
