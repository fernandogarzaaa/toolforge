import math

import pytest

from toolforge.stats import Interval, _percentile, bootstrap_ci, paired_bootstrap_diff


def test_constant_values_have_zero_width_ci():
    ci = bootstrap_ci([1.0] * 20)
    assert (ci.mean, ci.low, ci.high, ci.n) == (1.0, 1.0, 1.0, 20)


def test_ci_brackets_mean_and_is_deterministic():
    vals = [1.0] * 30 + [0.0] * 70
    a = bootstrap_ci(vals, n_boot=500, seed=3)
    b = bootstrap_ci(vals, n_boot=500, seed=3)
    assert a == b
    assert a.mean == pytest.approx(0.3)
    assert a.low < 0.3 < a.high
    # roughly +-9 points for n=100, p=0.3
    assert 0.18 < a.low < 0.26
    assert 0.34 < a.high < 0.42


def test_wider_alpha_gives_narrower_interval():
    vals = [1.0] * 30 + [0.0] * 70
    wide = bootstrap_ci(vals, n_boot=500, alpha=0.05, seed=1)
    narrow = bootstrap_ci(vals, n_boot=500, alpha=0.5, seed=1)
    assert narrow.high - narrow.low < wide.high - wide.low


def test_empty():
    ci = bootstrap_ci([])
    assert ci.n == 0 and math.isnan(ci.mean)


def test_percentile_interpolates():
    assert _percentile([0.0, 1.0], 0.5) == 0.5
    assert _percentile([0.0, 1.0, 2.0], 0.25) == 0.5
    assert _percentile([3.0], 0.9) == 3.0
    assert _percentile([0.0, 10.0], 1.0) == 10.0
    assert _percentile([0.0, 10.0], 0.0) == 0.0
    assert math.isnan(_percentile([], 0.5))


def test_paired_diff():
    a = [0.0] * 50 + [1.0] * 50
    b = [1.0] * 100
    d = paired_bootstrap_diff(a, b, n_boot=300, seed=0)
    assert d.mean == pytest.approx(0.5)
    assert d.low > 0.3
    assert d.high <= 0.7


def test_paired_diff_length_mismatch():
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1.0], [1.0, 0.0])


def test_interval_fmt():
    assert Interval(0.5, 0.25, 0.75, 4).fmt() == "50.0 [25.0, 75.0]"
    assert Interval(0.5, 0.25, 0.75, 4).fmt(pct=False) == "0.5 [0.2, 0.8]"
