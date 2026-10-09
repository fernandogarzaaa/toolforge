"""Bootstrap confidence intervals for accuracy numbers."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Interval:
    mean: float
    low: float
    high: float
    n: int

    def fmt(self, pct: bool = True) -> str:
        k = 100.0 if pct else 1.0
        return f"{self.mean * k:.1f} [{self.low * k:.1f}, {self.high * k:.1f}]"


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def bootstrap_ci(values: Sequence[float], n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> Interval:
    """Percentile bootstrap CI of the mean of ``values`` (e.g. 0/1 correctness)."""
    n = len(values)
    if n == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0)
    rng = random.Random(seed)
    mean = sum(values) / n
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return Interval(mean, _percentile(means, alpha / 2), _percentile(means, 1 - alpha / 2), n)


def paired_bootstrap_diff(
    a: Sequence[float], b: Sequence[float], n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> Interval:
    """CI of mean(b - a) over the same examples (paired resampling)."""
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    return bootstrap_ci([y - x for x, y in zip(a, b)], n_boot=n_boot, alpha=alpha, seed=seed)
