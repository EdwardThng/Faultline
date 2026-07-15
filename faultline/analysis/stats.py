"""Small statistics helpers.

Cells in the leaderboard are small (a handful of runs), so point estimates
are reported with Wilson 95% confidence intervals rather than bare rates.
"""

from __future__ import annotations

import math


def wilson_interval(
    successes: int, n: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (default 95%)."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    mid = len(s) // 2
    if len(s) % 2:
        return float(s[mid])
    return (s[mid - 1] + s[mid]) / 2.0
