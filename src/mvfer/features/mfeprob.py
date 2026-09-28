"""Summaries of multi-view expression probabilities.

These are descriptive model outputs and must not be interpreted as clinical
or autism-related labels.
"""

from __future__ import annotations

from collections.abc import Sequence

from .. import EXPRESSIONS


def mean_expression_probabilities(probabilities: Sequence[Sequence[float]]) -> dict[str, float]:
    """Return each expression's mean model probability across frames.

    Input rows must follow ``mvfer.EXPRESSIONS`` order and contain seven
    nonnegative values. Rows are normalized before averaging.
    """
    if not probabilities:
        return {name: 0.0 for name in EXPRESSIONS}
    totals = [0.0] * len(EXPRESSIONS)
    for row in probabilities:
        if len(row) != len(EXPRESSIONS) or any(value < 0 for value in row):
            raise ValueError(f"each probability row must contain {len(EXPRESSIONS)} nonnegative values")
        denominator = sum(row)
        if denominator <= 0:
            raise ValueError("each probability row must have a positive sum")
        for index, value in enumerate(row):
            totals[index] += value / denominator
    count = len(probabilities)
    return {name: totals[index] / count for index, name in enumerate(EXPRESSIONS)}
