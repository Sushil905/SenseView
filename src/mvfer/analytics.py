"""Transparent, dependency-light session descriptors for FER sequences."""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence

from . import EXPRESSIONS


def _coarse_grain(sequence: Sequence[int], scale: int) -> list[int]:
    """Majority label in each non-overlapping window; ignores an incomplete tail."""
    if scale < 1:
        raise ValueError("scale must be >= 1")
    return [Counter(sequence[i : i + scale]).most_common(1)[0][0]
            for i in range(0, len(sequence) - scale + 1, scale)]


def shannon_entropy(sequence: Sequence[int], n_classes: int = len(EXPRESSIONS)) -> float:
    if not sequence:
        return 0.0
    counts = Counter(sequence)
    n = len(sequence)
    return -sum((count / n) * math.log2(count / n) for count in counts.values())


def multiscale_entropy(sequence: Sequence[int], scales: Iterable[int] = (1, 2, 4, 8)) -> dict[str, float]:
    """Categorical multiscale entropy, normalized to [0, 1].

    This is a reproducible proxy for multiscale sequence complexity. Replace it
    with the study's exact multidimensional estimator only after validating its
    implementation against the original methodological details.
    """
    norm = math.log2(len(EXPRESSIONS))
    return {f"entropy_s{scale}": (shannon_entropy(_coarse_grain(sequence, scale)) / norm
                                    if len(sequence) >= scale else 0.0)
            for scale in scales}


def session_features(predictions: Sequence[int], confidences: Sequence[float] | None = None) -> dict[str, float]:
    """Build descriptive, not diagnostic, features from argmax expression IDs."""
    if any(label < 0 or label >= len(EXPRESSIONS) for label in predictions):
        raise ValueError("prediction IDs must be valid expression indices")
    n = len(predictions)
    result = {f"prop_{name}": (predictions.count(i) / n if n else 0.0)
              for i, name in enumerate(EXPRESSIONS)}
    result["n_frames"] = float(n)
    result["transition_rate"] = (sum(a != b for a, b in zip(predictions, predictions[1:])) / (n - 1)
                                 if n > 1 else 0.0)
    result.update(multiscale_entropy(predictions))
    if confidences is not None:
        if len(confidences) != n:
            raise ValueError("confidences and predictions must have equal length")
        result["mean_confidence"] = sum(confidences) / n if n else 0.0
    return result
