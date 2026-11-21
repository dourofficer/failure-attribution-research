"""Metrics for the probe layer sweep.

EXAMPLE: this exists to show what "core algorithm code" looks like
and what its tests look like (`tests/test_metrics.py`). Delete with the rest of
the example.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUROC via the Mann-Whitney rank statistic.

    Ties contribute 0.5, matching the pairwise definition. O(n log n) rather
    than the O(n^2) pairwise form, which is what makes it worth testing against
    that form (see `tests/test_metrics.py`).
    """
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=float)
    if labels.shape != scores.shape:
        raise ValueError(f"shape mismatch: {labels.shape} vs {scores.shape}")

    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC is undefined when one class is empty")

    ranks = rankdata(scores)  # average ranks for ties
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def mean_difference_direction(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Unit vector pointing from the negative class mean to the positive one.

    Sign convention: projecting `X` onto the returned vector must give a higher
    mean for the positive class. Silent sign inversions are a classic source of
    contaminated downstream results, so the convention is asserted in the tests.
    """
    labels = np.asarray(labels).astype(bool)
    X = np.asarray(X, dtype=float)
    if X.shape[0] != labels.shape[0]:
        raise ValueError(f"shape mismatch: {X.shape[0]} rows vs {labels.shape[0]} labels")
    if labels.all() or not labels.any():
        raise ValueError("both classes must be present")

    delta = X[labels].mean(axis=0) - X[~labels].mean(axis=0)
    norm = np.linalg.norm(delta)
    if norm == 0:
        raise ValueError("class means coincide; direction is undefined")
    return delta / norm
