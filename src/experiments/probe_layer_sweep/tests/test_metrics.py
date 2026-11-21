"""Tests for the sweep metrics.

EXAMPLE: the patterns here are the ones worth copying:

  * differential   — the fast implementation against a slow, obviously-correct one
  * known-answer   — cases where the result is derivable by hand
  * invariants     — properties that must hold whatever the data
  * sign convention — the direction points the way we claim it does

Test the core algorithm (metrics, projections, aggregation, loaders, span and
label derivation); skip plotting, argparse, and thin wrappers. Write the test
when the code's output first enters a worklog.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.experiments.probe_layer_sweep.metrics import auroc, mean_difference_direction


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# --- differential: fast implementation vs the slow obvious one ----------------

def auroc_pairwise(labels, scores):
    """O(n^2) definition: P(score_pos > score_neg), ties counting half."""
    pos = np.asarray(scores)[np.asarray(labels).astype(bool)]
    neg = np.asarray(scores)[~np.asarray(labels).astype(bool)]
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (pos.size * neg.size))


@pytest.mark.parametrize("n", [10, 50, 200])
def test_auroc_matches_pairwise_definition(rng, n):
    labels = rng.integers(0, 2, n).astype(bool)
    labels[0], labels[1] = True, False  # guarantee both classes
    scores = rng.normal(size=n)
    assert auroc(labels, scores) == pytest.approx(auroc_pairwise(labels, scores))


def test_auroc_matches_pairwise_definition_with_heavy_ties(rng):
    """Ties are where rank-based and pairwise forms most easily diverge."""
    labels = np.array([True, False] * 25)
    scores = rng.integers(0, 3, 50).astype(float)
    assert auroc(labels, scores) == pytest.approx(auroc_pairwise(labels, scores))


# --- known-answer cases -------------------------------------------------------

def test_auroc_perfect_separation_is_one():
    assert auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)


def test_auroc_perfect_inversion_is_zero():
    assert auroc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(0.0)


def test_auroc_constant_scores_is_chance():
    assert auroc([0, 1, 0, 1], [0.5] * 4) == pytest.approx(0.5)


# --- invariants ---------------------------------------------------------------

def test_auroc_label_flip_sums_to_one(rng):
    labels = np.array([True, False] * 20)
    scores = rng.normal(size=40)
    assert auroc(labels, scores) + auroc(~labels, scores) == pytest.approx(1.0)


def test_auroc_invariant_to_monotone_transform(rng):
    """Rank-based, so any strictly increasing map of the scores must not move it."""
    labels = np.array([True, False] * 20)
    scores = rng.normal(size=40)
    assert auroc(labels, np.exp(scores)) == pytest.approx(auroc(labels, scores))


def test_auroc_invariant_to_example_order(rng):
    labels = np.array([True, False] * 20)
    scores = rng.normal(size=40)
    perm = rng.permutation(40)
    assert auroc(labels[perm], scores[perm]) == pytest.approx(auroc(labels, scores))


def test_auroc_rejects_single_class():
    with pytest.raises(ValueError, match="undefined"):
        auroc([1, 1, 1], [0.1, 0.2, 0.3])


# --- sign convention ----------------------------------------------------------

def test_direction_points_from_negative_to_positive(rng):
    X = rng.normal(size=(200, 8))
    labels = rng.integers(0, 2, 200).astype(bool)
    X[labels] += 2.0  # positives genuinely shifted

    d = mean_difference_direction(X, labels)
    proj = X @ d
    assert proj[labels].mean() > proj[~labels].mean()
    assert np.linalg.norm(d) == pytest.approx(1.0)


def test_direction_flips_when_labels_flip(rng):
    X = rng.normal(size=(100, 5))
    labels = rng.integers(0, 2, 100).astype(bool)
    labels[0], labels[1] = True, False
    assert mean_difference_direction(X, labels) == pytest.approx(
        -mean_difference_direction(X, ~labels)
    )


# --- opt-in tiers -------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.slow
def test_real_model_layer_extraction():
    """Example of a test kept out of the default loop.

    Run with `uv run pytest -m gpu`; the default run excludes it so the fast
    suite stays usable after every change.
    """
    pytest.skip("reference example — replace with a real GPU check")
