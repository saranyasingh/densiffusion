"""Gaps of random length at random times (missing mode random_block)."""

import numpy as np
import pytest

from densiffusion.fx import build_observation_mask, random_block_observation_mask


def runs(mask, value):
    """Lengths of the runs equal to value, without the first and last, which
    the random start and the forced-observed endpoints may cut short."""
    padded = np.concatenate([[0], mask == value, [0]]).astype(int)
    edges = np.flatnonzero(np.diff(padded))
    return (edges[1::2] - edges[::2])[1:-1]


def test_runs_stay_within_bounds_and_vary():
    hidden = random_block_observation_mask(
        5000, min_gap=2, max_gap=12, min_observed=3, max_observed=9, seed=1
    )
    gaps, kept = runs(hidden, True), runs(hidden, False)
    assert gaps.min() >= 2 and gaps.max() <= 12
    assert kept.min() >= 3 and kept.max() <= 9
    assert len(set(gaps)) > 5 and len(set(kept)) > 3
    assert not hidden[0] and not hidden[-1]
    # Expected hidden fraction 7 / (7 + 6).
    assert hidden.mean() == pytest.approx(7 / 13, abs=0.03)


def test_every_hour_of_day_is_sometimes_hidden_and_sometimes_observed():
    hidden = random_block_observation_mask(24 * 200, seed=0)
    by_hour = hidden.reshape(-1, 24).mean(axis=0)
    assert by_hour.min() > 0.2 and by_hour.max() < 0.8


def test_seeded_and_dispatched():
    first = build_observation_mask(500, mode="random_block", seed=3)
    np.testing.assert_array_equal(first, random_block_observation_mask(500, seed=3))
    assert not np.array_equal(first, random_block_observation_mask(500, seed=4))


def test_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="minimum"):
        random_block_observation_mask(100, min_gap=5, max_gap=2)
