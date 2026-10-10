import numpy as np
import pytest

from densiffusion.conditioning import prepare_targets
from densiffusion.metrics import evaluate
from densiffusion.pipeline import impute


def test_explicit_mask_hides_targets_and_preserves_observations():
    class Sampler:
        def sample(self, observed, *, factor, n_samples, dt, rng, mask, times):
            assert factor == 1 and dt == 0.25
            assert np.isfinite(observed).all() and (observed[mask] == 0).all()
            np.testing.assert_array_equal(times, [[10, 10.25, 10.5, 10.75]] * 2)
            return np.repeat(np.where(mask, 1, observed)[:, None], n_samples, axis=1)

    values = np.arange(16, dtype=float).reshape(2, 4, 2)
    mask = np.array([True, False, False, True])
    observed = values.copy()
    observed[:, mask] = np.nan
    samples = impute(
        observed,
        Sampler(),
        factor=1,
        mask=mask,
        times=10 + np.arange(4) * 0.25,
        dt=0.25,
        n_samples=3,
    )
    np.testing.assert_array_equal(samples[:, 0, ~mask], values[:, ~mask])
    assert evaluate(samples, values, mask=mask)["crps"] == pytest.approx(
        np.abs(1 - values[:, mask]).mean()
    )
    assert np.isnan(observed[:, mask]).all()


def test_sampler_cannot_mutate_mask_to_change_observed_values():
    class BrokenSampler:
        def sample(self, observed, *, mask, n_samples, **kwargs):
            mask[:] = True
            return np.repeat((observed + 1)[:, None], n_samples, axis=1)

    mask = np.array([False, True, False])
    with pytest.raises(ValueError, match="preserve"):
        impute(np.zeros((1, 3, 1)), BrokenSampler(), factor=1, mask=mask)
    np.testing.assert_array_equal(mask, [False, True, False])


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros(4, dtype=bool),
        np.ones(4, dtype=bool),
        np.ones(4),
        np.zeros((2, 4), dtype=bool),
        np.array([False, True]),
    ],
)
def test_invalid_target_masks_are_rejected(mask):
    with pytest.raises(ValueError):
        prepare_targets(np.zeros((2, 4, 1)), mask)


def test_explicit_masks_keep_full_targets_and_require_complete_data():
    values = np.arange(16, dtype=float).reshape(2, 4, 2)
    mask = np.array([False, True, True, False])[None, :, None]
    target, expanded = prepare_targets(values, mask)
    np.testing.assert_array_equal(target, values)
    np.testing.assert_array_equal(expanded, np.broadcast_to(mask, values.shape))
    values[expanded] = np.nan
    with pytest.raises(ValueError, match="finite"):
        prepare_targets(values, mask)


def test_masked_imputation_rejects_grid_refinement():
    with pytest.raises(ValueError, match="factor"):
        impute(np.zeros((1, 3, 1)), None, factor=2, mask=np.array([False, True, False]))
