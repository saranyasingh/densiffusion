import numpy as np
import pytest

from densiffusion.metrics import crps, evaluate


@pytest.mark.parametrize("n_samples", [1, 2, 7])
def test_crps_matches_pairwise_definition(n_samples):
    rng = np.random.default_rng(42)
    target = rng.normal(size=(3, 5, 2))
    samples = rng.normal(size=(3, n_samples, 5, 2))
    expected = np.abs(samples - target[:, None]).mean(axis=1) - 0.5 * np.abs(
        samples[:, :, None] - samples[:, None, :]
    ).mean(axis=(1, 2))
    mask = np.zeros(target.shape, dtype=bool)
    mask[:, 1::2] = True
    assert crps(samples, target) == pytest.approx(expected.mean())
    assert evaluate(samples, target, mask=mask) == pytest.approx(
        {"crps": expected[mask].mean()}
    )
    assert crps(samples[:, ::-1], target) == pytest.approx(expected.mean())


def test_crps_known_values_and_observation_exclusion():
    target = np.array([0.0, 1.0, 0.0]).reshape(1, 3, 1)
    samples = np.array([[100.0, 0.0, 100.0], [100.0, 2.0, 100.0]]).reshape(1, 2, 3, 1)
    mask = np.array([False, True, False]).reshape(1, 3, 1)
    assert crps(samples, target, mask=mask) == 0.5
    assert crps(samples + 1e12, target + 1e12, mask=mask) == 0.5
    assert crps(target[:, None], target) == 0.0
    assert crps(target[:, None] + 2, target) == 2.0


@pytest.mark.parametrize(
    "mask",
    [np.zeros((1, 3, 1), dtype=bool), np.ones((1, 3, 1)), np.ones((1, 3), dtype=bool)],
)
def test_crps_rejects_invalid_masks(mask):
    with pytest.raises(ValueError, match="mask"):
        crps(np.zeros((1, 2, 3, 1)), np.zeros((1, 3, 1)), mask=mask)


@pytest.mark.parametrize(
    "samples",
    [np.zeros((1, 0, 3, 1)), np.zeros((1, 2, 4, 1)), np.full((1, 2, 3, 1), np.nan)],
)
def test_crps_rejects_invalid_samples(samples):
    with pytest.raises(ValueError):
        crps(samples, np.zeros((1, 3, 1)))
