import numpy as np
import pytest

from densiffusion.pipeline import impute


class StubDensifier:
    def __init__(self):
        self.calls = []

    def sample(self, observed, *, factor, n_samples, dt, rng):
        self.calls.append((observed.copy(), dt))
        batch, steps, channels = observed.shape
        dense = rng.normal(size=(batch, n_samples, (steps - 1) * factor + 1, channels))
        dense[:, :, ::factor, :] = observed[:, None, :, :]
        return dense


@pytest.mark.parametrize("factor", [1, 2, 3])
def test_imputation_preserves_observations_and_grid(factor):
    observed = np.arange(12, dtype=float).reshape(2, 3, 2)
    sampler = StubDensifier()
    samples = impute(observed, sampler, factor=factor, n_samples=3, dt=0.5)
    assert samples.shape == (2, 3, 2 * factor + 1, 2)
    for sample in np.moveaxis(samples, 1, 0):
        np.testing.assert_array_equal(sample[:, ::factor], observed)
    np.testing.assert_array_equal(sampler.calls[0][0], observed)
    assert sampler.calls[0][1] == 0.5


def test_densifier_cannot_change_observations():
    class BrokenDensifier(StubDensifier):
        def sample(self, observed, **kwargs):
            observed += 1
            return super().sample(observed, **kwargs)

    observed = np.zeros((2, 3, 1))
    with pytest.raises(ValueError, match="preserve"):
        impute(observed, BrokenDensifier())
    np.testing.assert_array_equal(observed, 0)


@pytest.mark.parametrize(
    "invalid", [np.zeros((2, 1, 3, 1)), np.full((2, 1, 5, 1), np.nan)]
)
def test_invalid_samples_are_rejected(invalid):
    class BrokenDensifier:
        def sample(self, observed, **kwargs):
            return invalid

    with pytest.raises(ValueError):
        impute(np.zeros((2, 3, 1)), BrokenDensifier())


def test_sampling_handles_reused_buffers():
    class BufferedDensifier:
        def __init__(self):
            self.buffer = np.zeros((1, 1, 5, 1))

        def sample(self, observed, **kwargs):
            self.buffer[:, :, 1::2] += 1
            return self.buffer

    sampler = BufferedDensifier()
    observed = np.zeros((1, 3, 1))
    first = impute(observed, sampler)
    second = impute(observed, sampler)
    np.testing.assert_array_equal(first[0, 0, :, 0], [0, 1, 0, 1, 0])
    np.testing.assert_array_equal(second[0, 0, :, 0], [0, 2, 0, 2, 0])


def test_sampling_seed_reproducibility():
    observed = np.zeros((2, 3, 1))
    sampler = StubDensifier()
    baseline = impute(observed, sampler, seed=42)
    np.testing.assert_array_equal(baseline, impute(observed, sampler, seed=42))
    assert not np.array_equal(baseline, impute(observed, sampler, seed=43))
