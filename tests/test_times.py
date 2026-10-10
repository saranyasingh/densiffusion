import numpy as np
import pytest

from densiffusion.data import make_windows, split_data


@pytest.mark.parametrize("axis", ["series", "time"])
@pytest.mark.parametrize("shared", [False, True])
def test_split_windows_preserve_time_origins_and_alignment(axis, shared):
    times = 10 + np.arange(20) * 0.25
    expanded = np.broadcast_to(times, (8, 20)).copy()
    if not shared:
        expanded += np.arange(8)[:, None] * 10
        times = expanded
    values = np.stack((expanded, expanded**2), axis=-1)
    splits = split_data(
        values, times=times, train_fraction=0.5, validation_fraction=0.25, axis=axis
    )
    for part, part_times in splits:
        windows, window_times = make_windows(
            part, times=part_times, window_length=3, stride=2
        )
        expected_times = np.stack(
            [
                row[start : start + 3]
                for row in part_times
                for start in range(0, part.shape[1] - 2, 2)
            ]
        )
        np.testing.assert_array_equal(window_times, expected_times)
        np.testing.assert_array_equal(windows[..., 0], window_times)
        np.testing.assert_array_equal(windows[..., 1], window_times**2)
        assert not np.shares_memory(part, values)
        assert not np.shares_memory(part_times, times)
        assert not np.shares_memory(window_times, part_times)
        assert not np.shares_memory(windows, part)
    dimension = 0 if axis == "series" else 1
    np.testing.assert_array_equal(
        np.concatenate([t for _, t in splits], axis=dimension), expanded
    )


@pytest.mark.parametrize(
    "times",
    [
        np.arange(4),
        np.zeros((3, 5)),
        np.array([0, 1, 1, 3, 4]),
        np.array([0, 1, 2, 3, np.nan]),
    ],
)
def test_invalid_window_times_are_rejected(times):
    with pytest.raises(ValueError):
        make_windows(np.zeros((2, 5, 1)), times=times, window_length=3)
