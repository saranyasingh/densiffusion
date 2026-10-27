import numpy as np
import pytest

from densiffusion.fred import align, fx_windows, parse_fred_csv


def test_parse_skips_holidays_and_keeps_values():
    text = (
        "observation_date,DEXUSEU\n"
        "1999-01-04,1.1812\n"
        "1999-01-05,\n"
        "1999-01-06,.\n"
        "1999-01-07,1.1636\n"
    )
    assert parse_fred_csv(text) == {"1999-01-04": 1.1812, "1999-01-07": 1.1636}


def test_parse_rejects_non_positive_rates():
    with pytest.raises(ValueError, match="positive"):
        parse_fred_csv("observation_date,X\n1999-01-04,0\n")


def test_align_keeps_common_dates_from_start_in_order():
    series = {
        "A": {"1998-12-31": 9.0, "1999-01-05": 2.0, "1999-01-04": 1.0},
        "B": {"1999-01-04": 3.0, "1999-01-05": 4.0, "1999-01-06": 5.0},
    }
    dates, values = align(series, start="1999-01-01")
    assert dates == ["1999-01-04", "1999-01-05"]
    np.testing.assert_array_equal(values, [[1.0, 2.0], [3.0, 4.0]])


def random_walks(n_series=3, length=200, seed=0):
    rng = np.random.default_rng(seed)
    scales = np.array([0.5, 1.0, 2.0])[:n_series, None]
    return np.cumsum(rng.normal(size=(n_series, length)) * scales, axis=1)


def test_windows_split_by_time_without_overlap():
    splits = fx_windows(random_walks(), window_length=10, stride=3)
    train, validation, test = (
        splits[k]["times"] for k in ("train", "validation", "test")
    )
    assert train.max() < validation.min() and validation.max() < test.min()
    for split in (splits[k] for k in ("train", "validation", "test")):
        assert split["values"].shape[1:] == (10, 1)
        assert split["times"].shape == split["values"].shape[:2]
        assert len(split["series"]) == len(split["values"])


def test_windows_start_at_zero_in_training_scale_units():
    log_prices = random_walks()
    splits = fx_windows(log_prices, window_length=10, stride=1)
    train = splits["train"]
    np.testing.assert_array_equal(train["values"][:, 0], 0)
    # Daily moves in the training period have unit standard deviation per series.
    for series in range(3):
        rows = train["values"][train["series"] == series, :, 0]
        steps = np.diff(rows, axis=1)
        assert steps.std() == pytest.approx(1.0, rel=0.05)


def test_scales_use_the_training_period_only():
    log_prices = random_walks()
    changed = log_prices.copy()
    changed[:, -40:] *= 10  # rewrite the validation and test periods
    original = fx_windows(log_prices, window_length=10, stride=1)
    rewritten = fx_windows(changed, window_length=10, stride=1)
    np.testing.assert_array_equal(original["scales"], rewritten["scales"])
    np.testing.assert_array_equal(
        original["train"]["values"], rewritten["train"]["values"]
    )
