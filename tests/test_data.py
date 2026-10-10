import json
from dataclasses import replace

import numpy as np
import pytest

from densiffusion.cli import main
from densiffusion.data import (
    GenerationConfig,
    generate,
    make_windows,
    save_dataset,
    split_data,
)


def stub_generator(*, n_series, n_steps, dt, rng, offset=0.0):
    return rng.normal(size=(n_series, n_steps, 1)) + offset


@pytest.fixture
def config():
    return GenerationConfig(
        generator=f"{__name__}:stub_generator",
        n_series=3,
        n_steps=8,
        dt=0.5,
        seed=42,
        params={"offset": 2.0},
    )


def test_generator_receives_parameters_and_seed(config):
    expected = np.random.default_rng(42).normal(size=(3, 8, 1)) + 2.0
    np.testing.assert_array_equal(generate(config), expected)
    np.testing.assert_array_equal(generate(config), generate(config))
    assert not np.array_equal(generate(config), generate(replace(config, seed=43)))


def test_dataset_round_trip_and_no_overwrite(tmp_path, config):
    path = tmp_path / "data" / "sample.npz"
    values = generate(config)
    save_dataset(path, values, config)
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["values"], values)
        np.testing.assert_array_equal(saved["times"], np.arange(8) * 0.5)
        assert GenerationConfig(**json.loads(saved["config"].item())) == config
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_dataset(path, values + 1, config)
    assert path.read_bytes() == original


@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt])
def test_failed_dataset_write_is_removed(tmp_path, config, monkeypatch, error_type):
    def fail_write(output, **kwargs):
        output.write(b"partial data")
        raise error_type("interrupted write")

    monkeypatch.setattr("densiffusion.data.np.savez_compressed", fail_write)
    path = tmp_path / "sample.npz"
    with pytest.raises(error_type, match="interrupted write"):
        save_dataset(path, generate(config), config)
    assert not path.exists()


@pytest.mark.parametrize("bad_values", [np.zeros((3, 8)), np.full((3, 8, 1), np.nan)])
def test_invalid_generator_output_is_rejected(monkeypatch, config, bad_values):
    monkeypatch.setattr(f"{__name__}.stub_generator", lambda **kwargs: bad_values)
    with pytest.raises(ValueError):
        generate(config)


@pytest.mark.parametrize(
    "changes",
    [
        {"n_steps": 0},
        {"n_series": 1.5},
        {"dt": float("inf")},
        {"seed": -1},
        {"generator": "invalid"},
        {"params": {"rng": 1}},
    ],
)
def test_invalid_config_is_rejected(config, changes):
    with pytest.raises(ValueError):
        replace(config, **changes)


def test_generation_cli(tmp_path, capsys):
    config_path = tmp_path / "generate.toml"
    config_path.write_text(
        f'generator = "{__name__}:stub_generator"\n'
        "n_series = 2\nn_steps = 5\nseed = 7\ndt = 0.25\n"
        "[params]\noffset = 4.0\n"
    )
    output = tmp_path / "sample.npz"
    arguments = ["generate", str(config_path), "--output", str(output)]
    main(arguments)
    with np.load(output, allow_pickle=False) as saved:
        np.testing.assert_array_equal(
            saved["values"], np.random.default_rng(7).normal(size=(2, 5, 1)) + 4.0
        )
    with pytest.raises(SystemExit, match="2"):
        main(arguments)
    assert "Output already exists" in capsys.readouterr().err


def test_missing_plugin_does_not_create_output(tmp_path, capsys):
    config_path = tmp_path / "generate.toml"
    config_path.write_text(
        'generator = "missing_densiffusion_plugin:generate"\n'
        "n_series = 1\nn_steps = 2\n"
    )
    output = tmp_path / "sample.npz"
    with pytest.raises(SystemExit, match="2"):
        main(["generate", str(config_path), "--output", str(output)])
    assert "missing_densiffusion_plugin" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("axis, dimension", [("series", 0), ("time", 1)])
def test_splits_preserve_order_and_isolate_partitions(axis, dimension):
    values = np.arange(100).reshape(10, 10, 1)
    parts = split_data(values, train_fraction=0.6, validation_fraction=0.2, axis=axis)
    assert [part.shape[dimension] for part in parts] == [6, 2, 2]
    np.testing.assert_array_equal(np.concatenate(parts, axis=dimension), values)
    for part in parts:
        assert not np.shares_memory(part, values)


def test_windows_preserve_order_without_sharing_storage():
    values = np.arange(2 * 9 * 2).reshape(2, 9, 2)
    windows = make_windows(values, window_length=5, stride=2)
    expected = np.stack(
        [series[start : start + 5] for series in values for start in (0, 2, 4)]
    )
    np.testing.assert_array_equal(windows, expected)
    windows[0, 2] = -1
    np.testing.assert_array_equal(windows[1, 0], values[0, 2])
    assert not np.shares_memory(windows, values)


def test_time_split_windows_do_not_cross_boundaries():
    values = np.arange(20).reshape(1, 20, 1)
    parts = split_data(
        values, train_fraction=0.5, validation_fraction=0.25, axis="time"
    )
    windows = [make_windows(part, window_length=4) for part in parts]
    for window, part in zip(windows, parts, strict=True):
        assert window.min() == part.min()
        assert window.max() == part.max()
    assert windows[0].max() < windows[1].min()
    assert windows[1].max() < windows[2].min()


@pytest.mark.parametrize(
    "train, validation", [(0, 0.2), (0.8, 0.2), (np.nan, 0.1), (0.6, 0.01)]
)
def test_split_rejects_invalid_or_empty_partitions(train, validation):
    with pytest.raises(ValueError):
        split_data(
            np.zeros((10, 10, 1)), train_fraction=train, validation_fraction=validation
        )


@pytest.mark.parametrize("length, stride", [(0, 1), (2.5, 1), (3, 0), (7, 1)])
def test_windows_reject_invalid_lengths(length, stride):
    with pytest.raises(ValueError):
        make_windows(np.zeros((2, 6, 1)), window_length=length, stride=stride)
