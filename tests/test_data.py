import json
from dataclasses import replace

import numpy as np
import pytest

from densiffusion.cli import main
from densiffusion.data import (
    GenerationConfig,
    generate,
    make_mask,
    mask_data,
    save_dataset,
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


def test_mask_data_hides_every_other_time_step():
    values = np.arange(12, dtype=float).reshape(3, 4, 1)
    masked, mask = mask_data(values, axis="time", step=2, offset=1)

    expected_mask = np.array([False, True, False, True])
    np.testing.assert_array_equal(mask, expected_mask)
    np.testing.assert_array_equal(masked[:, ~mask, :], values[:, ~mask, :])
    assert np.isnan(masked[:, mask, :]).all()
    np.testing.assert_array_equal(make_mask(4, step=2, offset=1), expected_mask)


def test_random_mask_uses_probability_and_seed():
    expected_mask = np.random.default_rng(7).random(8) < 0.4
    mask = make_mask(8, random=True, probability=0.4, rng=7)
    np.testing.assert_array_equal(mask, expected_mask)

    values = np.arange(24, dtype=float).reshape(3, 8, 1)
    masked, mask = mask_data(values, axis="time", random=True, probability=0.4, rng=7)
    np.testing.assert_array_equal(mask, expected_mask)
    np.testing.assert_array_equal(masked[:, ~mask, :], values[:, ~mask, :])
    assert np.isnan(masked[:, mask, :]).all()


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
