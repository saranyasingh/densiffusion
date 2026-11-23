"""Finding checks and comparison tables of the densify-then-forecast benchmark."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

_spec = importlib.util.spec_from_file_location(
    "run_densiffusion", Path(__file__).parents[1] / "scripts" / "run_densiffusion.py"
)
benchmark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(benchmark)

INTERVALS = {1: [0.01, 0.02], 0: [-0.01, 0.01], -1: [-0.02, -0.01]}


def flat_sweep(changes=None):
    """Every method scores like sparse (1.5x the GP) unless changed.

    changes maps (method, P/dt) to (CRPS ratio, significance against sparse).
    """
    changes = changes or {}
    sweep = []
    for ratio in (1.6, 2.5, 3, 4, 6, 8, 16):
        forecast = {"gp": {"crps": 2.0}, "sparse": {"crps": 3.0}}
        for method in benchmark.QUASI_METHODS[2:]:
            crps, sign = changes.get((method, ratio), (1.5, 0))
            forecast[method] = {
                "crps": 2.0 * crps,
                "gain_over_sparse_95": INTERVALS[sign],
            }
        history = {"x8": {"model": 3.0, "gp": 2.0}}
        sweep.append(
            {
                "gp": {"period": ratio},
                "dt": 1.0,
                "forecast": forecast,
                "densified_history": history,
                "densified_history_fine": history,
            }
        )
    return sweep


def verdicts(sweep):
    return [verdict for _, verdict, _ in benchmark.finding_checks(sweep)]


def test_flat_sweep_fails_the_gain_findings():
    assert verdicts(flat_sweep()) == [
        "no",
        "no",
        "yes",
        "no",
        "yes",
        "no",
        "no",
        "partly (1/2)",
    ]


def test_partial_gain_is_reported_with_its_size():
    sweep = flat_sweep({("fine_x2", 2.5): (1.25, 1), ("fine_x2", 3): (1.45, 0)})
    _, verdict, evidence = benchmark.finding_checks(sweep)[0]
    assert verdict == "partly (1/3)"
    assert evidence == "50%, 10% (ns), 0% (ns)"


def test_write_comparison_has_one_column_per_model(tmp_path):
    sweeps = {"pristi": flat_sweep(), "sssd": flat_sweep()}
    benchmark.write_comparison(sweeps, tmp_path)
    text = (tmp_path / "comparison.md").read_text()
    assert "| Finding | PriSTI | SSSD |" in text
    assert "| 2.5 | 1.500 | 1.500 |" in text
    assert "| 16 | 1.50 / 1.50 | 1.50 / 1.50 |" in text


def test_phase_shifted_windows_cover_every_offset():
    paths = np.arange(257.0)[None]
    windows = benchmark.phase_shifted_windows(paths)
    assert windows.shape == (16, 13)
    np.testing.assert_array_equal(windows[0], np.arange(0, 97, 8))
    np.testing.assert_array_equal(windows[3], np.arange(3, 100, 8))
    np.testing.assert_array_equal(windows[8], np.arange(160, 257, 8))
    # Shifted windows that would overrun the path move back one coarse step.
    np.testing.assert_array_equal(windows[9], np.arange(153, 250, 8))


def test_method_seed_depends_only_on_base_and_name():
    first = benchmark.method_seed(42, "interp_x2")
    assert first == benchmark.method_seed(42, "interp_x2")
    assert first != benchmark.method_seed(42, "sparse_shifted")
    assert first != benchmark.method_seed(43, "interp_x2")


def test_controls_are_skipped_when_not_run():
    rows = benchmark.control_checks(flat_sweep())
    assert [verdict for _, verdict, _ in rows] == ["n/a"] * len(benchmark.CONTROLS)


def test_paired_gain_averages_seeds():
    rng = np.random.default_rng(0)
    windows = len(benchmark.WINDOW_STARTS) * 6
    future = rng.normal(size=(windows, benchmark.HORIZON))
    run = {
        "windows": np.concatenate(
            [np.zeros((windows, benchmark.HISTORY)), future], axis=1
        ),
        "sparse": future[:, None] + rng.normal(size=(windows, 10, benchmark.HORIZON)),
        "interp_x2": future[:, None]
        + 0.5 * rng.normal(size=(windows, 10, benchmark.HORIZON)),
    }
    single = benchmark.paired_gain(run, "sparse", "interp_x2")
    repeated = benchmark.paired_gain([run, run], "sparse", "interp_x2")
    assert single[0] > 0
    np.testing.assert_allclose(repeated[0], single[0])
    assert repeated[1] <= repeated[0] <= repeated[2]
