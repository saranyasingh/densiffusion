"""Finding checks and comparison tables of the densify-then-forecast benchmark."""

import importlib.util
from pathlib import Path

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
