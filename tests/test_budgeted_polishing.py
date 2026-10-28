from dataclasses import asdict, replace

import pytest

from densiffusion.budgeted_polishing import (
    BudgetConfig,
    allocate,
    auxiliary_work,
    evaluate_run,
    ledger,
    smoke_config,
)
from densiffusion.polishing_experiment import seed_plan


@pytest.fixture
def profiles():
    # Independently verified by the matrix profiler and analytical formula.
    return {
        "forecaster": {"forward": 21158912, "training": 63433728},
        "mediator_32": {"forward": 2880, "training": 7872},
        "mediator_64": {"forward": 9856, "training": 28032},
    }


@pytest.mark.parametrize("width", [32, 64])
def test_maximal_allocation_charges_full_search_and_all_selection(profiles, width):
    config = BudgetConfig()
    plan = allocate(config, profiles, width)
    assert plan["ceiling"] == 173320595046400
    assert plan["methods"]["raw"]["epochs"] == 1200
    assert plan["methods"]["median"]["epochs"] == 1200
    learned = plan["methods"]["learned"]
    assert learned["epochs"] == 1192
    assert learned["noise_validation_count"] == 120
    assert learned["optimizer_updates"] == 4768
    assert learned["validation_epochs"] == [10, 50, 100, 200, 400, 800, 1192]
    assert learned["work"]["total"] <= plan["ceiling"]
    assert learned["unused_fraction"] < 0.0001
    assert (
        ledger(config, profiles["forecaster"], 1193, plan["auxiliary"]["total"])[
            "total"
        ]
        > plan["ceiling"]
    )
    assert all(len(p["validation_epochs"]) == 7 for p in plan["methods"].values())
    assert all(p["noise_validation_count"] == 120 for p in plan["methods"].values())


def test_auxiliary_ledger_includes_losing_candidates_and_reconstruction(profiles):
    work = auxiliary_work(BudgetConfig(), profiles, 64)
    assert work == {
        "candidate_training": 117650227200,
        "candidate_validation": 4590665728,
        "train_validation_preprocessing": 161480704,
        "total": 122402373632,
    }


def test_final_partial_monitor_interval_is_charged(profiles):
    config = smoke_config()
    plan = allocate(config, profiles, 64)
    assert plan["methods"]["learned"]["epochs"] == 19
    assert plan["methods"]["learned"]["noise_validation_count"] == 2
    assert plan["methods"]["raw"]["noise_validation_count"] == 2


def test_new_streams_do_not_reuse_previous_experiment():
    streams = [
        v for s in (29101, 29102, 29103) for v in seed_plan(s, 2026102801).values()
    ]
    old = [v for s in (28101, 28102, 28103) for v in seed_plan(s, 2026102705).values()]
    assert len(streams) == len(set(streams))
    assert not set(streams) & set(old)


@pytest.mark.parametrize(
    "changes",
    [
        {"samples": 32},
        {"base_epochs": 800},
        {"namespace": 2026102705},
        {"validation_epochs_prefix": (50, 100, 200, 400, 800)},
        {"n_test": 1024},
        {"seeds": (29101,)},
        {"context_length": 25},
    ],
)
def test_scientific_design_guard(changes):
    with pytest.raises(ValueError):
        BudgetConfig(**{**asdict(BudgetConfig()), **changes})


def test_smoke_namespace_guard():
    with pytest.raises(ValueError):
        replace(smoke_config(), namespace=2026102801)


def test_evaluation_cannot_generate_test_before_global_freeze(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Test generator must not be reached")

    monkeypatch.setattr(
        "densiffusion.budgeted_polishing.generate_noisy_periodic", forbidden
    )
    with pytest.raises(FileNotFoundError):
        evaluate_run(tmp_path)


def test_changed_frozen_artifact_stops_evaluation(tmp_path, monkeypatch):
    import hashlib
    import json

    config = tmp_path / "config.json"
    config.write_text("{}")
    (tmp_path / "FROZEN.json").write_text(
        json.dumps(
            {"files": {"config.json": hashlib.sha256(config.read_bytes()).hexdigest()}}
        )
    )
    config.write_text('{"changed": true}')
    monkeypatch.setattr(
        "densiffusion.budgeted_polishing.generate_noisy_periodic",
        lambda *a, **kw: pytest.fail("Reached test generator"),
    )
    with pytest.raises(ValueError, match="Frozen artifact changed"):
        evaluate_run(tmp_path)


def test_sampler_conditions_only_on_supplied_processed_history(monkeypatch):
    import numpy as np

    from densiffusion.budgeted_polishing import _sample
    from densiffusion.training import TrainingConfig

    processed = np.full((2, 32, 1), 0.123)
    calls = []

    class Sampler:
        def __init__(self, model, config):
            assert config.batch_size == 32

        def sample(self, observed, dense_history, **kwargs):
            assert observed is processed and dense_history is processed
            assert kwargs["horizon"] == 8 and kwargs["dt"] == 1
            calls.append(kwargs)
            return np.zeros((2, 4, 8, 1))

    monkeypatch.setattr("densiffusion.budgeted_polishing.DDPMForecastSampler", Sampler)
    samples = _sample(
        None,
        processed,
        {"experiment": asdict(smoke_config())},
        TrainingConfig(diffusion_steps=2),
        100,
    )
    assert samples.shape == (2, 4, 8, 1) and len(calls) == 1
