import pytest

from densiffusion.resource_allocation import (
    analytical_csdi_forward_work,
    best_within_budget,
    budget_table,
    count_csdi_work,
    count_mediator_work,
    fitting_ledger,
)


def test_budget_selection_charges_intervening_failed_candidates():
    rows = [
        {"epoch": 10, "crps": 0.3, "cost": 100},
        {"epoch": 20, "crps": 0.4, "cost": 200},
        {"epoch": 30, "crps": 0.1, "cost": 300},
    ]
    assert best_within_budget(rows, 99, "cost") is None
    selected = best_within_budget(rows, 250, "cost")
    assert selected["epoch"] == 10
    assert selected["spent_cost"] == 200
    assert selected["unused_budget"] == 50
    assert selected["validation_opportunities"] == 2
    assert best_within_budget(rows, 300, "cost")["epoch"] == 30


def test_budget_table_never_averages_only_affordable_seeds():
    traces = {
        "a": {"raw": [{"epoch": 1, "crps": 1, "cost": 1}]},
        "b": {"raw": [{"epoch": 1, "crps": 2, "cost": 2}]},
    }
    rows = budget_table(traces, ["raw"], [1, 2], "cost")
    assert rows[0]["methods"]["raw"]["mean_crps"] is None
    assert rows[1]["methods"]["raw"]["mean_crps"] == 1.5


def test_ledger_charges_setup_and_validation_once():
    ledger = fitting_ledger(
        epochs=10,
        n_train=20,
        n_validation=5,
        forward=2,
        training=5,
        noise_validation_count=2,
        noise_validation_steps=3,
        forecast_validation_count=4,
        forecast_samples=6,
        diffusion_steps=7,
        setup=11,
        extra_per_forecast_validation=13,
    )
    assert ledger == {
        "setup": 11,
        "optimization": 1000,
        "noise_validation": 60,
        "forecast_validation": 1732,
        "total": 2803,
    }


def test_work_counts_batch_linearity_and_attention():
    architecture = dict(
        n_channels=1,
        hidden_channels=8,
        residual_layers=1,
        attention_heads=2,
        diffusion_embedding_dim=8,
        time_embedding_dim=8,
        feature_embedding_dim=4,
        feedforward_dim=8,
        dropout=0.1,
    )
    short = count_csdi_work(architecture, 5, batch_size=2)
    assert short == count_csdi_work(architecture, 5, batch_size=4)
    assert short["forward_operations"]["aten.bmm"] == 4 * 5 * 5 * 8
    longer = count_csdi_work(architecture, 9)
    assert short["forward"] == analytical_csdi_forward_work(architecture, 5)
    assert longer["forward"] == analytical_csdi_forward_work(architecture, 9)
    assert longer["forward"] > short["forward"]
    assert short["training"] > short["forward"]


def test_mediator_profile_matches_three_dense_layers():
    result = count_mediator_work(32)
    assert result["forward"] == 2 * (12 * 32 + 32 * 32 + 32)
    # First-layer input gradients are unnecessary.
    assert result["training"] == 3 * result["forward"] - 2 * 12 * 32


@pytest.mark.parametrize("budget", [-1, float("nan"), float("inf")])
def test_invalid_budget(budget):
    with pytest.raises(ValueError):
        best_within_budget([], budget, "cost")


def test_out_of_order_costs_rejected():
    with pytest.raises(ValueError):
        best_within_budget(
            [
                {"epoch": 1, "crps": 1, "cost": 2},
                {"epoch": 2, "crps": 1, "cost": 1},
            ],
            3,
            "cost",
        )
