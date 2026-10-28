"""Explicit work ledgers and retrospective, budget-constrained selection.

Work means counted dense matrix FLOPs (multiply plus add = two operations),
not total hardware instructions, energy, or dollars. Uncounted operations and
measured elapsed times must be reported separately. A saved-trace frontier is
descriptive validation evidence, not a new held-out or compute-optimal result.
"""

import math

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.flop_counter import FlopCounterMode

from densiffusion.architectures.csdi import CSDIDenoiser
from densiffusion.history_polishing import BlindSpotMediator


def _positive(name, value, allow_zero=False):
    if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"Invalid {name}")


def analytical_csdi_forward_work(architecture, length):
    """Independent matrix-product derivation for this univariate architecture.

    Temporal attention contributes 4*r*h*L**2; projections/feedforward layers
    are linear in length and diffusion projections are independent of length.
    Feature attention is bypassed for one feature. Bias adds are excluded.
    """
    if architecture["n_channels"] != 1 or type(length) is not int or length < 2:
        raise ValueError("Formula requires univariate sequences of length >= 2")
    h = architecture["hidden_channels"]
    r = architecture["residual_layers"]
    d = architecture["diffusion_embedding_dim"]
    f = architecture["feedforward_dim"]
    side = (
        architecture["time_embedding_dim"] + architecture["feature_embedding_dim"] + 1
    )
    return (
        4 * r * h * length**2
        + (6 * h + 2 * h**2 + r * (16 * h**2 + 4 * h * f + 4 * side * h)) * length
        + 4 * d**2
        + 2 * r * d * h
    )


def count_csdi_work(architecture, length, *, batch_size=2):
    """Count actual forward/backward matrix operations, per input trajectory.

    Math SDPA and disabled MHA fastpath expose QK/AV matrix products to the
    counter; fused inference otherwise omits attention work on this CPU.
    This changes the profiling implementation only, never scientific weights.
    Input data need no gradients, as in the real trainers. Backward therefore
    need not be exactly twice forward. Optimizer/nonmatrix work is excluded.
    """
    if type(length) is not int or length < 2:
        raise ValueError("Require a sequence length of at least two")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Require a positive integer batch size")
    fastpath = torch.backends.mha.get_fastpath_enabled()
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            model = CSDIDenoiser(**architecture)
            values = torch.zeros(batch_size, length, architecture["n_channels"])
            mask = torch.ones_like(values, dtype=torch.bool)
            steps = torch.zeros(batch_size, dtype=torch.long)
            result = {}
            for training in (False, True):
                model.train(training)
                model.zero_grad(set_to_none=True)
                with (
                    torch.set_grad_enabled(training),
                    sdpa_kernel(SDPBackend.MATH),
                    FlopCounterMode(display=False) as counter,
                ):
                    prediction = model(
                        values, steps, observed=values, mask=mask, dt=1.0
                    )
                    if training:
                        prediction.square().mean().backward()
                name = "training" if training else "forward"
                result[name] = counter.get_total_flops() / batch_size
                result[f"{name}_operations"] = {
                    str(op): count / batch_size
                    for op, count in counter.flop_counts["Global"].items()
                }
            result["parameters"] = sum(p.numel() for p in model.parameters())
            result["length"] = length
            return result
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)


def count_mediator_work(width):
    """Per historical center, including first-layer weight but no input gradient."""
    with torch.random.fork_rng(devices=[]):
        model = BlindSpotMediator(width)
        result = {}
        for training in (False, True):
            model.zero_grad(set_to_none=True)
            with (
                torch.set_grad_enabled(training),
                FlopCounterMode(display=False) as counter,
            ):
                prediction = model(torch.zeros(2, 12))
                if training:
                    prediction.square().mean().backward()
            result["training" if training else "forward"] = (
                counter.get_total_flops() / 2
            )
        result["parameters"] = sum(p.numel() for p in model.parameters())
        return result


def fitting_ledger(
    *,
    epochs,
    n_train,
    n_validation,
    forward,
    training,
    noise_validation_count,
    noise_validation_steps,
    forecast_validation_count,
    forecast_samples,
    diffusion_steps,
    setup=0.0,
    extra_per_forecast_validation=0.0,
):
    """Charge all completed optimization and selection, without double counting."""
    inputs = locals().copy()
    for name, value in inputs.items():
        _positive(name, value, allow_zero=True)
    if any(
        type(inputs[name]) is not int
        for name in (
            "epochs",
            "n_train",
            "n_validation",
            "noise_validation_count",
            "noise_validation_steps",
            "forecast_validation_count",
            "forecast_samples",
            "diffusion_steps",
        )
    ):
        raise ValueError("Counts must be integers")
    result = {
        "setup": float(setup),
        "optimization": epochs * n_train * training,
        "noise_validation": (
            noise_validation_count * n_validation * noise_validation_steps * forward
        ),
        "forecast_validation": forecast_validation_count
        * (
            n_validation * forecast_samples * diffusion_steps * forward
            + extra_per_forecast_validation
        ),
    }
    result["total"] = sum(result.values())
    return result


def best_within_budget(candidates, budget, cost_key):
    """Select only scores already obtainable within the stated cost ceiling.

    Candidates must be in chronological order; cost is cumulative, including
    every preceding validation pass. No interpolation or future-score access.
    Ties select the first occurrence. None means no scored candidate fits.
    """
    _positive("budget", budget, allow_zero=True)
    previous_cost = -1.0
    previous_epoch = -1
    eligible = []
    for row in candidates:
        cost, score, epoch = row[cost_key], row["crps"], row["epoch"]
        _positive("cost", cost, allow_zero=True)
        _positive("CRPS", score, allow_zero=True)
        if cost < previous_cost or epoch <= previous_epoch:
            raise ValueError("Candidates must have increasing epochs and costs")
        previous_cost, previous_epoch = cost, epoch
        if cost <= budget:
            eligible.append(row)
    if not eligible:
        return None
    selected = min(eligible, key=lambda row: row["crps"])
    return {
        "epoch": selected["epoch"],
        "crps": selected["crps"],
        "selected_candidate_cost": selected[cost_key],
        "spent_cost": eligible[-1][cost_key],
        "unused_budget": budget - eligible[-1][cost_key],
        "validation_opportunities": len(eligible),
        "available_through_epoch": eligible[-1]["epoch"],
    }


def budget_table(traces, methods, budgets, cost_key):
    result = []
    for budget in budgets:
        row = {"budget": float(budget), "methods": {}}
        for method in methods:
            selections = {
                seed: best_within_budget(data[method], budget, cost_key)
                for seed, data in traces.items()
            }
            complete = all(value is not None for value in selections.values())
            scores = [v["crps"] for v in selections.values() if v is not None]
            row["methods"][method] = {
                "selections": selections,
                "all_seeds_available": complete,
                "mean_crps": float(np.mean(scores)) if complete else None,
                "seed_sd": float(np.std(scores, ddof=1)) if complete else None,
            }
        result.append(row)
    return result
