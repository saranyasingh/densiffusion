"""Summarize the completed frozen comparison; never fit or select models."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from densiffusion.polishing_analysis import _validation_plot

ROOT = Path(__file__).resolve().parent
PRIMARY = ROOT / "primary"
METHODS = ("raw", "learned", "median")
METRICS = ("crps", "mse", "coverage_90", "interval_width_90")


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    verification = read(PRIMARY / "verification.json")
    assert verification["passed"] and not verification["training_only"]
    analysis = read(PRIMARY / "analysis.json")
    assert analysis["protocol"]["status"] == "complete"
    manifest = read(PRIMARY / "config.json")
    config = manifest["experiment"]
    across = analysis["across_seed_descriptive"]
    means = {m: {k: across["scores"][m][k]["mean"] for k in METRICS} for m in METHODS}
    contrast_summary = {}
    bootstrap_checks = []
    for contrast in ("learned-minus-raw", "learned-minus-median", "median-minus-raw"):
        left, right = contrast.split("-minus-")
        differences, intervals = [], []
        for seed, data in analysis["seeds"].items():
            saved = data["paired_differences"][contrast]
            # Independent batching and direct sum implementation of whole-row bootstrap.
            rows = np.array([
                np.asarray(data["scores"][left]["per_trajectory"][metric])
                - np.asarray(data["scores"][right]["per_trajectory"][metric])
                for metric in METRICS
            ]).T
            rng = np.random.default_rng(saved["bootstrap_seed"])
            replicates = np.empty((10000, 4))
            for start in range(0, 10000, 125):
                count = min(125, 10000 - start)
                indices = rng.integers(len(rows), size=(count, len(rows)))
                replicates[start:start+count] = rows[indices].sum(axis=1) / len(rows)
            limits = np.quantile(replicates, [0.025, 0.975], axis=0)
            for index, metric in enumerate(METRICS):
                assert np.allclose(limits[:, index], saved[metric]["ci95"], rtol=1e-12, atol=1e-12)
                assert np.isclose(rows[:, index].mean(), saved[metric]["difference"], rtol=1e-12, atol=1e-12)
            bootstrap_checks.append({"seed": seed, "contrast": contrast, "all_four_metrics_passed": True})
            differences.append(saved["crps"]["difference"])
            intervals.append(saved["crps"]["ci95"])
        contrast_summary[contrast] = {
            "seed_crps_differences": differences,
            "seed_crps_ci95": intervals,
            "seeds_negative": sum(d < 0 for d in differences),
            "seeds_negative_ci_upper": sum(ci[1] < 0 for ci in intervals),
        }
    plans = {str(s): read(PRIMARY / f"seed_{s}/budget_plan.json") for s in config["seeds"]}
    outcome = {
        "status": "completed", "scope": "prospective matrix-work-matched polishing comparison",
        "mean_scores": means, "contrasts": contrast_summary,
        "primary_ceiling": next(iter(plans.values()))["ceiling"],
        "maximum_unused_fraction": max(p["unused_fraction"] for plan in plans.values() for p in plan["methods"].values()),
        "same_wall_time_claim": False, "optimal_allocation_claim": False,
        "convergence_claim": False, "densification_claim": False,
    }
    write(ROOT / "OUTCOME.json", outcome)
    write(ROOT / "bootstrap_recomputation.json", {"replicates": 10000, "unit": "paired whole test trajectory", "checks": bootstrap_checks})
    lines = []
    add = lines.append
    raw, median = contrast_summary["learned-minus-raw"], contrast_summary["learned-minus-median"]
    add(
        f"The prospective resource-matched polishing comparison is complete. "
        f"Learned polishing has lower test CRPS than raw inputs in **{raw['seeds_negative']}/3 seeds** "
        f"and lower CRPS than median filtering in **{median['seeds_negative']}/3 seeds**. "
        f"The corresponding counts of paired 95% intervals lying wholly below zero are "
        f"**{raw['seeds_negative_ci_upper']}/3** and **{median['seeds_negative_ci_upper']}/3**. "
        "This comparison matches a specified matrix-work budget; it does not establish equal wall time, "
        "energy, an optimal allocation, or a densification advantage."
    )
    add("\nThe [protocol](protocol.json) was fixed before scientific data generation and fitting. "
        "The [exact configuration](primary/config.json), [frozen model manifest](primary/FROZEN.json), "
        "[independent verification](primary/verification.json), [analysis](primary/analysis.json), "
        "and [structured outcome](OUTCOME.json) retain the audit trail. Earlier experiments remain unchanged.")
    add("\n| Method | CRPS | Ensemble-mean MSE | 90% coverage | 90% interval width |\n|---|---:|---:|---:|---:|")
    for method in METHODS:
        v = means[method]
        add(f"| {method} | {v['crps']:.6f} | {v['mse']:.6f} | {100*v['coverage_90']:.2f}% | {v['interval_width_90']:.6f} |")
    add("\nThese are equal-seed means over 2,048 fresh test trajectories per seed, with 64 forecast draws per trajectory. "
        "Targets are the original noisy future observations. No clean simulator signals are used. "
        "Lower CRPS and MSE are better; nominal coverage is 90%, and width alone does not measure calibration.")
    add("\n| Contrast | Mean relative CRPS reduction from baseline |\n|---|---:|")
    for baseline in ("raw", "median"):
        add(f"| learned versus {baseline} | {100*(1-means['learned']['crps']/means[baseline]['crps']):.3f}% |")
    add("\n**Per-seed evidence and uncertainty.** Intervals resample 2,048 whole paired trajectories 10,000 times, retaining all horizons. "
        "They condition on fitted models, preprocessing, and saved draws; they do not resample training or account for method search. "
        "They are unadjusted across comparisons and metrics. Three seeds provide limited evidence about training variation.")
    add("\n| Seed | Raw CRPS | Learned CRPS | Median CRPS | Learned minus raw (95% interval) | Learned minus median (95% interval) |\n|---|---:|---:|---:|---|---|")
    for seed, data in analysis["seeds"].items():
        scores = [data["scores"][m]["aggregate"]["crps"] for m in METHODS]
        cells = []
        for contrast in ("learned-minus-raw", "learned-minus-median"):
            d = data["paired_differences"][contrast]["crps"]
            cells.append(f"{d['difference']:+.6f} [{d['ci95'][0]:+.6f}, {d['ci95'][1]:+.6f}]")
        add(f"| {seed} | {scores[0]:.6f} | {scores[1]:.6f} | {scores[2]:.6f} | {cells[0]} | {cells[1]} |")
    add("\n| Method | Metric | Mean | Seed sample SD | Seed range |\n|---|---|---:|---:|---|")
    for method in METHODS:
        for metric in METRICS:
            d = across["scores"][method][metric]
            add(f"| {method} | {metric} | {d['mean']:.6f} | {d['sample_std']:.6f} | {d['minimum']:.6f}–{d['maximum']:.6f} |")
    add("\n| Seed | Contrast | Metric | Paired difference | 95% interval |\n|---|---|---|---:|---|")
    for seed, data in analysis["seeds"].items():
        for contrast, d in data["paired_differences"].items():
            for metric in METRICS:
                v = d[metric]
                add(f"| {seed} | {contrast} | {metric} | {v['difference']:+.6f} | [{v['ci95'][0]:+.6f}, {v['ci95'][1]:+.6f}] |")
    add("\n**Resource matching.** The ceiling is 173,320,595,046,400 counted dense matrix operations per method and seed. "
        "One multiplication and one addition count as two operations. The budget includes all four mediator candidates, "
        "masked reconstruction validation, preprocessing, forecaster fitting, all-step noise validation, and every forecast selection pass. "
        "Counts were independently derived and audited using actual forward and backward hooks. "
        "Nonmatrix arithmetic, optimizer work, median comparisons, memory traffic, and I/O are excluded from this unit; wall times are reported separately.")
    add("\n| Method | Forecaster epochs | Updates | Total charged matrix work (trillion) | Unused ceiling |\n|---|---:|---:|---:|---:|")
    first = next(iter(plans.values()))
    for method, p in first["methods"].items():
        add(f"| {method} | {p['epochs']} | {p['optimizer_updates']} | {p['work']['total']/1e12:.9f} | {100*p['unused_fraction']:.6f}% |")
    add("\n| Method | Auxiliary search + processing (trillion) | Forecaster optimization (trillion) | All-step noise validation (trillion) | Forecast selection (trillion) |\n|---|---:|---:|---:|---:|")
    for method, p in first["methods"].items():
        add(f"| {method} | " + " | ".join(f"{p['work'][k]/1e12:.9f}" for k in ("setup", "optimization", "noise_validation", "forecast_validation")) + " |")
    baseline_fit = first["methods"]["raw"]["work"]["optimization"]
    learned_fit = sum(first["methods"]["learned"]["work"][k] for k in ("setup", "optimization"))
    add(f"\nValidation work is identical and cancels in the comparison. Considering only forecaster optimization plus "
        f"auxiliary search/preprocessing, learned uses {100*(baseline_fit-learned_fit)/baseline_fit:.6f}% less counted work than raw. "
        "Thus the close match does not rely on including the common, expensive validation workload in the denominator.")
    test_sampling_work = config["n_test"] * config["samples"] * config["diffusion_steps"] * manifest["work_profiles"]["forecaster"]["forward"]
    test_mediator_work = config["n_test"] * config["context_length"] * manifest["work_profiles"]["mediator_64"]["forward"]
    combined_ceiling = first["ceiling"] + test_sampling_work
    combined_learned = first["methods"]["learned"]["work"]["total"] + test_sampling_work + test_mediator_work
    add(f"\nThe predeclared allocation ceiling covers model development. Held-out evaluation is separate: "
        f"each method uses {test_sampling_work/1e12:.9f} trillion matrix operations per seed for the same test sampler workload; "
        f"learned test preprocessing adds {test_mediator_work/1e12:.9f} trillion. "
        f"Adding these costs gives {combined_ceiling/1e12:.9f} trillion for raw/median and {combined_learned/1e12:.9f} trillion for learned "
        f"({100*(combined_ceiling-combined_learned)/combined_ceiling:.6f}% less). "
        "This supplementary accounting does not change the frozen allocation or checkpoint selection. "
        "The timing pilot, source profiling, smoke tests, and implementation checks are research overhead outside these model budgets.")
    add("\nAll three seeds selected width-64 Huber mediators, so these allocations are identical across seeds. "
        "Each method has seven forecast-CRPS candidates: 10, 50, 100, 200, 400, 800, and its final epoch. "
        "All 256 validation trajectories receive 64 draws per candidate. All-step denoising validation has 120 passes per method. "
        "Only forecast CRPS selects forecaster checkpoints. LR milestones occur at 75% and 90% of each method's allotted epochs.")
    add("\n| Seed | Raw selected epoch | Learned selected epoch | Median selected epoch | Mediator selected candidate / epoch |\n|---|---:|---:|---:|---|")
    for seed in config["seeds"]:
        folder = PRIMARY / f"seed_{seed}"
        a = read(folder / "training_audit.json")["methods"]
        m = read(folder / "mediator/summary.json")
        add(f"| {seed} | {a['raw']['selected_epoch']} | {a['learned']['selected_epoch']} | {a['median']['selected_epoch']} | {m['selected']} / {m['selected_epoch']} |")
    add("\nCompleting the budgets or selecting the final checkpoint does not establish convergence. "
        "This tests fixed pipelines at one budget. It does not independently vary mediator effort, forecaster effort, or tuning policy to locate an optimum. "
        "Fresh training data and initializations also differ from the earlier run, so a changed outcome cannot be attributed solely to the eight-epoch allocation adjustment.")
    add("\n**Compute costs.** One MPS worker and one Torch thread were used on an Apple M1 Max with 64 GiB memory, "
        "with method order rotated across seeds; [environment versions](environment.json) are recorded. "
        "The data-free [batch pilot](batch_timing_pilot.json) selected 1,024 equally for all methods. "
        "Forecaster fitting times already include validation sampling; the validation column below is a subset and must not be added again.")
    add("\n| Method | Forecaster fit incl. validation (s/seed) | Forecast validation subset (s/seed) | Auxiliary fit/search (s/seed) | Train/val processing (s/seed) | Test inference (s/seed) | Test processing (s/seed) |\n|---|---:|---:|---:|---:|---:|---:|")
    costs = [read(PRIMARY / f"seed_{s}/combined_timings.json") for s in config["seeds"]]
    for method in METHODS:
        values = [np.mean([r[k][method] for r in costs]) for k in ("forecaster_training_seconds", "validation_sampling_seconds")]
        aux = np.mean([r['mediator_training_seconds'] for r in costs]) if method == 'learned' else 0
        values += [aux]
        values += [np.mean([r[k][method] for r in costs]) for k in ("preprocessing_seconds", "test_inference_seconds", "test_preprocessing_seconds")]
        add(f"| {method} | " + " | ".join(f"{v:.6f}" for v in values) + " |")
    train_seconds = read(PRIMARY / "TRAINED.json")["wall_seconds"]
    test_seconds = read(PRIMARY / "EVALUATED.json")["wall_seconds"]
    add(f"\nTraining/selection phase: {train_seconds/60:.2f} minutes. Fresh-test evaluation phase: {test_seconds/60:.2f} minutes. "
        "Timing pilot, smoke testing, coding, analysis, and verification are separate research overhead. "
        "Measured wall times need not be identical under a matched matrix-work budget. Deployment still includes a small additional mediator or median preprocessing cost.")
    add("\n**Data, leakage controls, and reproduction.** The unchanged noisy-periodic generator draws independent amplitude Uniform(0.7,1.3), "
        "period Uniform(24,40), and phase Uniform(0,2π); the harmonic is 0.2 sin(2θ+0.4). Observations add Gaussian noise SD 0.15 "
        "and an independent Gaussian SD 2 spike with probability 0.1. Each split contains independent trajectories. "
        "Each history has 32 integer-time values and each future has eight original noisy observations. "
        "The mediator receives only the six fixed neighbors excluding the center, plus boundary availability masks; median uses edge padding on history alone. "
        "No future or clean values enter preprocessing. No normalization, densification, GP weights, or test-based tuning is used.")
    add("\nAll forecasters use four residual blocks, 64 hidden channels, eight attention heads, 128-dimensional time and diffusion embeddings, "
        "feature embedding 16, feedforward width 64, and dropout 0.1: 413,521 parameters each. Adam uses LR 0.001, weight decay 1e-6, "
        "gradient clipping at 1, and the recovered 50-step quadratic schedule from 1e-4 to 0.5. Initial weights, shuffle streams, and noise streams are paired. "
        "All mediators and allocations froze before any forecaster fitting. All models froze globally before any new test generation.")
    add("\n[Representative forecasts](primary/plots/representative_forecasts.svg), [per-horizon performance](primary/plots/per_horizon.svg), "
        "[complete validation curves](primary/plots/validation_crps.svg), and the [epoch-50-onward detail](preview/validation_detail.svg) "
        "show fixed examples and all horizons. PNG previews and actual visual-review notes are saved in `preview/` and `plot_review.json`. "
        "Coverage is marginal and finite-ensemble quantiles affect it; narrow intervals do not establish useful calibration.")
    add("\nIndependent verification replays every data split and boundary preprocessing, reconstructs mediator selection and allocations, "
        "checks actual update/forward counts, LR schedules, paired initialization, freeze ordering, and all validation ensembles, and recomputes "
        "test CRPS using the pairwise definition. A separate [bootstrap recomputation](bootstrap_recomputation.json) checks all 36 metric intervals. "
        "The full implementation suite passed 542 tests; final preservation and source checks are recorded in `final_checks.json`. No commit or push occurred.")
    add("\n```sh\nuv run --locked python scripts/train_budgeted_polishing.py \\\n  --config configs/history_polishing_budget_matched.json \\\n  --output outputs/budgeted-polishing-reproduction\nuv run --locked python scripts/verify_budgeted_polishing.py outputs/budgeted-polishing-reproduction\n```\nUse a fresh output directory. Full source snapshots, exact samples, configurations, logs, checkpoints, and timings are retained.")
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n")
    preview = ROOT / "preview"
    preview.mkdir(exist_ok=True)
    detail = copy.deepcopy(analysis)
    for entry in detail['seeds'].values():
        for review in entry['validation'].values():
            review['rows'] = [r for r in review['rows'] if r['epoch'] >= 50]
    _validation_plot(detail, preview / "validation_detail.svg")
    text = (preview / "validation_detail.svg").read_text().replace(
        "Noisy-future validation CRPS checkpoint candidates", "Validation CRPS detail | epochs 50 onward"
    ).replace("curves do not certify convergence.", "epoch 10 omitted from this detail; no convergence claim.")
    (preview / "validation_detail.svg").write_text(text)
    for path in [*PRIMARY.glob('plots/*.svg'), preview / 'validation_detail.svg']:
        text = path.read_text()
        import re
        match = re.search(r'width="(\d+)" height="(\d+)" viewBox="0 0 \d+ \d+"',text)
        if match:
            size = max(map(int,match.groups()))
            text = text[:match.start()] + f'width="{size}" height="{size}" viewBox="0 0 {size} {size}"' + text[match.end():]
        (preview / path.name).write_text(text)
    print(json.dumps(outcome,indent=2))


if __name__ == '__main__':
    main()
