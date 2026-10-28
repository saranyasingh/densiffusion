"""Record the inspected figures and interpretation of the frozen results."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    notes = {
        "representative_forecasts": (
            "All 18 panels are visible and readable: methods in columns and the first six test trajectories "
            "(IDs 512 through 517, seed 29101) in rows. Both preprocessing methods remove large historical "
            "spikes; median histories contain flat segments. The future spike in trajectory 513 lies outside "
            "the displayed central bands. The plots show noisy targets only. Fixed examples are illustrative, "
            "not evidence of an aggregate win."
        ),
        "per_horizon": (
            "All four metric panels, legends, axes and the nominal 90% reference are visible. "
            "Learned has lower mean CRPS than raw at all eight horizons, and lower than median at steps "
            "1 through 6; median is lower at steps 7 and 8. Median has lower mean MSE at every horizon. "
            "All coverage means are below 90%. Shading is explicitly a descriptive seed range, not a confidence interval."
        ),
        "validation_crps": (
            "All three seed panels and early checkpoint points are present with readable labels. "
            "Large epoch-10 scores compress later differences on the linear scale; the detail plot resolves these."
        ),
        "validation_detail": (
            "All three panels are readable. The title and caption explicitly omit epoch 10. "
            "Late improvements and reversals differ by seed and method; endpoint selections and apparent flatness "
            "do not certify convergence. No scientific curves or selected checkpoints were changed for plotting."
        ),
    }
    plots = []
    for stem, note in notes.items():
        png = ROOT / "preview" / f"{stem}.svg.png"
        with png.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        plots.append({"png": str(png.relative_to(ROOT)), "sha256": sha, "notes": note})
    write(ROOT / "plot_review.json", {
        "inspected": True,
        "reviewed_utc": datetime.now(UTC).isoformat(),
        "method": "Quick Look PNG rendering at 1800 pixels, all four images inspected using view_image",
        "layout": "No clipping or overlapping labels observed; square preview canvases retain the complete original SVG plots.",
        "plots": plots,
    })
    analysis = read(ROOT / "primary/analysis.json")
    outcome = read(ROOT / "OUTCOME.json")
    means = outcome["mean_scores"]
    interpretation = (
        "The learned model's CRPS advantage over raw inputs survives this resource definition in all three fresh seeds. "
        "The additional CRPS advantage over median filtering is small: its paired interval includes zero in two seeds. "
        "This does not establish a consistently superior learned preprocessing method. Learned polishing also has "
        "higher average ensemble-mean MSE than both baselines, while median filtering has better average coverage. "
        "All three methods substantially under-cover nominal 90% intervals. The learned model's narrower intervals "
        "are therefore not evidence of better calibration."
    )
    outcome["interpretation"] = interpretation
    outcome["relative_crps_reduction"] = {
        baseline: 1 - means["learned"]["crps"] / means[baseline]["crps"]
        for baseline in ("raw", "median")
    }
    outcome["all_methods_undercover_nominal_90"] = all(
        means[m]["coverage_90"] < 0.9 for m in means
    )
    outcome["visual_review"] = "plot_review.json"
    write(ROOT / "OUTCOME.json", outcome)
    report = ROOT / "REPORT.md"
    text = report.read_text()
    marker = "| Contrast | Mean relative CRPS reduction from baseline |"
    assert marker in text and "**Interpretation of the completed comparison.**" not in text
    explanation = (
        "**Interpretation of the completed comparison.** " + interpretation + "\n\n"
        "MSE improves with learned polishing in seeds 29101 and 29102, but deteriorates in seed 29103 "
        "(0.545192 versus 0.478749 raw and 0.474636 median). This third-seed deterioration reverses the "
        "average MSE comparison. Forecast CRPS was the predeclared selection metric; these secondary outcomes "
        "remain part of the result and did not trigger model revision.\n\n"
    )
    text = text.replace(marker, explanation + marker, 1)
    horizon = analysis["across_seed_descriptive"]["per_horizon"]
    table = [
        "**Per-horizon results.** These are equal-seed means. Step 1 is observation time 32 and step 8 is time 39. "
        "Learned CRPS is lower than raw at every horizon. Its advantage over median is confined to steps 1 through 6; "
        "median is lower at steps 7 and 8. Median has lower mean MSE at every horizon. All four metrics and seed ranges "
        "are also visible in the inspected per-horizon figure.",
        "",
        "| Forecast step | Method | CRPS | Ensemble-mean MSE | 90% coverage | 90% width |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for step in range(8):
        for method in ("raw", "learned", "median"):
            row = {metric: horizon[method][metric][step]["mean"] for metric in horizon[method]}
            table.append(
                f"| {step+1} | {method} | {row['crps']:.6f} | {row['mse']:.6f} | "
                f"{100*row['coverage_90']:.2f}% | {row['interval_width_90']:.6f} |"
            )
    marker = "**Resource matching.**"
    assert marker in text
    text = text.replace(marker, "\n".join(table) + "\n\n" + marker, 1)
    text = text.replace(
        "PNG previews and actual visual-review notes are saved in `preview/` and `plot_review.json`.",
        "All four rendered plots were inspected; [visual-review notes](plot_review.json) record the observations. "
        "PNG previews are saved in `preview/`."
    )
    report.write_text(text)
    print(json.dumps({"plots_inspected": 4, "interpretation": interpretation}, indent=2))


if __name__ == "__main__":
    main()
