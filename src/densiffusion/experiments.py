from pathlib import Path


def run_experiment(config_path: Path, *, output_dir: Path) -> dict[str, float]:
    """Run generation, splitting/windowing, training, forecasting, and evaluation."""
    raise NotImplementedError("The experiment runner is not implemented yet.")
