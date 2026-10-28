"""History-only median and blind-spot polishing for 32 observed time points.

The learned mediator receives fixed neighbor values and availability indicators,
never the center, clean signals, or future observations. No normalization or
residual path can introduce the excluded center indirectly. Fitting treats the
noisy centers as targets and selects by noisy-center validation MSE.
"""

import hashlib
import json
import shutil
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from densiffusion.contracts import Array, positive_int, validate_array

OFFSETS = (-3, -2, -1, 1, 2, 3)
HISTORY_LENGTH = 32


def _history(history: Array) -> Array:
    return validate_array(history, (None, HISTORY_LENGTH, 1))


def raw_history(history: Array) -> Array:
    """Return a separate, unchanged copy of the supplied history."""
    return _history(history).copy()


def neighbor_features(history: Array) -> Array:
    """Return six fixed-offset values followed by six availability indicators.

    Missing neighbors beyond either history boundary have value and mask zero.
    Available neighbors retain their fixed slot; slots never shift near edges.
    """
    values = _history(history)
    result = np.zeros((len(values), HISTORY_LENGTH, 12), dtype=np.float64)
    for slot, offset in enumerate(OFFSETS):
        centers = np.arange(
            max(0, -offset), min(HISTORY_LENGTH, HISTORY_LENGTH - offset)
        )
        result[:, centers, slot] = values[:, centers + offset, 0]
        result[:, centers, slot + len(OFFSETS)] = 1.0
    return result


def median_history(history: Array) -> Array:
    """Apply a seven-point median to the history alone, with edge padding."""
    values = _history(history)
    padded = np.pad(values[:, :, 0], ((0, 0), (3, 3)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 7, axis=1)
    return np.median(windows, axis=-1)[:, :, None]


class BlindSpotMediator(nn.Module):
    """Shared pointwise MLP: 12 neighbor features -> two ReLU layers -> scalar."""

    def __init__(self, hidden_channels: int = 32):
        super().__init__()
        positive_int("hidden_channels", hidden_channels)
        self.hidden_channels = hidden_channels
        self.network = nn.Sequential(
            nn.Linear(12, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim < 2 or features.shape[-1] != 12:
            raise ValueError("Expected features with final dimension 12")
        return self.network(features)


def polish_history(model: BlindSpotMediator, history: Array) -> Array:
    """Evaluate a fixed mediator from neighbors of these 32 observations only."""
    features = neighbor_features(history)
    parameter = next(model.parameters())
    tensor = torch.as_tensor(features, device=parameter.device, dtype=parameter.dtype)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            predictions = model(tensor).cpu().numpy().astype(np.float64)
    finally:
        model.train(was_training)
    return validate_array(predictions, (len(features), HISTORY_LENGTH, 1))


def _json(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _array_hash(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode())
    digest.update(values.dtype.str.encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(_array_hash(value.detach().cpu().numpy()).encode())
    return digest.hexdigest()


def _new_model(width: int, seed: int) -> BlindSpotMediator:
    # Construction uses the CPU RNG only; fork_rng restores the caller's state.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return BlindSpotMediator(width)


def load_mediator(checkpoint: str | Path, device: str = "cpu") -> BlindSpotMediator:
    """Load selected or candidate weights into an evaluation-only mediator."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = _new_model(saved["config"]["hidden_channels"], 0)
    model.load_state_dict(saved["model"])
    model.to(device).eval().requires_grad_(False)
    return model


def train_mediator(
    train_history: Array,
    val_history: Array,
    output: str | Path,
    seed: int,
    device: str = "cpu",
    smoke: bool = False,
) -> tuple[BlindSpotMediator, dict]:
    """Fit four bounded candidates using only supplied noisy histories.

    Candidate order is widths 32/64 crossed with MSE/Huber(delta=.2). Each gets
    an independent private initialization and shuffling stream. Score after
    epochs 1, 5, and every 10 (200 epochs total; smoke uses 5). First occurrence
    wins exact ties at both checkpoint and candidate selection. An existing
    output directory is rejected. No test or future-generation API is called.
    """
    train = _history(train_history).copy()
    validation = _history(val_history).copy()
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(smoke) is not bool:
        raise ValueError("smoke must be a boolean")
    if device not in ("cpu", "mps", "cuda"):
        raise ValueError("device must be cpu, mps, or cuda")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    epochs = 5 if smoke else 200
    score_epochs = sorted({1, 5, *range(10, epochs + 1, 10), epochs})
    streams = np.random.SeedSequence(seed).spawn(4)
    features = torch.as_tensor(
        neighbor_features(train).reshape(-1, 12), dtype=torch.float32
    )
    targets = torch.as_tensor(train.reshape(-1, 1), dtype=torch.float32)
    candidates = []
    started = perf_counter()
    for index, (width, loss_name) in enumerate(
        (width, loss_name) for width in (32, 64) for loss_name in ("mse", "huber")
    ):
        directory = output / f"candidate_{index:02d}_{width}_{loss_name}"
        directory.mkdir()
        init_stream, shuffle_stream = streams[index].spawn(2)
        init_seed = int(init_stream.generate_state(1)[0])
        shuffle_seed = int(shuffle_stream.generate_state(1)[0])
        shuffle = np.random.default_rng(shuffle_seed)
        config = {
            "hidden_channels": width,
            "hidden_layers": 2,
            "activation": "relu",
            "loss": loss_name,
            "huber_delta": 0.2,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "batch_size": 256,
            "epochs": epochs,
            "score_epochs": score_epochs,
            "offsets": list(OFFSETS),
            "feature_order": "six_neighbor_values_then_six_availability_indicators",
            "history_length": HISTORY_LENGTH,
            "init_seed": init_seed,
            "shuffle_seed": shuffle_seed,
            "device": device,
            "smoke": smoke,
        }
        _json(directory / "config.json", config)
        model = _new_model(width, init_seed).to(device)
        initial_sha256 = _state_hash(model.state_dict())
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        criterion = nn.MSELoss() if loss_name == "mse" else nn.HuberLoss(delta=0.2)
        best_score = float("inf")
        selection = []
        updates = 0
        candidate_started = perf_counter()
        with (directory / "training.jsonl").open("x") as log:
            for epoch in range(1, epochs + 1):
                model.train()
                permutation = shuffle.permutation(len(features))
                epoch_loss = 0.0
                for start in range(0, len(features), 256):
                    rows = permutation[start : start + 256]
                    batch_x = features[rows].to(device)
                    batch_y = targets[rows].to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(model(batch_x), batch_y)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite mediator training loss")
                    loss.backward()
                    optimizer.step()
                    epoch_loss += float(loss.detach().cpu()) * len(rows)
                    updates += 1
                row = {
                    "epoch": epoch,
                    "training_loss": epoch_loss / len(features),
                    "optimizer_updates": updates,
                }
                if epoch in score_epochs:
                    predictions = polish_history(model, validation)
                    score = float(np.mean((predictions - validation) ** 2))
                    row["validation_mse"] = score
                    selection.append({"epoch": epoch, "validation_mse": score})
                    if score < best_score:
                        best_score = score
                        state = {
                            key: tensor.detach().cpu().clone()
                            for key, tensor in model.state_dict().items()
                        }
                        checkpoint = {
                            "model": state,
                            "config": config,
                            "epoch": epoch,
                            "validation_mse": score,
                            "optimizer_updates": updates,
                            "initial_sha256": initial_sha256,
                        }
                        torch.save(checkpoint, directory / "best.pt")
                        np.save(
                            directory / "best_validation_predictions.npy", predictions
                        )
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
        _json(directory / "validation.json", selection)
        candidate = {
            "name": directory.name,
            "config": config,
            "selected_epoch": checkpoint["epoch"],
            "validation_mse": best_score,
            "initial_sha256": initial_sha256,
            "selected_sha256": _state_hash(checkpoint["model"]),
            "checkpoint_sha256": hashlib.sha256(
                (directory / "best.pt").read_bytes()
            ).hexdigest(),
            "optimizer_updates": updates,
            "training_seconds": perf_counter() - candidate_started,
        }
        candidates.append(candidate)
    selected = min(candidates, key=lambda candidate: candidate["validation_mse"])
    shutil.copyfile(output / selected["name"] / "best.pt", output / "selected.pt")
    frozen = load_mediator(output / "selected.pt", device=device)
    train_predictions = polish_history(frozen, train)
    validation_predictions = polish_history(frozen, validation)
    saved_predictions = np.load(
        output / selected["name"] / "best_validation_predictions.npy"
    )
    if not np.array_equal(validation_predictions, saved_predictions):
        raise AssertionError("Reloaded selected mediator predictions changed")
    np.savez_compressed(
        output / "selected_predictions.npz",
        train_predictions=train_predictions,
        validation_predictions=validation_predictions,
    )
    summary = {
        "seed": seed,
        "smoke": smoke,
        "selection_metric": "noisy_center_validation_mse",
        "selection_ties": "first_in_epoch_then_candidate_order",
        "train_shape": list(train.shape),
        "validation_shape": list(validation.shape),
        "train_history_sha256": _array_hash(train),
        "validation_history_sha256": _array_hash(validation),
        "validation_centers": int(validation.size),
        "center_excluded_from_features": True,
        "clean_values_or_future_observations_accessed": False,
        "candidates": candidates,
        "selected": selected["name"],
        "selected_epoch": selected["selected_epoch"],
        "validation_mse": selected["validation_mse"],
        "selected_checkpoint_sha256": hashlib.sha256(
            (output / "selected.pt").read_bytes()
        ).hexdigest(),
        "train_predictions_sha256": _array_hash(train_predictions),
        "validation_predictions_sha256": _array_hash(validation_predictions),
        "total_training_seconds": perf_counter() - started,
    }
    _json(output / "summary.json", summary)
    return frozen, summary
