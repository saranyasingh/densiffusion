"""Verify saved epoch-150 validation outputs across the two waveform attempts.

Run only once all three second-attempt epoch-150 archives and score rows exist:
  .venv/bin/python outputs/waveform-large-20261027/verify_shared_prefix.py

Reads existing validation artifacts and training logs only. It never constructs
a model, runs training/sampling, or accesses test observations or fine truth.
NPZ file bytes may differ because of ZIP metadata; decompressed arrays are
compared exactly, including dtype, shape, and values.
"""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

EPOCH = 150


def read_json(path):
    return json.loads(path.read_text())


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_hash(value):
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(str(value.shape).encode())
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def epoch_row(folder):
    rows = [
        row for row in read_json(folder / "validation.json") if row["epoch"] == EPOCH
    ]
    if len(rows) != 1:
        raise ValueError(
            f"Require exactly one completed epoch-{EPOCH} score in {folder}"
        )
    return rows[0]


def training_prefix(folder):
    rows = []
    with (folder / "training.jsonl").open() as stream:
        for line in stream:
            if not line.endswith("\n"):
                break  # An ongoing writer may have an incomplete final line.
            row = json.loads(line)
            if row["epoch"] > EPOCH:
                break
            rows.append(
                {
                    key: row[key]
                    for key in (
                        "epoch",
                        "train_loss",
                        "learning_rate",
                        "validation_loss",
                    )
                    if key in row
                }
            )
    if [row["epoch"] for row in rows] != list(range(1, EPOCH + 1)):
        raise ValueError(
            f"Complete training records through epoch {EPOCH} required: {folder}"
        )
    return rows


def learning_rates(config):
    milestones = [
        max(1, int(f * config["epochs"])) for f in config["lr_decay_fractions"]
    ]
    return [
        config["learning_rate"]
        * config["lr_decay_factor"]
        ** sum(milestone < epoch for milestone in milestones)
        for epoch in range(1, EPOCH + 1)
    ]


def verify(root):
    first = root / "attempt_01_gp_recipe"
    second = root / "attempt_02_longer"
    manifests = [read_json(folder / "config.json") for folder in (first, second)]
    seeds = manifests[0]["experiment"]["seeds"]
    if len(seeds) != 3 or manifests[1]["experiment"]["seeds"] != seeds:
        raise ValueError("Both attempts must contain the same three paired seeds")
    # Check every prerequisite before opening any ensemble. validation.json is
    # written after the NPZ closes, so its candidate row certifies write completion.
    candidate_rows = {}
    for attempt in (first, second):
        for seed in seeds:
            folder = attempt / f"seed_{seed}"
            archive = folder / f"validation_epoch_{EPOCH:04d}.npz"
            if not archive.is_file() or archive.stat().st_size == 0:
                raise FileNotFoundError(f"Epoch-{EPOCH} archive not ready: {archive}")
            row = epoch_row(folder)
            if row["samples"] != archive.name:
                raise ValueError(
                    f"Candidate row references an unexpected archive: {folder}"
                )
            candidate_rows[(attempt.name, seed)] = row
    left_training, right_training = [m["training"] for m in manifests]
    shared_training = all(
        left_training[key] == right_training[key]
        for key in set(left_training) | set(right_training)
        if key != "epochs"
    )
    sampling_keys = ("samples", "sample_batch_size", "device", "threads")
    conditions = {
        "identical_architecture": manifests[0]["architecture"]
        == manifests[1]["architecture"],
        "identical_training_settings_except_total_epochs": shared_training,
        "identical_learning_rates_during_epochs_1_through_150": (
            learning_rates(left_training) == learning_rates(right_training)
        ),
        "identical_sampling_settings": all(
            manifests[0]["experiment"][key] == manifests[1]["experiment"][key]
            for key in sampling_keys
        ),
        "identical_dataset_hash_manifest": manifests[0]["data_sha256"]
        == manifests[1]["data_sha256"],
        "neither_attempt_reuses_gp_weights": all(
            m["recovered_gp_recipe"]["gp_weights_reused"] is False for m in manifests
        ),
    }
    result = {
        "check": "shared optimization prefix and exact saved validation ensembles",
        "epoch": EPOCH,
        "checked_utc": datetime.now(UTC).isoformat(),
        "expected_reason": (
            "Both runs train epochs 1..150 at the same learning rate. The 200-epoch "
            "run's first LR drop occurs after epoch150, affecting epoch151 onward. "
            "Extra epoch50/100 CRPS calls should leave training RNG and model modes "
            "unchanged. The 600-epoch run first changes LR after epoch450."
        ),
        "scope": (
            "Exact saved output and training-loss-prefix check; parameter-by-parameter "
            "weight equality is not claimed because both epoch150 checkpoints were "
            "not separately archived. No new training, sampling, test or fine truth."
        ),
        "manifest_conditions": conditions,
        "config_sha256": {
            folder.name: file_hash(folder / "config.json") for folder in (first, second)
        },
        "score_intervals": [m["experiment"]["score_interval"] for m in manifests],
        "seeds": {},
    }
    for seed in seeds:
        folders = [attempt / f"seed_{seed}" for attempt in (first, second)]
        plans = [read_json(folder / "seeds.json") for folder in folders]
        prefixes = [training_prefix(folder) for folder in folders]
        rows = [candidate_rows[(attempt.name, seed)] for attempt in (first, second)]
        archives = [folder / f"validation_epoch_{EPOCH:04d}.npz" for folder in folders]
        entry = {
            "identical_seed_plans": plans[0] == plans[1],
            "identical_training_loss_lr_validation_loss_prefix": prefixes[0]
            == prefixes[1],
            "identical_reported_scores": rows[0]["scores"] == rows[1]["scores"],
            "identical_reported_linear_scores": rows[0]["linear"] == rows[1]["linear"],
            "data_files_match_manifest": all(
                file_hash(folder / "train_validation.npz")
                == manifests[index]["data_sha256"][str(seed)]
                for index, folder in enumerate(folders)
            ),
            "reported_crps": [row["scores"]["crps"] for row in rows],
            "arrays": {},
        }
        with (
            np.load(archives[0], allow_pickle=False) as left,
            np.load(archives[1], allow_pickle=False) as right,
        ):
            entry["identical_archive_keys"] = set(left.files) == set(right.files)
            for name in sorted(set(left.files) | set(right.files)):
                if name not in left or name not in right:
                    entry["arrays"][name] = {
                        "exactly_equal": False,
                        "missing_in_one_archive": True,
                    }
                    continue
                a, b = left[name], right[name]
                same_shape = a.shape == b.shape
                same_dtype = a.dtype == b.dtype
                equal = same_shape and same_dtype and np.array_equal(a, b)
                record = {
                    "exactly_equal": bool(equal),
                    "shapes": [list(a.shape), list(b.shape)],
                    "dtypes": [a.dtype.str, b.dtype.str],
                    "array_sha256": [array_hash(a), array_hash(b)],
                }
                if same_shape and a.dtype.kind in "fiu" and b.dtype.kind in "fiu":
                    record["max_absolute_difference"] = float(
                        np.max(np.abs(a.astype(float) - b.astype(float)))
                    )
                entry["arrays"][name] = record
        entry["passed"] = all(
            value for key, value in entry.items() if isinstance(value, bool)
        ) and all(record["exactly_equal"] for record in entry["arrays"].values())
        result["seeds"][str(seed)] = entry
    result["passed"] = all(conditions.values()) and all(
        entry["passed"] for entry in result["seeds"].values()
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root.resolve()
    destination = root / f"shared_prefix_epoch_{EPOCH:04d}.json"
    if destination.exists():
        raise FileExistsError(f"Verification already exists: {destination}")
    result = verify(root)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps({"passed": result["passed"], "report": str(destination)}, indent=2)
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
