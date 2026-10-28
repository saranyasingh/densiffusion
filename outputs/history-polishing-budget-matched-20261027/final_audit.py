"""Audit a completed run and preserve a final artifact manifest, without fitting."""

import hashlib
import json
import re
import subprocess
import tarfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
PRIMARY = ROOT / "primary"


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    protected = read(ROOT / "protected_artifacts.json")
    assert all(digest(REPO / p) == h for p, h in protected.items())
    protocol = read(ROOT / "protocol.json")
    assert all(digest(REPO / p) == h for p, h in protocol["script_sha256"].items())
    snapshot_count = 0
    with tarfile.open(PRIMARY / "source.tar.gz") as archive:
        for item in archive.getmembers():
            if item.isfile():
                with archive.extractfile(item) as stream:
                    expected = hashlib.file_digest(stream, "sha256").hexdigest()
                assert digest(REPO / item.name) == expected, item.name
                snapshot_count += 1
    frozen = read(PRIMARY / "FROZEN.json")
    assert all(digest(PRIMARY / p) == h for p, h in frozen["files"].items())
    verification = read(PRIMARY / "verification.json")
    assert verification["passed"] and not verification["training_only"]
    bootstrap = read(ROOT / "bootstrap_recomputation.json")
    assert len(bootstrap["checks"]) == 9
    assert all(row["all_four_metrics_passed"] for row in bootstrap["checks"])
    review = read(ROOT / "plot_review.json")
    assert review["inspected"] and len(review["plots"]) == 4
    for item in review["plots"]:
        assert digest(ROOT / item["png"]) == item["sha256"]
    report = (ROOT / "REPORT.md").read_text()
    for target in re.findall(r"\]\(([^)]+)\)", report):
        assert not target.startswith(("http:", "https:"))
        assert (ROOT / target).exists(), target
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True)
    assert head == (ROOT / "initial_git_head.txt").read_text()
    status = subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True)
    (ROOT / "final_git_head.txt").write_text(head)
    (ROOT / "final_git_status.txt").write_text(status)
    checks = {
        "passed": True,
        "completed_utc": datetime.now(UTC).isoformat(),
        "protected_prior_files_unchanged": len(protected),
        "scientific_source_hashes_match_protocol": True,
        "snapshot_files_match_current_repo": snapshot_count,
        "frozen_training_artifacts_unchanged": len(frozen["files"]),
        "independent_pipeline_verification": verification["passed"],
        "bootstrap_intervals_independently_recomputed": 36,
        "plots_inspected": len(review["plots"]),
        "report_links_resolve": True,
        "implementation_checks": read(ROOT / "implementation_checks.json"),
        "git_head_unchanged": head.strip(),
        "no_commit_or_push": True,
        "test_based_tuning": False,
        "scope": "one prospective matrix-work-matched polishing budget; no optimality or densification claim",
    }
    write(ROOT / "final_checks.json", checks)
    artifacts = {
        str(p.relative_to(ROOT)): {"sha256": digest(p), "bytes": p.stat().st_size}
        for p in sorted(ROOT.rglob("*"))
        if p.is_file() and p.name != "artifact_manifest.json"
    }
    write(ROOT / "artifact_manifest.json", artifacts)
    print(json.dumps({**checks, "manifest_files": len(artifacts)}, indent=2))


if __name__ == "__main__":
    main()
