"""One-page matplotlib overview of self-sup threshold sweeps.

    uv run --with matplotlib python scripts/plot_selfsup_overview.py \
        --root densiffusion_results
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

RUN_RE = re.compile(
    r"elec_selfsup_(?P<mode>block|random)_"
    r"(?P<sweep>A|B)_"
    r"(?P<label>[^/]+?)(?:_\d+(?:_\d+)?)?$"
)

BG = "#f7f3ea"
PANEL = "#fffdf8"
HEADER = "#ebe3d3"
BORDER = "#c9bfae"
TEXT = "#2f2f2f"
MUTED = "#6e6a63"
ACCENT = "#2f6f6a"
ACCENT2 = "#b45a3c"
ACCENT3 = "#6b7c3d"
SPARSE = "#7a6f64"
SUP = "#3d5a80"
BEST = "#dceeea"


def load_rows(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.rglob("results.json")):
        if path.parent.name == "supervised":
            continue
        match = RUN_RE.search(path.parent.name)
        if not match:
            continue
        meta = match.groupdict()
        payload = json.loads(path.read_text())
        forecast = payload["forecast"]
        imputation = payload["imputation"]
        row = {
            **meta,
            "obs": float(payload["observed_fraction"]),
            "ssl": float(payload["ssl_mask_rate"]),
            "self_smape": forecast["zeroshot_csdi"]["smape"],
            "sparse_smape": forecast["zeroshot_sparse"]["smape"],
            "lin_smape": forecast["zeroshot_linear"]["smape"],
            "oracle_smape": forecast["dense_oracle"]["smape"],
            "self_crps": imputation["csdi_crps"],
            "lin_crps": imputation["linear_crps"],
        }
        if meta["sweep"] == "A" and meta["mode"] == "block":
            row["x"] = float(meta["label"].replace("obs", "")) / 24.0
        elif meta["sweep"] == "A" and meta["mode"] == "random":
            row["x"] = float(meta["label"].replace("keep", ""))
        else:
            row["x"] = float(meta["label"].replace("ssl", ""))
        supervised = path.parent / "supervised" / "results.json"
        if supervised.exists():
            supervised_payload = json.loads(supervised.read_text())
            row["sup_smape"] = supervised_payload["forecast"]["zeroshot_csdi"]["smape"]
            row["sup_crps"] = supervised_payload["imputation"]["csdi_crps"]
        rows.append(row)
    return rows


def style_ax(ax, title: str) -> None:
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.grid(True, color=BORDER, alpha=0.45, linewidth=0.6)
    ax.set_title(title, loc="left", color=TEXT, fontsize=10, pad=6)


def plot_smape(ax, points: list[dict], *, xlabel: str, with_sup: bool) -> None:
    points = sorted(points, key=lambda row: row["x"])
    xs = [row["x"] for row in points]
    ax.plot(
        xs,
        [row["self_smape"] for row in points],
        "o-",
        color=ACCENT,
        lw=2,
        label="Self-sup CSDI",
        ms=5,
    )
    if with_sup and all("sup_smape" in row for row in points):
        ax.plot(
            xs,
            [row["sup_smape"] for row in points],
            "s--",
            color=SUP,
            lw=2,
            label="Supervised",
            ms=5,
        )
    ax.plot(
        xs,
        [row["sparse_smape"] for row in points],
        "o-",
        color=SPARSE,
        lw=1.6,
        label="Sparse",
        ms=4,
    )
    ax.plot(
        xs,
        [row["lin_smape"] for row in points],
        "o-",
        color=ACCENT2,
        lw=1.6,
        label="Linear",
        ms=4,
    )
    ax.plot(
        xs,
        [row["oracle_smape"] for row in points],
        "o-",
        color=ACCENT3,
        lw=1.6,
        label="Dense oracle",
        ms=4,
    )
    ax.set_xlabel(xlabel, color=TEXT, fontsize=8)
    ax.set_ylabel("sMAPE (lower better)", color=TEXT, fontsize=8)
    ax.legend(frameon=False, fontsize=7, loc="upper right")


def draw_table(ax, headers: list[str], data: list[list[str]], title: str) -> None:
    ax.set_axis_off()
    ax.set_title(title, loc="left", color=TEXT, fontsize=10, pad=4)
    n_rows, n_cols = len(data), len(headers)
    best: dict[int, float] = {}
    for col in range(1, n_cols):
        values = []
        for row in data:
            try:
                values.append(float(str(row[col]).replace("%", "")))
            except ValueError:
                continue
        if values:
            best[col] = min(values)

    cell_h = 1.0 / (n_rows + 1)
    for col, header in enumerate(headers):
        x0 = col / n_cols
        ax.add_patch(
            FancyBboxPatch(
                (x0, 1 - cell_h),
                1 / n_cols,
                cell_h,
                boxstyle="square,pad=0",
                facecolor=HEADER,
                edgecolor=BORDER,
                linewidth=0.4,
                transform=ax.transAxes,
                clip_on=False,
            )
        )
        ax.text(
            x0 + 0.5 / n_cols,
            1 - 0.5 * cell_h,
            header,
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
            color=TEXT,
            transform=ax.transAxes,
        )
    for index, row in enumerate(data):
        y0 = 1 - (index + 2) * cell_h
        for col, value in enumerate(row):
            x0 = col / n_cols
            face = PANEL if index % 2 == 0 else BG
            try:
                number = float(str(value).replace("%", ""))
                if col in best and abs(number - best[col]) <= 1e-9 * max(
                    1.0, abs(best[col])
                ):
                    face = BEST
            except ValueError:
                pass
            ax.add_patch(
                FancyBboxPatch(
                    (x0, y0),
                    1 / n_cols,
                    cell_h,
                    boxstyle="square,pad=0",
                    facecolor=face,
                    edgecolor=BORDER,
                    linewidth=0.35,
                    transform=ax.transAxes,
                    clip_on=False,
                )
            )
            ax.text(
                x0 + 0.5 / n_cols,
                y0 + 0.5 * cell_h,
                str(value),
                ha="center",
                va="center",
                fontsize=7,
                color=TEXT,
                transform=ax.transAxes,
            )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("densiffusion_results"))
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: ROOT/selfsup_thresholds_overview.png",
    )
    args = parser.parse_args(argv)
    output = args.output or (args.root / "selfsup_thresholds_overview.png")
    rows = load_rows(args.root)
    if len(rows) != 20:
        raise SystemExit(f"expected 20 self-sup cells, found {len(rows)} under {args.root}")

    fig = plt.figure(figsize=(14.5, 16.5), dpi=160, facecolor=BG)
    grid = gridspec.GridSpec(
        5,
        2,
        figure=fig,
        height_ratios=[0.55, 1.15, 1.0, 1.15, 1.0],
        hspace=0.38,
        wspace=0.22,
        left=0.06,
        right=0.98,
        top=0.955,
        bottom=0.03,
    )

    banner = fig.add_subplot(grid[0, :])
    banner.set_axis_off()
    banner.text(
        0.0,
        0.78,
        "Self-supervised CSDI → Chronos-2 threshold sweeps",
        fontsize=16,
        fontweight="bold",
        color=TEXT,
        transform=banner.transAxes,
    )
    banner.text(
        0.0,
        0.42,
        "Job 63098029 · electricity · 20/20 cells · lower sMAPE / CRPS is better",
        fontsize=10,
        color=MUTED,
        transform=banner.transAxes,
    )
    banner.text(
        0.0,
        0.08,
        "Threshold: self-sup helps once ≥ ~50% observed "
        "(block obs≥12/24 or random keep≥0.5). "
        "Prefer SSL rate ≈ 0.25–0.4 (block) or 0.25 / 0.55 (random).",
        fontsize=9.5,
        color=TEXT,
        transform=banner.transAxes,
    )

    block_a = [row for row in rows if row["sweep"] == "A" and row["mode"] == "block"]
    random_a = [row for row in rows if row["sweep"] == "A" and row["mode"] == "random"]
    ax_ba = fig.add_subplot(grid[1, 0])
    style_ax(ax_ba, "Sweep A · block missing · Chronos sMAPE")
    plot_smape(
        ax_ba,
        block_a,
        xlabel="Observe fraction within 24h period",
        with_sup=True,
    )
    ax_ba.axvline(0.5, color=ACCENT, ls=":", lw=1.2, alpha=0.8)
    ax_ba.text(0.505, ax_ba.get_ylim()[1] * 0.92, "threshold ≈ 0.5", color=ACCENT, fontsize=7)

    ax_ra = fig.add_subplot(grid[1, 1])
    style_ax(ax_ra, "Sweep A · random missing · Chronos sMAPE")
    plot_smape(ax_ra, random_a, xlabel="Keep probability", with_sup=True)
    ax_ra.axvline(0.5, color=ACCENT, ls=":", lw=1.2, alpha=0.8)
    ax_ra.text(0.505, ax_ra.get_ylim()[1] * 0.92, "threshold ≈ 0.5", color=ACCENT, fontsize=7)

    ax_bat = fig.add_subplot(grid[2, 0])
    block_a = sorted(block_a, key=lambda row: row["x"])
    draw_table(
        ax_bat,
        ["cell", "self", "sup", "sparse", "linear", "oracle", "self CRPS", "sup CRPS"],
        [
            [
                row["label"],
                f"{row['self_smape']:.3f}",
                f"{row['sup_smape']:.3f}",
                f"{row['sparse_smape']:.3f}",
                f"{row['lin_smape']:.3f}",
                f"{row['oracle_smape']:.3f}",
                f"{row['self_crps']:.2f}",
                f"{row['sup_crps']:.2f}",
            ]
            for row in block_a
        ],
        "Sweep A · block · sMAPE / CRPS (green = best in column)",
    )

    ax_rat = fig.add_subplot(grid[2, 1])
    random_a = sorted(random_a, key=lambda row: row["x"])
    draw_table(
        ax_rat,
        ["cell", "self", "sup", "sparse", "linear", "oracle", "self CRPS", "sup CRPS"],
        [
            [
                row["label"],
                f"{row['self_smape']:.3f}",
                f"{row['sup_smape']:.3f}",
                f"{row['sparse_smape']:.3f}",
                f"{row['lin_smape']:.3f}",
                f"{row['oracle_smape']:.3f}",
                f"{row['self_crps']:.2f}",
                f"{row['sup_crps']:.2f}",
            ]
            for row in random_a
        ],
        "Sweep A · random · sMAPE / CRPS",
    )

    block_b = [row for row in rows if row["sweep"] == "B" and row["mode"] == "block"]
    random_b = [row for row in rows if row["sweep"] == "B" and row["mode"] == "random"]
    ax_bb = fig.add_subplot(grid[3, 0])
    style_ax(ax_bb, "Sweep B · block 12/12 · Chronos sMAPE vs SSL rate")
    plot_smape(
        ax_bb,
        block_b,
        xlabel="SSL mask rate (of observed points)",
        with_sup=False,
    )

    ax_rb = fig.add_subplot(grid[3, 1])
    style_ax(ax_rb, "Sweep B · random keep=0.5 · Chronos sMAPE vs SSL rate")
    plot_smape(
        ax_rb,
        random_b,
        xlabel="SSL mask rate (of observed points)",
        with_sup=False,
    )

    ax_bbt = fig.add_subplot(grid[4, 0])
    block_b = sorted(block_b, key=lambda row: row["x"])
    draw_table(
        ax_bbt,
        ["ssl", "self sMAPE", "sparse", "linear", "self CRPS", "lin CRPS", "note"],
        [
            [
                f"{row['ssl']:.2f}",
                f"{row['self_smape']:.3f}",
                f"{row['sparse_smape']:.3f}",
                f"{row['lin_smape']:.3f}",
                f"{row['self_crps']:.2f}",
                f"{row['lin_crps']:.2f}",
                (
                    "best"
                    if abs(row["ssl"] - 0.4) < 1e-9
                    else (
                        "helps"
                        if row["self_smape"] < row["sparse_smape"]
                        else "hurts"
                    )
                ),
            ]
            for row in block_b
        ],
        "Sweep B · block · SSL rate table",
    )

    ax_rbt = fig.add_subplot(grid[4, 1])
    random_b = sorted(random_b, key=lambda row: row["x"])
    notes = {0.1: "mild", 0.25: "strong", 0.4: "≈ sparse", 0.55: "best", 0.7: "hurts"}
    draw_table(
        ax_rbt,
        ["ssl", "self sMAPE", "sparse", "linear", "self CRPS", "lin CRPS", "note"],
        [
            [
                f"{row['ssl']:.2f}",
                f"{row['self_smape']:.3f}",
                f"{row['sparse_smape']:.3f}",
                f"{row['lin_smape']:.3f}",
                f"{row['self_crps']:.2f}",
                f"{row['lin_crps']:.2f}",
                notes.get(round(row["ssl"], 2), ""),
            ]
            for row in random_b
        ],
        "Sweep B · random · SSL rate table",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor=fig.get_facecolor())
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
