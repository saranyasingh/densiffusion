# Sparse multivariate TS → CSDI → Chronos-2

End-to-end comparison of [Chronos-2](https://arxiv.org/abs/2610.41956)
forecasting on a sparsely observed multivariate series versus the same series
after CSDI imputation.

## Recommended default: electricity + block missing

Hourly LSTNet electricity (periodic daily/weekly load). Default missingness
hides **12h blocks** (then observes 12h), so linear fill must bridge diurnal
peaks. Chronos evaluation is **zero-shot only** (`ENABLE_CHRONOS_FINETUNE =
False` in `scripts/run_csdi_chronos2.py`; flip to restore LoRA fine-tuning).

Anchors and metrics:

- **seasonal naive** and **last value**
- **dense_oracle**: Chronos-2 on true dense history (upper bound)
- **zeroshot_sparse / zeroshot_csdi / zeroshot_linear**
- Errors: MAE, RMSE, sMAPE, **MASE**, WQL, **SQL**

### Commands

```sh
uv sync --extra research --extra chronos
uv run python scripts/prepare_electricity.py --output data/electricity.npz

uv run --extra research --extra chronos python scripts/run_csdi_chronos2.py \
    --data data/electricity.npz --output outputs/electricity_csdi_chronos2 \
    --device cuda

# harder random sparsity
uv run --extra research --extra chronos python scripts/run_csdi_chronos2.py \
    --data data/electricity.npz --output outputs/electricity_random \
    --missing-mode random --keep-probability 0.25 --device cuda
```

Cluster:

```sh
sbatch scripts/cluster/electricity_csdi_chronos2.sbatch
```

## FX preset

```sh
uv run python scripts/prepare_fx.py --output data/fx.npz
uv run --extra research --extra chronos python scripts/run_csdi_chronos2.py \
    --preset fx --data data/fx.npz --output outputs/fx_csdi_chronos2 --device cuda
```

## Reading `results.json`

- `imputation.csdi_crps` vs `imputation.linear_crps`
- `forecast.zeroshot_*`, `forecast.dense_oracle`, `forecast.seasonal_naive`
- `deltas_csdi_minus_sparse` / `deltas_csdi_minus_linear`

## Plots

The runner writes figures into the output directory when matplotlib is available.
Re-plot any finished run with:

```sh
uv run --with matplotlib python scripts/plot_csdi_chronos2_results.py \
    --run outputs/electricity_csdi_chronos2_<jobid>
```

Artifacts: `table.png`, `bars.png`, `imputation.png`, `examples.png`
(examples need `imputed.npz`).

## Self-supervised sparse training

The supervised runner above trains CSDI with dense ground truth at every
base-missing time. [`scripts/run_csdi_chronos2_selfsup.py`](../scripts/run_csdi_chronos2_selfsup.py)
keeps the same electricity → Chronos-2 evaluation, but training is nested:

1. Simulate a sparse observation set (`--missing-mode block` or `random`).
2. Among *available* times only, hide a fraction `--ssl-mask-rate` as SSL
   targets. Naturally missing times never enter the loss or conditioning.
3. Densify all base-missing times; zero-shot Chronos on sparse / CSDI / linear /
   dense-oracle contexts.

```sh
uv run --extra research --extra chronos python \
    scripts/run_csdi_chronos2_selfsup.py \
    --data data/electricity.npz \
    --output outputs/elec_selfsup_block \
    --missing-mode block --ssl-mask-rate 0.25 \
    --supervised-baseline --device cuda
```

`results.json` adds `protocol`, `ssl_mask_rate`, `observed_fraction`, and
`imputation.ssl_crps` (held-out SSL targets). With `--supervised-baseline`,
a sibling `supervised/` run uses the old dense-target protocol under the same
base mask.

### Threshold sweeps (1D)

Two separate sweeps per missing mode (not a 2D grid):

| Sweep | Fixed | Varies |
|-------|-------|--------|
| **A** base sparsity | `ssl_mask_rate=0.25` | random `keep_probability`, or block observe fraction in a 24h period |
| **B** SSL hardness | block 12/12 or random keep 0.5 | `ssl_mask_rate` |

```sh
# list the 20 cells (A/B × block/random)
uv run python scripts/sweep_csdi_chronos2_selfsup.py --list

# one cell locally
uv run --extra research --extra chronos python \
    scripts/sweep_csdi_chronos2_selfsup.py \
    --sweep A --missing-mode block --index 3 \
    --data data/electricity.npz --output-root outputs \
    --device cuda --supervised-baseline

# cluster array (tasks 0–19)
sbatch scripts/cluster/electricity_csdi_chronos2_selfsup.sbatch

# threshold curves once runs finish
uv run --with matplotlib python scripts/plot_selfsup_thresholds.py \
    --root outputs
```

Plots: `thresholds_{block|random}_{A|B}.png` plus `thresholds_summary.json`
(includes a steepest-drop `threshold_hint_x` on the self-sup forecast curve).
