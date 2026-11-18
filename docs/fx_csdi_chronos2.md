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
