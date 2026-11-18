# Sparse FX → CSDI → Chronos-2

End-to-end comparison of [Chronos-2](https://arxiv.org/abs/2610.41956)
forecasting on a sparsely observed FX series versus the same series after
multivariate CSDI imputation.

## Why this design

- **Dataset.** LSTNet daily exchange rates (8 currencies, 1990–2016). One
  multivariate series so CSDI’s feature attention and Chronos-2’s native
  multivariate forecasting both matter.
- **Sparsity.** Regular keep-every-`sparse_step` grid (default 2). Endpoints
  stay observed so windows are well conditioned.
- **Strong CSDI.** Reference-size network, quadratic 50-step schedule
  (`beta` 1e-4→0.5), Adam weight decay, LR decays at 75%/90%, all-step
  validation — the settings selected in `docs/csdi.md`. Channel z-scores are
  fit on train observations only. Overlapping windows + multi-draw median
  aggregation produce the dense series.
- **Strong Chronos-2.** Fine-tune `amazon/chronos-2` with LoRA on the train
  split (full fine-tune optional). Arms:
  1. sparse coarse grid
  2. CSDI-imputed dense grid
  3. linear-interpolation imputed dense grid (ablation)
  4. zero-shot sparse / CSDI (no fine-tune)
- **Fair scoring.** Rolling test origins; MAE / RMSE / sMAPE / WQL on the
  shared coarse calendar grid so sparse and dense forecasters are compared
  on the same future days.

## Commands

```sh
uv sync --extra research --extra chronos
uv run python scripts/prepare_fx.py --output data/fx.npz

uv run --extra research --extra chronos python scripts/run_fx_csdi_chronos2.py \
    --data data/fx.npz --output outputs/fx_csdi_chronos2 --device cuda

# smoke
uv run --extra research --extra chronos python scripts/run_fx_csdi_chronos2.py \
    --data data/fx.npz --output outputs/fx_csdi_chronos2_quick --quick --device cpu
```

Cluster:

```sh
sbatch scripts/cluster/fx_csdi_chronos2.sbatch
```

Artifacts under `--output`: `csdi.pt`, `imputed.npz`, `chronos_*/`,
`results.json`, `training.log`.

## Reading `results.json`

- `imputation.csdi_crps` vs `imputation.linear_crps` — hidden-point ensemble CRPS
- `forecast.sparse` / `forecast.csdi_imputed` / `forecast.linear_imputed` —
  fine-tuned Chronos-2
- `forecast.zeroshot_*` — base Chronos-2
- `deltas_csdi_minus_sparse` — negative values mean CSDI+Chronos wins
