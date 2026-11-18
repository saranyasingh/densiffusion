# Sparse multivariate TS → CSDI → Chronos-2

End-to-end comparison of [Chronos-2](https://arxiv.org/abs/2610.41956)
forecasting on a sparsely observed multivariate series versus the same series
after CSDI imputation.

## Recommended default: electricity

Hourly LSTNet electricity (periodic daily/weekly load). Default preset keeps
every 4th hour so linear fill must bridge peak/off-peak gaps — a harder setting
than smooth daily FX.

- **Channels.** Top-20 highest-variance clients by default (`prepare_electricity.py
  --n-channels 20`); full 321 makes CSDI feature attention very expensive.
- **Windows.** 97-hour windows, stride 48; Chronos context 168h, horizon 24h.
- **Strong CSDI.** Reference-size net, quadratic 50-step schedule, LR/weight decay
  (see `docs/csdi.md`).
- **Chronos-2 arms.** Fine-tuned sparse / CSDI-imputed / linear-imputed, plus
  zero-shot sparse & CSDI. Scored on the shared coarse grid (MAE/RMSE/sMAPE/WQL).

### Commands

```sh
uv sync --extra research --extra chronos
uv run python scripts/prepare_electricity.py --output data/electricity.npz

uv run --extra research --extra chronos python scripts/run_csdi_chronos2.py \
    --data data/electricity.npz --output outputs/electricity_csdi_chronos2 \
    --device cuda

# smoke
uv run --extra research --extra chronos python scripts/run_csdi_chronos2.py \
    --data data/electricity.npz --output outputs/electricity_quick \
    --quick --device cpu
```

Cluster:

```sh
sbatch scripts/cluster/electricity_csdi_chronos2.sbatch
```

## FX preset (legacy)

Daily exchange rates remain available if you want the smooth baseline contrast:

```sh
uv run python scripts/prepare_fx.py --output data/fx.npz
uv run --extra research --extra chronos python scripts/run_csdi_chronos2.py \
    --preset fx --data data/fx.npz --output outputs/fx_csdi_chronos2 --device cuda
```

## Reading `results.json`

- `imputation.csdi_crps` vs `imputation.linear_crps`
- `forecast.sparse` / `forecast.csdi_imputed` / `forecast.linear_imputed`
- `forecast.zeroshot_*`
- `deltas_csdi_minus_sparse` — negative means CSDI+Chronos wins
