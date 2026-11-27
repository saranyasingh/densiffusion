# Dense history, naturally sparse future: EPA PM2.5

Many EPA PM2.5 monitors sampled daily for years and then moved to the national
1-in-3 or 1-in-6 day calendar. Nothing is removed by hand: the sparse days are
the agency's schedule, and every test score is a real reading.

## Setup

- **Data:** EPA AQS daily PM2.5 (parameter 88101), 1999–2025, public domain.
  `scripts/prepare_aqs.py` builds `data/aqs_pm25.npz` (2,242 sites × 9,862
  days) and lists the sites whose daily era is followed by a sparse era.
- **Sites:** 27 sites with at least 3 sparse years and a neighbour (within
  100 km) reporting on at least 90% of their missing days; 3 had too few
  forecast days, leaving 24. Up to 8 neighbours per site, chosen because they
  report in both eras.
- **Task:** in the sparse era the site reports on day F − 3 and next on F.
  Predict F from the site's earlier readings and its neighbours up to F − 1.
- **Training:** daily era only (first 80% train, last 20% validation). CSDI
  sees one channel per site, trained on daily-era windows hidden in exactly
  the sparse pattern; only entries that really exist are scored.
- **Baselines:** last reading, recent mean, ridge on the inputs the sparse
  era provides (`ridge_direct`), and same-day neighbour fill followed by a
  ridge forecaster trained on complete histories (`fill_then_ridge`).
- **Ours:** CSDI forecasting directly, and CSDI's fills fed draw by draw to the
  same complete-history forecaster (`csdi_fill_then_ridge`).
- **Scores:** CRPS on the original scale. One seed, Modal L4.

## Results

| | Median vs best baseline | ≥2% better | Better at all | ≥2% worse |
|---|---|---|---|---|
| CSDI fill → forecast | −1.0% | 7 / 24 | 15 / 24 | 5 / 24 |
| CSDI forecasting directly | +14% | 3 / 24 | 3 / 24 | 21 / 24 |

- `ridge_direct` is the best baseline at 17 sites and `fill_then_ridge` at 7.
- In the daily era, forecasting with the **true** missing days beats the best
  baseline by only 1.1% (median). Perfect filling would barely help, because
  the neighbours already carry the information.
- Caveats: one seed, and the win rule was not fixed before running.

## Reproduce

```sh
uv run python scripts/prepare_aqs.py
modal run --detach --env Densiffusion scripts/modal_run.py \
    --script run_aqs_forecast.py --spawn --timeout-minutes 180 --args \
    "--skip 0 --top 7 --output outputs/aqs_forecast/part0; \
     --skip 7 --top 7 --output outputs/aqs_forecast/part1; \
     --skip 14 --top 7 --output outputs/aqs_forecast/part2; \
     --skip 21 --top 7 --output outputs/aqs_forecast/part3"
modal volume get --env Densiffusion densiffusion-outputs aqs_forecast outputs/
uv run python scripts/summarize_aqs_forecast.py outputs/aqs_forecast
```
