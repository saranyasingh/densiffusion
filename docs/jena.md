# Jena solar radiation: does the quasi-periodic result carry over?

The quasi-periodic sweep (section 3 of
[pristi_experiments.md](pristi_experiments.md)) used synthetic GP paths. This
experiment repeats it on measured data: shortwave downward radiation (SWDR,
W/m^2) from the Max Planck Institute for Biogeochemistry's rooftop weather
station in Jena, recorded every 10 minutes since late 2003. The data are
published under CC-BY-4.0 at
<https://www.bgc-jena.mpg.de/wetter/weather_data.html>.

Solar radiation has a fixed period, P = 24 h, so the coarse spacing dt sets
P/dt as in the synthetic sweep. Unlike the synthetic wave, it is zero at
night and peaks sharply by day, and clouds add irregular dips.

## Setup

- **Data:** `scripts/prepare_jena.py` downloads the station's files and puts
  SWDR on a regular 10-minute grid (0.16% missing). Years 2004-2018 are for
  training, 2019-2020 for validation and 2021-2024 for testing.
- **Sweep:** dt in {880, 560, 480, 400, 240, 160, 80} minutes, i.e. P/dt in
  {1.64, 2.57, 3, 3.6, 6, 9, 18}. Each dt is a multiple of 80 minutes, so the
  x8 fine grid lies on the measurements.
- **Paths:** 256 / 32 / 64 train / validation / test paths of 33 coarse
  points, evenly spaced in time within each split, skipping any with missing
  values. Windows (9 history and 4 future coarse points), model sizes,
  training schedule and 100 draws per window match the synthetic sweep.
- **Normalisation:** as in the synthetic sweep, but with the window scale
  floored at 20 W/m^2 (about 2% of peak radiation). At dt = 80 minutes, 15% of
  histories fall entirely at night, and without a floor their near-zero scale
  would inflate the rest of the window by up to 2,000 times.
- **Methods:** sparse, sparse with all phase shifts, coarse-data x2 and x8,
  interpolated x2 and x8, fine-data x2 and x8, fine-data x2 with interpolated
  history, and repeating the last value, all defined as in the synthetic
  sweep.
- **Perfect filler:** there is no exact GP, so the perfect filler gives the fine-data
  forecaster the true fine history (one history, 100 futures).
- **Scores:** CRPS relative to the sparse forecaster, with 95% bootstrap
  intervals over test paths for the gain over sparse.

```sh
uv run python scripts/prepare_jena.py
for task in jena_1.64 jena_2.57 jena_3 jena_3.6 jena_6 jena_9 jena_18; do
    uv run --extra research python scripts/run_jena.py run --model pristi \
        --device mps --task $task --output outputs/jena/pristi --threads 1 &
done; wait
uv run --extra research --with matplotlib python scripts/run_jena.py report \
    --output outputs/jena/pristi
```

## Differences from the synthetic sweep

- **No best-possible reference:** without an exact GP, there is no "1.00 =
  perfect" line, and the perfect filler uses the true history instead of posterior
  draws.
- **Phase is locked to the clock:** at integer P/dt (3, 6, 9, 18) the coarse
  points fall at the same times of day every day. The synthetic wave's phase
  drifted.
- **Time of day is a strong signal the models do not see:** windows carry no
  timestamps. A forecaster given the time of day, or a physical clear-sky
  model, would be a fairer real-world baseline and has not been run.

## Results (PriSTI, one seed)

Forecast CRPS relative to the sparse forecaster (below 1 is better; `+` / `-`
significantly better / worse than sparse, 95% bootstrap intervals over test
paths):

| P/dt | dt | Sparse CRPS (W/m^2) | Coarse-data x2 | Interpolated x2 | Fine-data x2 | Fine-data x2, interp. history | Perfect filler x2 (true history) | Sparse, all phases | Repeat last value |
|---|---|---|---|---|---|---|---|---|---|
| 1.64 | 14.7 h | 63.8 | 1.215 - | **0.796 +** | 1.049 | 1.343 - | 0.849 + | 1.072 - | 3.25 |
| 2.57 | 9.3 h | 56.7 | 1.354 - | 0.849 + | 0.973 | 0.957 | **0.768 +** | 1.070 - | 3.51 |
| 3 | 8 h | 40.0 | 1.371 - | 0.970 | 0.955 | 0.933 + | **0.917 +** | 1.017 | 4.66 |
| 3.6 | 6.7 h | 52.3 | 1.210 - | 0.890 + | 1.051 | 0.925 + | **0.875 +** | 0.992 | 3.22 |
| 6 | 4 h | 44.2 | 1.157 - | 0.997 | 1.049 | 1.046 | **0.956** | 0.969 | 4.81 |
| 9 | 2.7 h | 42.4 | 1.112 - | 1.025 | 1.049 | 1.044 | 1.020 | **0.969** | 4.32 |
| 18 | 1.3 h | 61.4 | 1.030 | 1.035 | 1.027 | 1.012 | **0.988** | 1.023 | 2.00 |

x8 versions are worse than x2 almost everywhere (full table in
`outputs/jena/pristi/results.md`). Densified history CRPS at x8: fine-data
imputers 18-50 W/m^2, coarse-trained 20-235 W/m^2, linear interpolation's
absolute error 24-148 W/m^2.

`sweep_focus.png` shows the x2 methods; `sweep.png` shows every method.

### Findings

- **The synthetic pattern carries over.** Forecasting on a 2x grid filled by
  straight lines, with coarse data only, is significantly better than sparse
  for P/dt = 1.64, 2.57 and 3.6, by 11-20%. From P/dt = 6 upwards no x2
  method is significantly better than sparse.
- **Learned fine-data densification does not help.** Fine-data x2 is never
  significantly better than sparse, and is worse than interpolated x2 at
  P/dt = 1.64, 2.57 and 3.6, although its densified x8 histories are 1.8-3
  times more accurate than interpolation there.
- **The true fine history does help.** The same forecaster given the
  measured fine history (perfect filler) beats sparse by 8-23% for P/dt <= 3.6. Given
  learned or straight-line histories it gains little, so here the learned
  densifier is the bottleneck, unlike in the synthetic sweep.
- **Coarse-only learned densification hurts everywhere** (11-37% worse than
  sparse up to P/dt = 9).
- **Fine data alone do not help:** the sparse forecaster trained on all phase
  shifts is no better than sparse.
- Sparse is a sensible baseline: repeating the last value is 2-5 times worse.

### Caveats

- One seed and one model; CSDI, SSSD and TimeAutoDiff have not been run.
- The models see no timestamps; a time-of-day or clear-sky baseline might
  beat all of them.
- At integer P/dt the coarse points fall at fixed times of day, so P/dt = 3
  and 6 may behave differently from the non-integer ratios.
