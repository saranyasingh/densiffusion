# Chronos-2 experiments: plan, main comparisons and costs

This file fixes the comparisons before the runs, so the results cannot pick
them. Tasks 2 and 3 below are run with `scripts/modal_chronos.py`. Task 1, the
quasi-periodic write-up, lives on the `nihar/exp3-unified` branch in
`docs/quasiperiodic_writeup.md`.

## Shared scoring

- **Headline score: scaled CRPS** of the 24-hour forecast against the true
  hourly values. For task 3 each chunk's CRPS is divided by the mean absolute
  observed value of its own history, so large and small customers count
  equally. Task 2 forecasts 20 customers jointly and keeps Saranya's seasonal
  scale (`scaled_crps`).
- **CRPS of a quantile forecast**: Chronos-2's quantiles (1%-99%) are sampled
  at evenly spaced probabilities, and every method is scored from the same
  number of samples.
- **Per-draw pooling**: Chronos-2 forecasts from each of K CSDI draws
  separately and the forecasts are pooled; "median" forecasts once from the
  median of the same K draws.
- **Also reported**: absolute error of the median forecast (MAE), and how well
  the gaps were filled.
- **Intervals**: task 3 resamples customers (95%); task 2 reports the spread
  over 3 seeds.
- **Seeds**: 0, 1 and 2, fixed in advance; every completed run is reported.

## Task 2: Saranya's pipeline, one forecast per draw

`run_csdi_chronos2.py`, as in her setup: 20 highest-variance customers as one
20-channel series, 12:00-23:00 hidden every day, supervised CSDI (64 channels,
4 layers), zero-shot `amazon/chronos-2`, one-week history, next-day forecast.
Two changes: forecasts start at 12:00 (`--origin-hour 12`), so every history
ends on an observed hour, and `--stitch block`, so each draw is a single CSDI
sample and the median and per-draw forecasts use the same draws.

Main comparisons, scaled CRPS, mean over 3 seeds:

| | Comparison | Question |
|---|---|---|
| T2-1 | per-draw CSDI vs CSDI median | Does keeping the uncertainty about the gaps help? |
| T2-2 | per-draw CSDI vs straight lines | Does CSDI beat the simplest fill? |
| T2-3 | each fill vs gappy history | Does filling help at all? |
| T2-4 | everything vs complete history | How much do the gaps cost, and how much is recovered? |

```sh
uv run --with modal==1.6.4 python scripts/modal_chronos.py launch \
    --script run_csdi_chronos2.py --name task2 --seeds 0 1 2 \
    --args "--missing-mode block --origin-hour 12 --per-draw-forecasts 16 --stitch block"
```

## Task 3: per customer, foundation-model style

`run_chronos_chunks.py`: 8-day chunks (7-day history, next day forecast at
midnight) from single customers, all 321 available. 20% of customers are held
out entirely; CSDI trains on the first 70% of the record, picks its checkpoint
on the next 15%, and test chunks come from the last 15%. 1000 test chunks per
group (training customers, held-out customers), fixed by `--test-seed` and the
same for every method, protocol and seed. Chunks that are mostly zero or flat
are skipped. CSDI trains for 100 epochs on 40,000 histories: the pilot (20
epochs) was still improving steadily at the full learning rate (validation
loss 0.33 to 0.25 over epochs 2-14), so 20 epochs undertrained it. This was
decided from the validation curve, before any comparison was computed. All
customers are used: the highest-variance ones are simply
the largest (median load 9,900 for the top 20, 592 over all), with about the
same relative variability, and scaling each chunk by its own level already
makes sizes count equally. `--customers N` restricts to the N
highest-variance customers if needed. Univariate CSDI (64 channels, 4 layers), 16 draws,
`autogluon/chronos-2-small`.

Gaps: a random number (1-10) of gaps, each with a random start and a random
length (1-24 hours), possibly overlapping, so the missing fraction varies
(at least 24 hours stay observed).

Configurations (folder names under the run root):

| Config | Gaps | CSDI training |
|---|---|---|
| `random_selfsup` | random | self-supervised: never sees a hidden value; learns by filling 1-2 extra observed hours per history |
| `random_supervised` | random | supervised: learns from the true values in the gaps |
| `daily_supervised` | 12:00-23:00 daily, forecasts at 12:00 | supervised: the bridge to task 2 |

Main comparisons, **held-out customers** first (generalisation), then training
customers; scaled CRPS averaged over 3 seeds:

| | Comparison | Question |
|---|---|---|
| C1 | per-draw CSDI vs straight lines (`random_selfsup`) | **Primary**: does a filler that never sees complete data beat the simplest fill? |
| C2 | per-draw CSDI vs CSDI median (same draws) | Does keeping the uncertainty about the gaps help? |
| C3 | self-supervised vs supervised CSDI (per-draw) | How much does complete training data add? |
| C4 | straight lines, and per-draw CSDI, vs gappy history | Does filling help Chronos-2 at all, given it can mask missing values itself? |
| C5 | everything vs complete history | How much do the gaps cost, and how much is recovered? |

Secondary, not used to pick headlines:

- the same comparisons by missing fraction (0-15%, 15-30%, 30-50%, 50%+);
- training vs held-out customers;
- the bridge: `daily_supervised` (per customer) against task 2 (20 customers
  jointly) with the same daily gaps. They differ in customers and data split
  too, so this is a rough comparison;
- how often a history ends in a gap (there, filling the last gap is partly
  forecasting).

```sh
for config in "random selfsup 0" "random supervised 0" "daily supervised 12"; do
    set -- $config
    uv run --with modal==1.6.4 python scripts/modal_chronos.py launch \
        --script run_chronos_chunks.py --name chunks_e100/$1_$2 --seeds 0 1 2 \
        --args "--gaps $1 --protocol $2 --start-hour $3"
done
modal volume get --env Densiffusion densiffusion-outputs nihar/chronos/chunks_e100 \
    outputs/modal/chronos/
uv run --extra research python scripts/run_chronos_chunks.py compare \
    --output outputs/modal/chronos/chunks_e100
```

## Costs

Estimates from the measured step-1 run times and Modal's published rates
(8-core CPU container about $0.51/hour; L4 with 4 cores and 32 GB about
$1.24/hour). The first run of each task is the pilot: if it costs more than
twice its estimate, the rest is not launched until rechecked.

| Task | Runs | Hours per run (est.) | Cost (est.) |
|---|---|---|---|
| 1. Training-length curve: 150 and 1350 epochs, 3 models x 3 seeds (CPU) | 18 | 0.4 / 2.5 | $13 |
| 1. Jena, PriSTI, 3x training, 3 seeds (CPU) | 3 | 3-6 | $5-9 |
| 2. Saranya's pipeline, per-draw, 3 seeds (L4) | 3 | 1.5 | $6 |
| 3. Three configurations x 3 seeds (L4) | 9 | 0.75 | $9 |
| **Total** | | | **about $35-40** |

Spent before these runs: about $8 of the team's $750.
