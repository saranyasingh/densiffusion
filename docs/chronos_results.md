# Chronos-2 experiments: results

Results for tasks 2 and 3 of `docs/chronos_experiment_plan.md`, which fixed
the comparisons, scoring and seeds (0, 1, 2) before the runs. Every completed
run is included. Lower scores are better.

## Task 2: Saranya's pipeline, one forecast per draw

20 customers as one 20-channel series, 12:00-23:00 hidden every day,
supervised CSDI, zero-shot `amazon/chronos-2`, forecasts starting at 12:00
(no history ends in a gap), 16 CSDI draws, `--stitch block`. Scaled CRPS
(CRPS over Saranya's seasonal scale), 156 forecasts per seed:

| Chronos-2 history | Mean | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|---|
| Gappy (missing hours masked) | 3.17 | 3.17 | 3.17 | 3.17 |
| Straight-line fill | 2.69 | 2.69 | 2.69 | 2.69 |
| CSDI, median of draws | 0.94 | 0.79 | 0.95 | 1.08 |
| CSDI, one forecast per draw, pooled | **0.88** | 0.75 | 0.92 | 0.98 |
| Complete history | 0.63 | 0.63 | 0.63 | 0.63 |
| Same hour yesterday | 1.06 | 1.06 | 1.06 | 1.06 |

Only the CSDI rows depend on the seed: the gaps are fixed and Chronos-2 is
deterministic.

- **T2-1, per-draw vs median:** per-draw is better in all three seeds, by 4%,
  4% and 9%.
- **T2-2 and T2-3, CSDI vs straight lines and gaps:** CSDI is far better than
  either, both of which are much worse than repeating yesterday.
- **T2-4, complete history:** CSDI recovers most, not all, of what the gaps
  cost (0.88 against 0.63, from 3.17).
- **Caveat:** CSDI varies a lot between seeds (gap-filling CRPS 0.17-0.33, vs
  0.96 for straight lines); in two seeds the median forecast is no better than
  repeating yesterday. CSDI trains on about 380 windows here.
- **Why CSDI wins so clearly here:** the same 12 hours are missing every day,
  so the gappy history never contains an afternoon or evening, the forecast's
  first 12 hours are exactly those hours, and supervised CSDI learned them
  from complete training data.

## Task 3: per customer, foundation-model style

8-day chunks from single customers (all 321), 1000 test chunks per group,
fixed across methods and seeds, 3 seeds, univariate CSDI trained for 100
epochs on 40,000 histories, `autogluon/chronos-2-small`. Scaled CRPS: CRPS
over the chunk's mean absolute observed history value. Differences are
positive when the first method is better; 95% intervals resample customers.
Full tables, including training customers and results by missing fraction:
`outputs/modal/chronos/chunks_e100/comparison.md` (Modal volume
`nihar/chronos/chunks_e100`).

### Held-out customers (never seen in training)

| Config | Missing | Gappy | Straight lines | CSDI median | CSDI per-draw | Complete | Same hour yesterday |
|---|---|---|---|---|---|---|---|
| Random gaps, self-supervised CSDI | 31% | 0.073 | 0.124 | 0.073 | 0.071 | 0.062 | 0.112 |
| Random gaps, supervised CSDI | 31% | 0.073 | 0.124 | 0.067 | 0.065 | 0.062 | 0.112 |
| Daily 12:00-23:00 gaps, supervised CSDI | 50% | 0.248 | 0.194 | 0.101 | 0.093 | 0.072 | 0.168 |

| | Comparison (held-out customers) | Random, self-sup | Random, supervised | Daily, supervised |
|---|---|---|---|---|
| C1 | per-draw CSDI vs straight lines | +0.053 [+0.045, +0.062] | +0.059 [+0.050, +0.068] | +0.102 [+0.078, +0.129] |
| C2 | per-draw vs median | +0.0013 [+0.0009, +0.0018] | +0.0019 [+0.0012, +0.0028] | +0.0081 [+0.0060, +0.0105] |
| C4a | straight lines vs gappy | -0.051 [-0.060, -0.043] | -0.051 [-0.060, -0.043] | +0.054 [+0.033, +0.076] |
| C4b | per-draw CSDI vs gappy | +0.0019 [+0.0005, +0.0035] | +0.0078 [+0.0057, +0.0103] | +0.155 [+0.112, +0.205] |
| C3 | self-supervised vs supervised (per-draw), random gaps | -0.0059 [-0.0078, -0.0043] | | |

Training customers give the same picture (C3: -0.0060 [-0.0076, -0.0045]).

### Findings

1. **C1 (primary): CSDI beats straight lines, but straight lines are the wrong
   bar.** With random gaps, straight-line filling is much worse than giving
   Chronos-2 the gaps as missing values (C4a: 0.124 against 0.073); Chronos-2
   handles missing values well on its own. The informative comparison is
   therefore C4b, against the gappy history.
2. **A filler that never sees complete data adds little.** Self-supervised
   CSDI beats the gappy history by 0.002 (about 3%), recovering about a fifth
   of what the gaps cost (0.073 against 0.062 with the complete history).
3. **Complete training data is what helps.** Supervised CSDI beats the gappy
   history by 0.008 (about 11%), recovering about three-quarters of the cost,
   and beats self-supervised CSDI (C3).
4. **The gain grows with the missing fraction.** With 50% or more missing
   (held-out customers): gappy 0.090, self-supervised 0.084, supervised 0.072,
   complete 0.062. Below 15% missing, nothing differs.
5. **Per-draw beats the median everywhere, but by little** (about 2-3% of the
   score with random gaps, about 8% with daily gaps).
6. **Fixed daily gaps are a special case.** When the same hours are always
   missing, the gappy history never shows them and filling helps hugely
   (0.093 against 0.248), as in task 2.
7. **Generalisation:** held-out and training customers give the same results.

This mirrors the synthetic results (`docs/quasiperiodic_writeup.md` on
`nihar/exp3-unified`): filling in from sparse data alone adds little beyond
what a capable forecaster already gets from the sparse history; what helps is
knowledge brought from complete data.

### Follow-up: heavy gaps

`--gaps heavy` hides a random 50-80% of each history (68% on average) in
gaps of 6-48 hours; same test chunks, CSDI size and training, 3 seeds.
Held-out customers:

| CSDI training | Gappy | Straight lines | CSDI median | CSDI per-draw | Complete | Same hour yesterday |
|---|---|---|---|---|---|---|
| Self-supervised | 0.096 | 0.256 | 0.157 | 0.140 | 0.059 | 0.122 |
| Supervised | 0.096 | 0.256 | 0.080 | 0.072 | 0.059 | 0.122 |

| | Comparison (held-out) | Self-supervised | Supervised |
|---|---|---|---|
| C1 | per-draw CSDI vs straight lines | +0.117 [+0.102, +0.131] | +0.184 [+0.160, +0.210] |
| C2 | per-draw vs median | +0.0178 [+0.0154, +0.0206] | +0.0073 [+0.0060, +0.0092] |
| C4a | straight lines vs gappy | -0.160 [-0.183, -0.138] | -0.160 [-0.183, -0.138] |
| C4b | per-draw CSDI vs gappy | **-0.043 [-0.054, -0.034]** | **+0.024 [+0.021, +0.028]** |
| C3 | self-supervised vs supervised | -0.067 [-0.080, -0.057] | |

- **Supervised CSDI gains most here:** 25% better than the gappy history,
  recovering about two-thirds of what the gaps cost (0.096 against 0.059).
- **Self-supervised CSDI makes forecasts 45% worse than leaving the gaps.**
  Its gap-filling error is 0.18-0.20 (supervised 0.071, straight lines 0.33).
  It learned by filling runs of 1-2 hours, and here must fill up to 48. The
  result holds in every seed and whether or not the history ends in a gap
  (71% do). A self-supervised variant that masks long runs, matching the test
  gaps, is the natural next check.
- **Per-draw beats the median by 9-11%**, as with daily gaps: the more
  uncertain the fill, the more pooling helps.
- Training customers give the same picture.

### Caveats

- One dataset, small models (CSDI, Chronos-2-small), 3 seeds.
- With random gaps, about 31% of histories end in a gap; there, filling the
  last gap is partly forecasting.
- Chronos-2's pretraining may include this dataset (not checked).
- The task 2 and task 3 daily-gap results differ in customers (20 largest vs
  all), modelling (joint vs per customer) and Chronos-2 size, so they are not
  directly comparable.
