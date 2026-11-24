# Densify, then forecast: results on a partially resolved periodic signal

**Status:** complete. Synthetic runs: one seed with the original training
(section 3), and seeds 1-3 at 150, 450 and 1350 epochs (sections 4 and 6).
Jena: one seed with the original training, three with 3x training
(section 7).

## 1. Question

Given a coarsely sampled history, does it help a diffusion forecaster to first
fill in the history on a finer grid (densify) and then forecast? Densifying
adds no information about the future: the filled-in values are a function of
the coarse observations plus whatever the filler learned in training. So any
gain must come from the forecaster using the information better, or from
knowledge the filler brings from its training data.

The test case is a wave that is barely resolved by the coarse grid, where
densifying looked most promising.

## 2. Setup

**Data.** Synthetic Gaussian-process paths with a quasi-periodic kernel
(cosine of period P times a Matérn 3/2 envelope), so the exact GP forecast is
known and is the best possible forecast. The coarse spacing dt sets P/dt, the
number of samples per period: 1.6, 2.5, 3, 4, 6, 8, 16. Below P/dt = 2 the
wave is aliased. 256 training, 32 validation and 64 test paths; each forecast
uses 9 coarse history points and predicts the next 4 coarse points. Paths come
from fixed data seeds, so a run's seed changes network initialisation,
training order and sampling, not the data.

**Models.** Four diffusion denoisers at about 45,000 parameters each: PriSTI,
CSDI, TimeAutoDiff and SSSD (`densify_forecast.small_model`).

**Methods.** Every method forecasts the same 4 coarse future points from the
same 9 coarse history points, and is scored on those points only; a x2
forecaster's midpoint predictions are discarded.

| Method | Training data | Test history |
|---|---|---|
| Sparse | coarse | coarse |
| Sparse, 2x width | coarse | coarse; twice as wide, about 3-4x the parameters |
| Straight-line x2 | coarse, midpoints by straight lines | straight lines |
| Coarse-data x2 | coarse; a filler learned from coarse data only | learned fill |
| Fine-data x2 | true fine paths | learned fill (filler trained on fine paths) |
| Perfect filler x2 | true fine paths | midpoints drawn from the exact GP posterior |
| Sparse, all phase shifts | fine paths seen at the coarse spacing from all 8 offsets | coarse |
| Joint x2 | true fine paths; one network fills and forecasts | coarse |
| Fine-data x2, straight-line or blank history | fine-data x2's forecaster | straight lines / midpoints left to the forecaster |

**Scoring.** 100 draws per test window (10 filled-in histories x 10 futures),
fair ensemble CRPS, reported relative to the exact GP forecast (1.00 = best
possible). Significance: 95% paired bootstrap intervals over test paths; with
several seeds, the intervals resample seeds as well as paths.

## 3. Results with the original training (150 epochs, one seed)

Forecast error relative to the exact GP. `+` / `-`: significantly better /
worse than the same model's sparse forecaster.

| P/dt | | PriSTI | CSDI | TimeAutoDiff | SSSD |
|---|---|---|---|---|---|
| 1.6 | Sparse | 1.69 | 2.55 | 2.33 | 1.17 |
| | Straight-line x2 | 1.20 + | 1.53 + | 1.59 + | 1.18 |
| | Fine-data x2 | 1.35 + | 2.97 - | 1.45 + | 1.23 |
| 2.5 | Sparse | 1.80 | 2.70 | 2.31 | 1.18 |
| | Straight-line x2 | 1.29 + | 2.04 + | 1.91 + | 1.24 - |
| | Fine-data x2 | 1.21 + | 2.40 + | 1.40 + | 1.34 - |
| 3 | Sparse | 1.46 | 3.08 | 1.49 | 1.16 |
| | Straight-line x2 | 1.31 + | 1.65 + | 1.57 - | 1.24 - |
| | Fine-data x2 | 1.28 + | 1.79 + | 1.49 | 1.19 |
| 4 | Sparse | 1.33 | 3.39 | 1.25 | 1.18 |
| | Straight-line x2 | 1.27 + | 1.82 + | 1.41 - | 1.21 |
| | Fine-data x2 | 1.26 + | 2.29 + | 1.32 - | 1.23 |

Taken at face value: a x2 grid helps a lot in the aliased cases, even when
filled by straight lines; learned filling beats straight lines for
TimeAutoDiff only; and SSSD, whose sparse forecaster is already close to the
exact GP, gains nothing. Coarse-data filling fails at P/dt 3-4, and every x8
method is worse than sparse once the wave is resolved (P/dt >= 6). Full
tables: `docs/quasiperiodic_models.md`.

But every sparse forecaster except SSSD's is 1.3-3.4x the exact GP's error,
which suggested the forecasters were undertrained.

## 4. Results with 3x training (450 epochs, 3 seeds)

All forecasters trained 3x longer (the phase-shifted one 56 epochs, the same
number of updates), seeds 1-3, x2 methods only. SSSD is excluded (section 8).

| P/dt | | PriSTI | CSDI | TimeAutoDiff |
|---|---|---|---|---|
| 1.6 | Sparse | 1.22 | 1.63 | 1.15 |
| | Sparse, 2x width | 1.19 | 1.37 + | 1.13 |
| | Straight-line x2 | 1.15 + | 1.32 + | 1.20 |
| | Fine-data x2 | 1.21 | 1.77 | 1.31 - |
| | Perfect filler x2 | 1.17 | 1.48 | 1.17 |
| 2.5 | Sparse | 1.31 | 1.78 | 1.18 |
| | Sparse, 2x width | 1.21 + | 1.44 + | 1.11 + |
| | Straight-line x2 | 1.19 + | 1.38 + | 1.27 - |
| | Fine-data x2 | 1.19 + | 1.84 | 1.40 - |
| | Perfect filler x2 | 1.11 + | 1.19 + | 1.13 |
| 3 | Sparse | 1.27 | 1.86 | 1.24 |
| | Sparse, 2x width | 1.27 | 1.36 + | 1.24 |
| | Straight-line x2 | 1.19 + | 1.31 + | 1.23 |
| | Fine-data x2 | 1.17 + | 1.63 | 1.33 |
| | Perfect filler x2 | 1.12 + | 1.28 + | 1.13 + |
| 4 | Sparse | 1.23 | 1.40 | 1.21 |
| | Sparse, 2x width | 1.27 | 1.30 + | 1.20 |
| | Straight-line x2 | 1.20 | 1.36 | 1.18 |
| | Fine-data x2 | 1.21 | 1.45 | 1.19 |
| | Perfect filler x2 | 1.18 | 1.27 + | 1.15 + |

Straight-line x2's gain over sparse, in units of the exact GP's error, with
95% intervals (P/dt 1.6, 2.5, 3, 4):

| | PriSTI | CSDI | TimeAutoDiff |
|---|---|---|---|
| over sparse | +0.06, +0.12, +0.08, +0.03 (ns) | +0.31, +0.40, +0.54, +0.04 (ns) | -0.05 (ns), -0.09, +0.00 (ns), +0.03 (ns) |
| over sparse, 2x width | +0.04 (ns), +0.02 (ns), +0.08, +0.07 | +0.04 (ns), +0.06 (ns), +0.05 (ns), -0.06 | -0.06 (ns), -0.15, +0.01 (ns), +0.02 (ns) |

Findings:

1. **Training longer helps the sparse forecaster most.** At P/dt 2.5, PriSTI
   goes from 1.81 to 1.31, CSDI from 2.66 to 1.78, TimeAutoDiff from 2.17 to
   1.18 (seeds 1-3 at 150 and 450 epochs; section 6).
2. **Most of the x2 grid's advantage was undertraining.** Straight-line x2's
   gain over sparse falls from about 0.5 of the GP's error to 0.06-0.12 for
   PriSTI, and to nothing for TimeAutoDiff, where x2 is now significantly
   worse at P/dt 2.5. For CSDI a sizeable gain remains at 450 epochs, and a
   2x wider sparse forecaster matches straight-line x2 at every P/dt; with
   9x training CSDI's gain shrinks to PriSTI's size and the wider forecaster
   no longer matches it (section 6).
3. **Learned filling never beats straight lines.** Fine-data x2 is no better
   than straight-line x2 for any model, and significantly worse for CSDI and
   TimeAutoDiff, the model where it had clearly won with the original
   training.
4. **The perfect filler still helps a little** (for example CSDI at P/dt 2.5:
   1.19 against 1.44 for the wide sparse forecaster). Its fills carry no extra
   information about the path, but they carry knowledge of the process; the
   learned fillers do not pass that on.
5. **Controls.** A forecaster trained on fine data cannot use coarse
   histories directly, whether given straight lines or left to fill the
   midpoints itself (both much worse than sparse). The joint model, worse than
   the separate filler and forecaster with the original training, now matches
   or beats it for CSDI and TimeAutoDiff. Coarse-data filling still fails at
   P/dt 3-4. Full tables: `outputs/modal/qp_long/comparison.md`.

## 5. Conclusion

Most of the gain from forecasting on a densified grid was undertraining. With
the original training a x2 grid removed up to 70% of the sparse forecaster's
excess error; with 3x and 9x training the sparse forecaster catches up
almost entirely. A small advantage of straight-line x2 remains for PriSTI and
CSDI, about 0.05-0.10 of the exact GP's error (5-8%), which a wider sparse
forecaster does not close; TimeAutoDiff shows none. Learned filling never
beats straight lines, and is often worse. On real data (Jena) the gain
disappears completely with longer training. The one consistent exception is
the perfect filler, which still helps: knowledge of the process could help
the forecaster, but the fillers trained here do not deliver it.

## 6. Training-length curve

Seeds 1-3, x2 methods, all forecasters at 150, 450 and 1350 epochs. Error
relative to the exact GP at 150 / 450 / 1350 epochs:

| P/dt | Method | PriSTI | CSDI | TimeAutoDiff |
|---|---|---|---|---|
| 1.6 | Sparse | 1.47 / 1.22 / 1.20 | 2.69 / 1.63 / 1.28 | 2.10 / 1.15 / 1.20 |
| | Straight-line x2 | 1.22 / 1.15 / 1.13 | 1.89 / 1.32 / 1.20 | 1.75 / 1.20 / 1.22 |
| 2.5 | Sparse | 1.81 / 1.31 / 1.25 | 2.66 / 1.78 / 1.29 | 2.17 / 1.18 / 1.22 |
| | Straight-line x2 | 1.35 / 1.19 / 1.16 | 1.91 / 1.38 / 1.24 | 1.91 / 1.27 / 1.23 |
| 3 | Sparse | 1.37 / 1.27 / 1.27 | 3.12 / 1.86 / 1.34 | 1.34 / 1.24 / 1.23 |
| | Straight-line x2 | 1.32 / 1.19 / 1.17 | 1.85 / 1.31 / 1.24 | 1.59 / 1.23 / 1.29 |
| 4 | Sparse | 1.31 / 1.23 / 1.24 | 3.09 / 1.40 / 1.34 | 1.30 / 1.21 / 1.23 |
| | Straight-line x2 | 1.23 / 1.20 / 1.21 | 1.78 / 1.36 / 1.26 | 1.39 / 1.18 / 1.25 |

Straight-line x2's gain over sparse, in units of the exact GP's error, at P/dt
1.6, 2.5, 3 and 4 ((ns): not significant):

| Epochs | PriSTI | CSDI | TimeAutoDiff |
|---|---|---|---|
| 150 | +0.25, +0.46, +0.05 (ns), +0.07 | +0.80, +0.76, +1.27, +1.31 | +0.36, +0.26, -0.25, -0.10 |
| 450 | +0.06, +0.12, +0.08, +0.03 (ns) | +0.31, +0.40, +0.54, +0.04 (ns) | -0.05 (ns), -0.09, +0.00 (ns), +0.03 (ns) |
| 1350 | +0.07, +0.09, +0.10, +0.03 (ns) | +0.08, +0.05 (ns), +0.10, +0.08 | -0.02, -0.01, -0.06, -0.02 (all ns) |

- **The gain mostly disappears by 450 epochs** and does not grow back.
- **A small gain persists for PriSTI and CSDI.** PriSTI's is stable from 450
  to 1350 epochs; CSDI's shrinks to the same size by 1350. TimeAutoDiff's is
  zero from 450 on.
- **It is not capacity.** At 1350 epochs the 2x wider sparse forecaster is no
  better than the normal one (PriSTI 1.24-1.30, CSDI 1.26-1.32), and
  straight-line x2 beats it for PriSTI at every P/dt.
- **Learned filling stays behind.** At 1350 epochs fine-data x2 is about level
  with straight lines for PriSTI and worse for CSDI (1.41-1.96) and
  TimeAutoDiff (1.32-1.48), worse even than sparse.
- **The perfect filler keeps a lead** (1350 epochs, P/dt 2.5: PriSTI 1.10,
  CSDI 1.14, TimeAutoDiff 1.15).

## 7. Real data: Jena solar radiation

PriSTI on hourly shortwave radiation from the MPI-BGC Jena station (period
24 h). Original: one seed (42), 150 epochs (`docs/jena.md`). Rerun: seeds 1-3,
450 epochs, x2 methods. Forecast CRPS relative to the sparse forecaster;
brackets count seeds significantly better (+) or worse (-) than sparse.

| P/dt | Sparse CRPS (W/m^2) | Straight-line x2 | Fine-data x2 | Perfect filler (true history) x2 |
|---|---|---|---|---|
| 1.64 | 63.8 -> 49.2 | 0.80 + -> 1.00 | 1.05 -> 1.08 [2-] | 0.85 + -> 0.90 [3+] |
| 2.57 | 56.7 -> 44.7 | 0.85 + -> 1.03 [1-] | 0.97 -> 1.12 [2-] | 0.77 + -> 0.93 [2+] |
| 3 | 40.0 -> 37.5 | 0.97 -> 1.01 | 0.96 -> 0.98 | 0.92 + -> 0.99 |
| 3.6 | 52.3 -> 45.5 | 0.89 + -> 1.00 | 1.05 -> 1.12 [3-] | 0.88 + -> 0.92 [3+] |
| 6 | 44.2 -> 43.6 | 1.00 -> 0.97 | 1.05 -> 1.10 [1-] | 0.96 -> 0.92 [2+] |
| 9 | 42.4 -> 39.9 | 1.03 -> 1.00 | 1.05 -> 1.14 [1-] | 1.02 -> 1.03 |
| 18 | 61.4 -> 60.1 | 1.04 -> 0.99 | 1.03 -> 0.99 | 0.99 -> 1.00 |

- **Training longer improves the sparse forecaster by 13-23%** at P/dt
  1.64-3.6.
- **Straight-line x2's 11-20% gain is gone:** it now matches sparse (1.00-1.03)
  and beats it in no seed.
- **Learned filling hurts:** fine-data x2 is 8-14% worse than sparse at most
  ratios.
- **The true fine history still helps** by 7-10% at P/dt 1.64, 2.57 and 3.6,
  so on real data too the gap is the filler, not the information.

## 8. Limitations

- Small models (about 45k parameters), one family of synthetic processes, and
  one real dataset with one model.
- SSSD is excluded from the longer training: its forecaster's training
  diverges (non-finite loss) at about epoch 100 whether the learning rate is
  1e-3 or 3e-4. With the original 150 epochs the learning-rate drop at epoch
  112 came just in time on the CPU.
- Seeds vary training and sampling, not the data.
- "Sparse, 2x width" has about 3-4x the parameters; it tests whether capacity
  closes the gap, not a matched comparison. At 1350 epochs it is slightly
  worse than the normal sparse forecaster for PriSTI, possibly overfitting.
- The training-length curve has three points; the learning-rate schedule
  scales with the number of epochs.
- Jena: one model (PriSTI); the original run used seed 42 and the rerun seeds
  1-3.

## 9. Reproducing

```sh
# Original training, one model (section 3): see docs/quasiperiodic_models.md.
# Sections 4 and 6 on Modal (CPU), x2 methods only:
modal deploy --env Densiffusion scripts/modal_sweep.py
M="sparse learned_x2 interp_x2 fine_x2 oracle_x2 fineinterp_x2 sparse_shifted joint_x2 fineblank_x2 sparse_wide"
for epochs in 150 450 1350; do
    uv run --with modal==1.6.4 python scripts/modal_sweep.py launch \
        --models pristi csdi timeautodiff --seeds 1 2 3 --name qp_len$epochs \
        --extra="--forecaster-epochs $epochs --methods $M"
done
modal volume get --env Densiffusion densiffusion-outputs nihar/qp_len150 outputs/modal/
uv run --extra research --with matplotlib python scripts/run_densiffusion.py \
    compare --models pristi csdi timeautodiff --seeds 1 2 3 \
    --output outputs/modal/qp_len150

# Section 7 (on the nihar/jena branch, which adds --sweep jena):
uv run --with modal==1.6.4 python scripts/modal_sweep.py launch --sweep jena \
    --models pristi --seeds 1 2 3 --name jena_long --extra="--forecaster-epochs \
450 --methods sparse_shifted learned_x2 interp_x2 fine_x2 fineinterp_x2 oracle_x2"
```

The 450-epoch runs are stored as `nihar/qp_long` on the volume, the 150- and
1350-epoch runs as `nihar/qp_len150` and `nihar/qp_len1350`, and the Jena rerun
as `nihar/jena_long`. Seed-averaged comparisons: `comparison.md` in each
downloaded folder.
