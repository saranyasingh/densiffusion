# Densify, then forecast: results on a partially resolved periodic signal

**Status:** sections 1-5 and 8 are final. Sections 6 (training-length curve)
and 7 (Jena with longer training) are waiting for their runs; their commands
are in section 9.

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
   goes from 1.80 to 1.31, CSDI from 2.70 to 1.78, TimeAutoDiff from 2.31 to
   1.18 (different seeds; section 6 will compare like with like).
2. **Most of the x2 grid's advantage was undertraining.** Straight-line x2's
   gain over sparse falls from about 0.5 of the GP's error to 0.06-0.12 for
   PriSTI, and to nothing for TimeAutoDiff, where x2 is now significantly
   worse at P/dt 2.5. For CSDI a sizeable gain remains, but a 2x wider sparse
   forecaster matches straight-line x2 at every P/dt: what remains is
   capacity.
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

With adequate training, densifying a coarse history gives little or nothing
over a well-trained sparse forecaster of sufficient capacity. The large gains
seen with the original training mostly made up for undertrained forecasters,
and the rest for capacity; learned filling never beats straight lines. The
one consistent exception, the perfect filler, shows that knowledge of the
process could help, but the fillers trained here do not deliver it to the
forecaster.

## 6. Training-length curve (pending)

Seeds 1-3, x2 methods, all forecasters at 150, 450 and 1350 epochs, so the
comparison above uses the same seeds at every length. Question: do sparse and
straight-line x2 converge to the same error as training grows (the x2 grid
only speeds up training), or stay apart (a real inductive bias)?

## 7. Real data: Jena solar radiation (update pending)

With the original training (PriSTI, one seed; `docs/jena.md` on the
`nihar/jena` branch), straight-line x2 beat sparse by 11-20% at P/dt 1.64,
2.57 and 3.6, fine-data x2 never beat sparse, and the true fine history helped
by 8-23%. Given section 4, that gain may also be undertraining; the rerun
trains every forecaster 3x longer, with 3 seeds.

## 8. Limitations

- Small models (about 45k parameters), one family of synthetic processes, and
  one real dataset with one model.
- SSSD is excluded from the longer training: its forecaster's training
  diverges (non-finite loss) at about epoch 100 whether the learning rate is
  1e-3 or 3e-4. With the original 150 epochs the learning-rate drop at epoch
  112 came just in time on the CPU.
- Seeds vary training and sampling, not the data.
- "Sparse, 2x width" has about 3-4x the parameters; it tests whether capacity
  closes the gap, not a matched comparison.

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
modal volume get --env Densiffusion densiffusion-outputs nihar/qp_len450 outputs/modal/
uv run --extra research --with matplotlib python scripts/run_densiffusion.py \
    compare --models pristi csdi timeautodiff --seeds 1 2 3 \
    --output outputs/modal/qp_len450
```

(Section 4's runs are stored as `nihar/qp_long`, the 450-epoch run.)
