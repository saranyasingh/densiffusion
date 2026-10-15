# PriSTI: known shortcomings and future improvements

Open issues in `src/densiffusion/architectures/pristi.py` (temporal-only PriSTI).

1. **Observation mask as a network input**
   - Both `forward` methods receive `mask`, but use it only in fixed operations:
     choosing points to interpolate from (encoder) and zeroing the noisy input
     at observed points (denoiser). As in the paper, no learned layer sees it.
   - After interpolation, observed and filled-in points look the same, so
     different observation patterns can give identical encoder inputs.
   - Irrelevant for our fixed alternate mask, but matters once masks vary.
   - Could pass the mask to the encoder beside each interpolated value, and/or
     to the denoiser as an input channel (as CSDI does), off by default as an
     ablation.
2. **Sharing time and diffusion-step encodings**
   - `_time_embedding` (U_tem) and `_StepEmbedding` (with the float16 fix) are
     private to `pristi.py`.
   - DiffMAE and TimeDiT-lite need the same pieces; separate copies can drift
     and add uncontrolled differences between architectures.
   - Step embeddings differ by paper (CSDI/PriSTI vs. DiT), so decide whether
     all models share one or keep their own.
   - Move both into `architectures/common.py` when a second architecture is
     written.
3. **Sampling resolution (`dt`) is ignored**
   - Positions are window indices, so identical arrays at different spacings
     give identical predictions.
   - The proposal trains at 2Δt → Δt and applies at Δt → Δt/2, where values
     have a different scale.
   - Multiplying positions by `dt` is not a fix: the encoding becomes nearly
     constant for small `dt`, and one training spacing cannot teach what `dt`
     means.
   - Better: normalise each window by its observed increment scale, or
     condition on `dt` and train at several resolutions.
4. **Noise schedule**
   - The shared trainer and sampler support only a linear schedule; the paper
     uses quadratic (1e-4 to 0.2, 50-100 steps).
   - Adding a schedule option to `TrainingConfig` and `DDPMSampler` changes
     shared code, so agree it with the team.
5. **Training masks are fixed per example**
   - The trainer uses one fixed mask per window; the paper redraws point,
     block, or hybrid masks every iteration.
   - Could add a per-batch mask generator to the trainer, or repeat windows with
     different masks.
6. **No modelling across channels**
   - The spatial components are removed, so channels share weights but never
     interact, and GroupNorm statistics are per channel.
   - Fine for univariate data, but cannot capture correlated multivariate
     series such as exchange rates.
   - Could restore attention across channels with queries and keys from the
     prior, using full or learned adjacency since financial data has no graph.
7. **Encoder recomputed at every sampling step**
   - The encoder's output depends only on the observations but is recomputed
     at each diffusion step.
   - Caching the context per sampling batch in the shared sampler would save
     about 20-30% of sampling time (the encoder's measured share per step).
8. **Training details differ from the paper**
   - No learning-rate decay (the paper decays at 75% and 90% of training).
   - Batch size 64 by default (paper: 16).
   - Validation loss uses random diffusion steps (paper: averages all steps).
   - Untuned: only a short Brownian-motion run so far (CRPS 0.047 vs. 0.040 for
     the exact GP posterior).
9. **Minor**
   - The last layer's unused residual output leaves about 4,000 parameters
     without gradient (as in the reference; no effect on outputs).
   - Prior-guided attention weights cannot adapt to the noisy sample (PriSTI's
     design).
