# Resource allocation audit

Equal forecaster parameter counts or optimizer updates do not imply equal
pipeline resources. Densification changes sequence length and adds densifier
training, cached augmentation, and history sampling. History polishing in the
latest experiment keeps sequence length unchanged and has a much smaller setup
cost.

The completed [local audit](../outputs/compute-allocation-20261027/REPORT.md)
reconstructs cumulative training/selection costs and best available validation
checkpoints under common resource ceilings. It also benchmarks native CPU/MPS
kernels on synthetic tensors. All earlier scientific artifacts are preserved.

```sh
uv run --locked python scripts/audit_compute_allocation.py \
  --output outputs/compute-allocation-reproduction
uv run --locked pytest tests/test_resource_allocation.py
```

Use a fresh output directory. No scientific training, test generation, or test
ensemble loading occurs. The audit recomputes all four forecast metrics from
the 63 saved polishing validation ensembles. The older waveform runner saved
validation score logs, but not corresponding checkpoint ensembles; those scores
are checked against training logs and cannot be independently rescored here.

The reproducible work unit is **dense matrix FLOPs**, with a multiply and an add
counted as two. Forward and backward are profiled independently. Math attention
is used only for counting, because fused CPU inference hides attention products
from the counter. The counts agree with an independent formula for every
profiled sequence length and scale linearly with batch size. This is a defined
compute proxy; activation, normalization, optimizer, median comparisons, data
movement, I/O, and other nonmatrix work are excluded. Native-kernel timings use
the normal implementation and include optimizer/clipping for training steps.

For each historical checkpoint the audit charges complete preceding setup,
optimization, denoising validation, and forecast validation. Mediator setup
includes all four candidates, not just the winner. Failed validation candidates
still consume resources. Deployment work for a 64-draw forecast is separate.
Reported elapsed times include nonmatrix overhead; their hardware and
concurrency limitations are recorded in the report.

The recorded result distinguishes two questions:

- Small waveform densification used 2.32 times the direct model's counted
  training/selection work for linear interpolation and 2.67 times for learned
  densification. At the direct model's full resource ceiling, direct forecasting
  has the best saved validation result. The larger waveform model's transfer
  gate had already failed, so it supplies no larger-model forecast comparison.
- Learned polishing added 0.63% of forecaster optimization work, or 0.071% of
  total counted training/selection work. Its epoch-200 models beat the best raw
  validation checkpoints through epoch 1,200 in all three seeds. Its epoch-400
  models also beat the best median checkpoints through epoch 1,200 in all seeds.

These are retrospective validation comparisons. Unspent budget, sparse
checkpoints, original learning-rate schedules, and selection optimism prevent
an optimal-allocation or fresh resource-matched test claim. The report specifies
the separate training, selection, and deployment constraints required for such
a trial. No convergence or universal advantage is established.
