# Quasi-periodic sweep across models

Section 3 of [pristi_experiments.md](pristi_experiments.md) found that
densifying helps forecast a barely resolved wave, but only with fine training
data. This experiment repeats that sweep with every denoiser in the
repository, to check that the findings are not specific to PriSTI, and adds
controls for where the fine-data gain comes from.

## Setup

Everything except the network is identical to section 3: the quasi-periodic
kernel, P/dt in {1.6, 2.5, 3, 4, 6, 8, 16}, 256 / 32 / 64 paths, 9 history
and 4 future coarse points, the training schedule and epochs, 100 draws per
test window (10 densified histories x 10 futures) and the fair ensemble CRPS.
Each network is built by `densify_forecast.small_model` at about 45k
parameters:

| Model | Configuration | Parameters |
|---|---|---|
| PriSTI | width 32, 4 heads, 2 layers | 44,201 |
| CSDI | 32 channels, 4 heads, 2 residual layers | 45,097 |
| SSSD | width 24, 2 S4 blocks, state size 32 | 45,601 |
| TimeAutoDiff | width 24, 2 recurrent layers, embedding 32 | 47,561 |

### Matched-factor baseline and controls

The fine-data forecaster differs from the sparse one in two ways: it trains
on true fine paths, and its test histories are densified. Three methods
separate these:

- **Interpolated x2:** trained and tested on linearly interpolated coarse
  windows, the matched-factor baseline for fine-data x2. Section 3 only had
  interpolated x8.
- **Fine-data x2, interpolated history:** the fine-data x2 forecaster, but
  with test histories filled in by linear interpolation instead of the
  fine-data imputers. Comparing it with fine-data x2 shows what the learned
  densifier contributes.
- **Sparse, all phase shifts:** a sparse forecaster trained on the fine paths
  seen at the coarse spacing from all 8 offsets, so it has the same
  information as the fine data but never densifies. It has 8 times as many
  windows and trains for the same number of updates as the sparse forecaster
  (18 epochs, validating every epoch). Comparing it with fine-data x2 shows
  whether densification is needed at all, or the fine data alone suffice.

These were added after the first sweep. `run --extend` adds them to a finished
run without retraining or resampling anything else. Each new method draws its
own sampling seed, so `--extend` and a fresh run give identical samples.

## Commands

```sh
for model in csdi timeautodiff; do
    for task in qp_1.6 qp_2.5 qp_3 qp_4 qp_6 qp_8 qp_16; do
        uv run --extra research python scripts/run_densiffusion.py run \
            --model $model --task $task \
            --output outputs/quasiperiodic/$model --threads 1 &
    done
done
for task in qp_1.6 qp_2.5 qp_3 qp_4 qp_6 qp_8 qp_16; do
    uv run --extra research python scripts/run_densiffusion.py run \
        --model sssd --device mps --task $task \
        --output outputs/quasiperiodic/sssd --threads 1 &
done; wait
uv run --extra research --with matplotlib python \
    scripts/run_densiffusion.py compare --output outputs/quasiperiodic
```

PriSTI's results come from `outputs/densiffusion_quasiperiodic` (linked as
`outputs/quasiperiodic/pristi`), extended with the new methods. With
`--model pristi` the pipeline reproduces those samples exactly. SSSD runs on
the Apple GPU (`--device mps`), where it is about 7 times faster than on the
CPU. `compare` writes `comparison.md`, with the finding checks, the controls
and per-method tables across models, and `comparison.png`, with each model's
sweep.

## Results

Pending: the sweep is running.
