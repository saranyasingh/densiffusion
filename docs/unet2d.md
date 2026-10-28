# 2-D U-Net denoiser for image imputation

`UNet2DDenoiser` in `src/densiffusion/architectures/unet2d.py` is a small
convolutional U-Net that plugs into the shared `ImputationModel`,
`train_densifier` and `DDPMSampler` exactly like `CSDIDenoiser`, with the same
`forward(noisy, steps, *, observed, mask, dt)` interface and `(batch, time,
channels)` tensors. It exists because the image scripts feed CSDI an image as
rows-as-time and columns-as-channels, so CSDI attends down columns and along
rows with no notion of 2-D locality. The U-Net instead reshapes
`(batch, rows, columns * planes)` to `(batch, planes, rows, columns)` and
convolves over the pixel grid. `planes` is the number of adjacent channels per
pixel: one for a grey image, three for RGB in the cat layout, two in residual
mode below.

Conditioning follows CSDI: the observed plane with hidden entries cleared, the
noisy plane with observed entries cleared and the observed mask are the input
channels, and every residual block adds a learned sinusoidal embedding of the
diffusion step. `dt` is validated but unused, since rows are pixels. Odd sizes
such as 33 and 65 need no padding: each level downsamples with a stride-two
convolution and upsamples by nearest-neighbour interpolation back to the skip
connection's size. The final convolution starts at zero, as in CSDI.

Defaults are 32 hidden channels with level multipliers `(1, 2, 4)`, one block
per level, two at the bottom, GroupNorm with eight groups and no dropout, about
1.9M parameters. Every level width must be divisible by the group count.

## Residual mode

`scripts/impute_images_csdi.py --residual` diffuses only the hidden residual
around linear interpolation, in the spirit of ResShift and CorrDiff. The
interpolant `b` is computed from the observed pixels, and each pixel becomes
two adjacent channels, `(x - b, b)`, with the mask hiding only the first. The
residual is exactly zero at observed pixels, `b` is a fully observed plane the
network can read, and the sampler's observation clamping keeps both fixed, so
observed pixels are consistent by construction. Samples are mapped back with
`residual + baseline`. The layout is architecture-agnostic: CSDI can run in
residual mode too, with `n_channels = 2 * size`.

## Running the synthetic comparison

```sh
uv run --extra research --with matplotlib python scripts/impute_images_csdi.py \
    --output outputs/unet_images --denoiser unet --residual
```

`--denoiser csdi` (the default) reproduces the original CSDI run; combine
either denoiser with `--residual`. `--channel-multipliers` and
`--blocks-per-level` size the U-Net; `--hidden-channels` is shared with CSDI.

`results.json` now reports three oracle numbers for the exact GP posterior: the
shared 20-draw ensemble `crps` used in the table, `crps_fair` from the unbiased
n(n-1) spread estimator (`metrics.crps(..., fair=True)`), and `crps_exact`,
the closed-form CRPS of the Gaussian posterior marginals. Diffusion models
report `crps` and `crps_fair`; comparisons should use the same estimator and
draw count on both sides.

## Results on the synthetic task

Script defaults (Matérn 3/2, lengthscale 5 px, 33×33, every other row and
column observed, 2000 training images, 100 epochs on CPU), scored on the same
100 test images with 20 draws. Spread is the mean per-pixel sample standard
deviation at hidden pixels over the stored examples.

| Imputer | CRPS | vs oracle | Fair CRPS | RMSE | Spread | Params | Train |
|---|---|---|---|---|---|---|---|
| Exact GP posterior | 0.0740 | — | 0.0704 (exact 0.0706) | 0.1286 | 0.124 | | |
| U-Net | 0.0741 | +0.2% | 0.0711 | 0.1282 | 0.106 | 1.93M | 60 min |
| U-Net residual | 0.0741 | +0.2% | 0.0710 | 0.1282 | 0.107 | 1.93M | 56 min |
| CSDI | 0.0776 | +4.9% | 0.0742 | 0.1347 | 0.121 | 157k | 104 min |
| Linear interpolation | 0.1289 | +74% | | 0.1651 | | | |

The two models fail in complementary ways. CSDI's spread matches the oracle
but its posterior mean is worse than the GP mean; the U-Net's mean matches the
GP mean but its samples are about 14% under-dispersed. Residual mode changes
nothing for the U-Net; for CSDI at a reduced budget (1000 images, 30 epochs)
it cut CRPS from 0.3455 to 0.0953, so it mainly speeds up convergence. A
532k-parameter U-Net (`--hidden-channels 16`) scored 0.0742 at the reduced
budget in four minutes. Outputs live in `outputs/csdi_vs_unet/`.
