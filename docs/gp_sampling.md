# GP sampling

Draws complete Gaussian-process (GP) trajectories for synthetic experiments.
Hiding points, splitting, and windowing belong to the shared data pipeline;
this module only generates full paths.

## Files

| File | Contents |
| --- | --- |
| `src/densiffusion/gp/kernels.py` | Kernel interface, built-in covariance families, `KERNELS` registry, `get_kernel` |
| `src/densiffusion/gp/sampling.py` | `sample_gp`, `covariance_factor`, `generate` plug-in |
| `src/densiffusion/gp/posterior.py` | `gp_posterior`: exact posterior mean and covariance |
| `configs/gp.example.toml` | Example generation config |
| `tests/test_gp.py` | Sampler checks |
| `tests/test_gp_posterior.py` | Posterior checks |
| `scripts/plot_gp_samples.py` | Demo plot |

## Usage

```python
import numpy as np
from densiffusion.gp.sampling import sample_gp

times = np.arange(0, 10, 0.5)  # any 1-D grid; irregular and unsorted are fine
paths = sample_gp(
    times, "exponential", rng=0, n_samples=4, variance=1.0, lengthscale=2.0
)
paths.shape  # (4, 20): one row per trajectory, columns in the order of times
```

As a dataset, through the existing CLI:

```sh
cp configs/gp.example.toml configs/local.toml
uv run densiffusion generate configs/local.toml --output data/gp.npz
```

This saves `values` shaped `(n_series, n_steps, 1)` at times
`arange(n_steps) * dt`, plus the full config (kernel name, parameters, mean,
seed), which is everything needed to rebuild the covariance exactly.

Demo plot (writes `outputs/gp_samples.png`; matplotlib is not a project
dependency, so `--with` installs it for this run only):

```sh
uv run --with matplotlib python scripts/plot_gp_samples.py
```

## Conventions

- **Kernel signature:** `kernel(t, s, **params)` takes two 1-D time arrays and
  returns the `len(t) x len(s)` matrix `K[i, j] = k(t[i], s[j])`. It takes times,
  not distances, so non-stationary kernels (for example fractional Brownian
  motion) fit the same interface.
- **Built-in kernels:**
  - `exponential`: `variance * exp(-|t - s| / lengthscale)` (Matérn 1/2)
  - `brownian`: `variance * min(t, s)`
  - `fractional_brownian`: `0.5 * variance * (|t|^(2H) + |s|^(2H) - |t-s|^(2H))`
  - `matern_3_2`: `variance * (1 + sqrt(3) r / ell) * exp(-sqrt(3) r / ell)`
  - `matern_5_2`: `variance * (1 + sqrt(5) r / ell + 5 r^2 / (3 ell^2)) * exp(-sqrt(5) r / ell)`
  - `periodic`: periodic covariance with a period parameter
- **Mean:** a scalar or one value per time; default 0.
- **Noise:** none. Draws are noiseless latent values.
- **No jitter:** samples use exactly `K`. `covariance_factor` tries Cholesky;
  if `K` is singular but valid (repeated timestamps, low-rank kernels, very
  smooth kernels on dense grids) it uses an eigendecomposition and zeroes
  eigenvalues within roundoff (`n * eps * largest eigenvalue`, the
  `numpy.linalg.matrix_rank` threshold). Asymmetric or materially
  non-positive-semidefinite matrices raise `ValueError`.
- **Randomness:** `rng` is a `numpy.random.Generator` or an integer seed. Global
  random state is never touched. The same seed and inputs reproduce the same
  draws.
- **Parameter typos fail:** an unknown keyword such as `length_scale` raises
  `TypeError` instead of being ignored.

## Checks

`uv run pytest tests/test_gp.py` covers:

1. The same seed reproduces draws; a different seed changes them.
2. Over 200,000 draws on an irregular, unsorted grid, the empirical mean and
   covariance (including off-diagonal entries) match the requested values.
3. `A @ A.T` reconstructs `K` for positive-definite, repeated-time, rank-1, and
   rank-2 covariances; a repeated timestamp gives identical values.
4. A smooth kernel on a 1000-point grid (where Cholesky fails) still samples.
5. Bad inputs fail clearly: invalid parameters, non-finite or non-1-D times,
   unknown kernel, wrong-length mean, asymmetric or non-PSD covariance.
6. The `generate` plug-in matches `sample_gp`, and the example config runs.

## Posterior

`gp_posterior(obs_times, obs_values, query_times, kernel, *, mean=0.0,
**kernel_params)` conditions the same GP on noiseless observed values and
returns the posterior `(mean, covariance)` at `query_times`. `obs_values` is
`(n_obs,)` or `(batch, n_obs)`; the covariance depends only on the times, so all
rows share it. Pass the saved dataset config (kernel, parameters, mean) to
describe hidden points of generated data exactly. A singular `K` is handled with a pseudo-inverse (no jitter); noiseless observations
`K` cannot produce, such as different values at a repeated time, raise
`ValueError`.

## Supported process families

The GP layer now supports a small set of common covariance families directly:

- short-range rough process: `exponential` / Matérn 1/2
- diffusion-like process: `brownian`
- rough long-memory process: `fractional_brownian` with Hurst exponent `hurst`
- smoother latent processes: `matern_3_2`, `matern_5_2`
- periodic structure: `periodic`

All of these are selected the same way as the original exponential kernel: pass
`kernel` as a name in `KERNELS` or as a custom callable with the same contract.
No sampler changes are needed beyond defining a valid covariance matrix.
