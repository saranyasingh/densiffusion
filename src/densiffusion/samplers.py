from typing import Literal

import numpy as np
import torch
from torch import nn

from densiffusion.conditioning import prepare_observations
from densiffusion.contracts import Array, Mask, has_regular_spacing, positive_int
from densiffusion.training import TrainingConfig


class DDPMSampler:
    """DDPM imputation with fixed posterior variance and no value clipping."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        *,
        prediction_type: Literal["noise", "clean"] = "noise",
    ):
        if prediction_type not in ("noise", "clean"):
            raise ValueError("prediction_type must be 'noise' or 'clean'")
        self.model = model
        self.config = config
        self.prediction_type = prediction_type

    @torch.no_grad()
    def sample(
        self,
        observed: Array,
        *,
        factor: int,
        n_samples: int,
        dt: float,
        rng: np.random.Generator,
        mask: Mask | None = None,
        times: Array | None = None,
    ) -> Array:
        """Sample at spacing dt / factor, batching draws by config.batch_size."""
        positive_int("n_samples", n_samples)
        observed, hidden, target_times = prepare_observations(
            observed, factor=factor, dt=dt, mask=mask, times=times
        )
        batch, length, channels = observed.shape
        if not hidden.any():
            return np.repeat(observed[:, None], n_samples, axis=1)
        if not has_regular_spacing(target_times, dt / factor):
            raise ValueError("DDPM requires regular times with spacing dt / factor")

        parameter = next(self.model.parameters(), None)
        if parameter is None or not parameter.is_floating_point():
            raise ValueError("Model must have floating-point parameters")
        dtype, device = parameter.dtype, parameter.device
        beta = torch.linspace(
            self.config.beta_start,
            self.config.beta_end,
            self.config.diffusion_steps,
            dtype=dtype,
            device=device,
        )
        alpha = 1 - beta
        alpha_bar = alpha.cumprod(0)
        if not ((alpha > 0) & (alpha_bar < 1)).all():
            raise ValueError("Diffusion schedule is not representable in model dtype")
        previous = torch.cat((alpha_bar.new_ones(1), alpha_bar[:-1]))
        variance = beta * (1 - previous) / (1 - alpha_bar)
        if self.prediction_type == "noise":
            state_weight = alpha.rsqrt()
            prediction_weight = -beta * state_weight / (1 - alpha_bar).sqrt()
        else:
            state_weight = alpha.sqrt() * (1 - previous) / (1 - alpha_bar)
            prediction_weight = beta * previous.sqrt() / (1 - alpha_bar)
            state_weight[0], prediction_weight[0] = 0, 1
        std = variance.sqrt()
        generator = torch.Generator().manual_seed(int(rng.integers(2**63)))

        def noise(shape):
            return torch.randn(
                shape, generator=generator, dtype=dtype, device="cpu"
            ).to(device)

        samples = np.empty((batch * n_samples, length, channels))
        modes = [(module, module.training) for module in self.model.modules()]
        try:
            self.model.eval()
            for start in range(0, len(samples), self.config.batch_size):
                stop = min(start + self.config.batch_size, len(samples))
                indices = np.arange(start, stop) // n_samples
                conditioning = torch.as_tensor(
                    observed[indices], dtype=dtype, device=device
                )
                if not torch.isfinite(conditioning).all():
                    raise ValueError("Observed values overflow the model dtype")
                batch_mask = torch.as_tensor(hidden[indices], device=device)
                values = torch.where(
                    batch_mask, noise(conditioning.shape), conditioning
                )
                for step in range(self.config.diffusion_steps - 1, -1, -1):
                    steps = torch.full(
                        (stop - start,), step, dtype=torch.long, device=device
                    )
                    prediction = self.model(
                        values,
                        steps,
                        observed=conditioning,
                        mask=batch_mask,
                        dt=dt / factor,
                    )
                    if (
                        not isinstance(prediction, torch.Tensor)
                        or prediction.shape != values.shape
                    ):
                        raise ValueError("Prediction must have the same shape as input")
                    mean = (
                        state_weight[step] * values
                        + prediction_weight[step] * prediction
                    )
                    if step > 0:
                        mean = mean + std[step] * noise(values.shape)
                    values = torch.where(batch_mask, mean, conditioning)
                    if not torch.isfinite(values).all():
                        raise FloatingPointError("Non-finite DDPM samples")
                samples[start:stop] = values.cpu().to(torch.float64).numpy()
        finally:
            for module, training in modes:
                module.training = training

        samples = samples.reshape(batch, n_samples, length, channels)
        return np.where(hidden[:, None], samples, observed[:, None])
