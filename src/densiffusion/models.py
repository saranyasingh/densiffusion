from torch import Tensor, nn


class ImputationModel(nn.Module):
    """Compose a denoiser with an optional encoder of observed values."""

    def __init__(self, denoiser: nn.Module, *, encoder: nn.Module | None = None):
        super().__init__()
        self.encoder = encoder
        self.denoiser = denoiser

    def forward(
        self,
        noisy: Tensor,
        steps: Tensor,
        *,
        observed: Tensor,
        mask: Tensor,
        dt: float,
    ) -> Tensor:
        conditions = {"observed": observed, "mask": mask, "dt": dt}
        if self.encoder is not None:
            conditions["context"] = self.encoder(observed, mask=mask, dt=dt)
        return self.denoiser(noisy, steps, **conditions)
