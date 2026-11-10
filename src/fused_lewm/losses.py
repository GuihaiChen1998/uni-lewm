import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .model import WorldModel


class SIGReg(nn.Module):
    """Sketched Gaussian characteristic-function discrepancy (Epps-Pulley).

    Integrates over [-3,3] using symmetry and a standard-normal weight. For
    [B,T,D], statistics are over independent batch trajectories at each time;
    temporal frames are not falsely counted as B*T independent observations.
    This self-contained quadrature implementation is not upstream LeWM code.
    """

    def __init__(self, projections: int = 64, points: int = 17):
        super().__init__()
        if projections < 1 or points < 2:
            raise ValueError("need positive projections and at least two quadrature points")
        self.projections = projections
        self.register_buffer("grid", torch.linspace(0, 3, points))

    def forward(self, z: Tensor) -> Tensor:
        if z.ndim == 2:
            z = z[:, None]
        if z.ndim != 3 or z.shape[0] < 2:
            raise ValueError("SIGReg needs [B,T,D] or [B,D] with B>=2")
        # Evaluate characteristic functions in fp32 even under mixed precision.
        with torch.autocast(device_type=z.device.type, enabled=False):
            z = z.float()
            directions = F.normalize(
                torch.randn(z.shape[-1], self.projections, device=z.device), dim=0
            )
            projected = torch.einsum("btd,dr->tbr", z, directions)
            angles = projected[..., None] * self.grid
            real = angles.cos().mean(1)
            imag = angles.sin().mean(1)
            target = torch.exp(-0.5 * self.grid.square())
            discrepancy = (real - target).square() + imag.square()
            weight = target / (2 * torch.pi) ** 0.5
            return 2 * z.shape[0] * torch.trapezoid(discrepancy * weight, self.grid, dim=-1).mean()


def world_model_loss(
    model: WorldModel,
    observations: Tensor,
    actions: Tensor,
    regularizer: SIGReg,
    coefficient: float,
) -> tuple[Tensor, dict]:
    """observations [B,L+H,C,S,S], actions [B,L+H-1]. Both targets get gradients."""
    batch, length = observations.shape[:2]
    context = model.config.history
    if length <= context or actions.shape != (batch, length - 1):
        raise ValueError(
            "training needs aligned context+future observations and transition actions"
        )
    # Keep future frames out of context batch statistics when the optional
    # shared projector BatchNorm is enabled. Encoding all frames in one batch
    # would leak future images through normalization even without attention.
    context_z = model.encode(observations[:, :context].flatten(0, 1)).reshape(batch, context, -1)
    target_z = model.encode(observations[:, context:].flatten(0, 1)).reshape(
        batch, length - context, -1
    )
    encoded = torch.cat([context_z, target_z], dim=1)
    predicted = model.rollout(
        encoded[:, :context], actions[:, : context - 1], actions[:, context - 1 :]
    )
    per_horizon = (predicted - encoded[:, context:]).float().square().mean(dim=(0, 2))
    pred_loss = per_horizon.mean()
    reg_loss = regularizer(encoded)
    total = pred_loss + coefficient * reg_loss
    metrics = {
        "loss": total.detach(),
        "prediction": pred_loss.detach(),
        "sigreg": reg_loss.detach(),
        "latent_std": encoded.detach().float().std(dim=0).mean(),
        "last_horizon_mse": per_horizon[-1].detach(),
    }
    return total, metrics
