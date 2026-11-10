from dataclasses import asdict, dataclass
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    image_size: int = 32
    patch_size: int = 8
    channels: int = 3
    num_actions: int = 4
    dim: int = 192
    depth: int = 6
    heads: int = 3
    mlp_ratio: int = 4
    history: int = 4
    dropout: float = 0.0
    shared: bool = True
    projection_batchnorm: bool = True

    def __post_init__(self):
        for name in (
            "image_size",
            "patch_size",
            "channels",
            "num_actions",
            "dim",
            "depth",
            "heads",
            "mlp_ratio",
            "history",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.image_size % self.patch_size or self.dim % self.heads:
            raise ValueError("image_size must divide by patch_size; dim must divide by heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


class GoalEncoding(NamedTuple):
    state: Tensor
    outgoing: Tensor


class WorldModel(nn.Module):
    """One shared trunk/readout, with image and causal latent tokenizers.

    There is no goal input to the dynamics. STOP is protocol metadata, never a
    physical transition. The independent ablation changes only trunk sharing.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        c = config
        self.patch = nn.Conv2d(c.channels, c.dim, c.patch_size, stride=c.patch_size)
        self.state_token = nn.Parameter(torch.randn(1, 1, c.dim) * 0.02)
        self.null_action = nn.Parameter(torch.randn(1, 1, c.dim) * 0.02)
        self.action_embedding = nn.Embedding(c.num_actions + 1, c.dim)
        self.spatial_position = nn.Parameter(
            torch.randn(1, (c.image_size // c.patch_size) ** 2 + 2, c.dim) * 0.02
        )
        self.temporal_position = nn.Parameter(torch.randn(1, 2 * c.history + 1, c.dim) * 0.02)
        self.token_types = nn.Embedding(3, c.dim)  # latent, action, query
        self.trunk = self._new_trunk()
        # Do not register the same module twice: state_dict has a single shared trunk.
        self.independent_dynamics = None if c.shared else self._new_trunk()
        readout_layers = [nn.Linear(c.dim, 2 * c.dim)]
        if c.projection_batchnorm:
            readout_layers.append(nn.BatchNorm1d(2 * c.dim, affine=False))
        readout_layers.extend([nn.GELU(), nn.Linear(2 * c.dim, c.dim)])
        self.readout = nn.Sequential(*readout_layers)
        # Image and latent paths have different activation distributions. Share
        # every trainable readout weight, but not BatchNorm running statistics.
        # Mixing these buffers made training look healthy while eval collapsed.
        self.dynamics_norm = (
            nn.BatchNorm1d(2 * c.dim, affine=False) if c.projection_batchnorm else None
        )

    def _read(self, hidden: Tensor, prediction: bool = False) -> Tensor:
        if prediction and self.dynamics_norm is not None:
            return self.readout[3](self.readout[2](self.dynamics_norm(self.readout[0](hidden))))
        return self.readout(hidden)

    def _new_trunk(self):
        c = self.config
        # Construct layers separately; TransformerEncoder clones identical initial
        # parameters otherwise. Independently initialize each cloned layer below.
        layer = nn.TransformerEncoderLayer(
            c.dim,
            c.heads,
            c.dim * c.mlp_ratio,
            c.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        trunk = nn.TransformerEncoder(
            layer, c.depth, norm=nn.LayerNorm(c.dim), enable_nested_tensor=False
        )
        for block in trunk.layers:
            nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        return trunk

    @property
    def dynamics_trunk(self):
        return self.trunk if self.independent_dynamics is None else self.independent_dynamics

    @property
    def stop_index(self):
        return self.config.num_actions

    def condition(self, probabilities: Tensor, known: Tensor) -> Tensor:
        """Normalize missing conditions, preserving exact zero/uniform equivalence."""
        if probabilities.shape[-1] != self.config.num_actions + 1:
            raise ValueError("action condition must have k+1 coordinates")
        if known.shape != probabilities.shape[:-1] or known.dtype != torch.bool:
            raise ValueError("known must be a boolean mask with the action batch shape")
        if not torch.isfinite(probabilities).all():
            raise ValueError("action condition must be finite")
        valid = probabilities[known]
        if valid.numel() and (
            (valid < 0).any() or not torch.allclose(valid.sum(-1), torch.ones_like(valid[..., 0]))
        ):
            raise ValueError("known action conditions must be probability distributions")
        embedded = probabilities @ self.action_embedding.weight
        return torch.where(known[..., None], embedded, self.null_action[0, 0])

    def encode(
        self, observations: Tensor, incoming: Tensor | None = None, known: Tensor | None = None
    ) -> Tensor:
        """Canonical per-frame read. Actual actions belong in the history window."""
        c = self.config
        if observations.ndim != 4 or tuple(observations.shape[1:]) != (
            c.channels,
            c.image_size,
            c.image_size,
        ):
            raise ValueError("observations must be [B,C,image_size,image_size]")
        batch = len(observations)
        if incoming is None:
            incoming = observations.new_zeros(batch, c.num_actions + 1)
        if known is None:
            known = torch.zeros(batch, dtype=torch.bool, device=observations.device)
        if known.any():
            raise ValueError(
                "canonical observation encoding uses known=False; pass actions to predict"
            )
        action = self.condition(incoming, known)[:, None]
        patches = self.patch(observations).flatten(2).transpose(1, 2)
        tokens = torch.cat([self.state_token.expand(batch, -1, -1), patches, action], dim=1)
        encoded = self.trunk(tokens + self.spatial_position)
        return self._read(encoded[:, 0])

    def encode_goal(self, observations: Tensor) -> GoalEncoding:
        incoming = observations.new_full(
            (len(observations), self.config.num_actions + 1), 1 / (self.config.num_actions + 1)
        )
        state = self.encode(observations, incoming=incoming)
        outgoing = observations.new_zeros(incoming.shape)
        outgoing[:, self.stop_index] = 1
        return GoalEncoding(state, outgoing)

    def _action_ids(self, actions: Tensor) -> Tensor:
        if actions.dtype.is_floating_point:
            if actions.ndim < 2 or actions.shape[-1] != self.config.num_actions + 1:
                raise ValueError("floating actions must be k+1 one-hot vectors")
            ids = actions.argmax(-1)
            exact = F.one_hot(ids, self.config.num_actions + 1).to(actions.dtype)
            if not torch.allclose(actions, exact):
                raise ValueError(
                    "soft actions require particle_rollout; do not average action embeddings"
                )
            actions = ids
        if actions.dtype != torch.long:
            raise ValueError("actions must be int64 IDs or floating one-hot vectors")
        if ((actions < 0) | (actions >= self.config.num_actions)).any():
            raise ValueError(
                "physical actions must be in [0,k); STOP terminates, it is not dynamics"
            )
        return actions

    def predict(self, history: Tensor, actions: Tensor) -> Tensor:
        """actions[:,i] is the action leaving history[:,i], including the new action."""
        c = self.config
        actions = self._action_ids(actions)
        if history.ndim != 3 or history.shape[-1] != c.dim:
            raise ValueError("history must be [B,L,dim]")
        batch, length, _ = history.shape
        if not 1 <= length <= c.history or tuple(actions.shape) != (batch, length):
            raise ValueError("need 1..history state/action pairs with matching batch sizes")
        states = history + self.token_types.weight[0]
        acts = self.action_embedding(actions) + self.token_types.weight[1]
        pairs = torch.stack([states, acts], dim=2).flatten(1, 2)
        query = self.state_token.expand(batch, -1, -1) + self.token_types.weight[2]
        tokens = torch.cat([pairs, query], dim=1)
        tokens = tokens + self.temporal_position[:, : tokens.shape[1]]
        mask = torch.ones(
            tokens.shape[1], tokens.shape[1], dtype=torch.bool, device=tokens.device
        ).triu(1)
        hidden = self.dynamics_trunk(tokens, mask=mask)
        return self._read(hidden[:, -1], prediction=True)

    def rollout(self, history: Tensor, past_actions: Tensor, future_actions: Tensor) -> Tensor:
        """Open-loop rollout. No future observations enter this computation."""
        future_actions = self._action_ids(future_actions)
        past_actions = self._action_ids(past_actions)
        if past_actions.shape != (len(history), history.shape[1] - 1):
            raise ValueError("past_actions must be [B,L-1]")
        if future_actions.ndim != 2 or future_actions.shape[0] != len(history):
            raise ValueError("future_actions must be [B,H]")
        if future_actions.shape[1] < 1:
            raise ValueError("rollout horizon must be positive")
        outputs = []
        for action in future_actions.unbind(1):
            aligned = torch.cat([past_actions, action[:, None]], dim=1)
            next_state = self.predict(history, aligned)
            outputs.append(next_state)
            history = torch.cat([history, next_state[:, None]], dim=1)[:, -self.config.history :]
            past_actions = (
                aligned[:, -(history.shape[1] - 1) :] if history.shape[1] > 1 else aligned[:, :0]
            )
        return torch.stack(outputs, dim=1)

    def particle_rollout(
        self,
        history: Tensor,
        past_actions: Tensor,
        probabilities: Tensor,
        particles: int,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Represent a soft action distribution by sampled complete trajectories."""
        if particles < 1 or probabilities.ndim != 3 or probabilities.shape[0] != len(history):
            raise ValueError("need particles>0 and probabilities [B,H,k]")
        if probabilities.shape[-1] != self.config.num_actions:
            raise ValueError("particle distributions are over the k physical actions")
        if (
            not torch.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or not torch.allclose(probabilities.sum(-1), torch.ones_like(probabilities[..., 0]))
        ):
            raise ValueError("invalid action probabilities")
        batch, horizon, count = probabilities.shape
        samples = torch.multinomial(
            probabilities.reshape(-1, count), particles, replacement=True, generator=generator
        )
        samples = samples.reshape(batch, horizon, particles).transpose(1, 2).contiguous()
        predictions = self.rollout(
            history.repeat_interleave(particles, 0),
            past_actions.repeat_interleave(particles, 0),
            samples.flatten(0, 1),
        )
        return predictions.reshape(batch, particles, horizon, -1), samples

    def forward(
        self,
        *,
        observations: Tensor | None = None,
        history: Tensor | None = None,
        actions: Tensor | None = None,
    ) -> Tensor:
        if observations is not None and history is None and actions is None:
            return self.encode(observations)
        if observations is None and history is not None and actions is not None:
            return self.predict(history, actions)
        raise ValueError("supply either observations or (history, actions)")

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def config_dict(self) -> dict:
        return asdict(self.config)


def matched_independent_config(reference: ModelConfig) -> ModelConfig:
    """Nearest parameter-count control by width search, preserving depth and heads."""
    with torch.random.fork_rng(devices=[]):
        target = WorldModel(reference).parameter_count()
    choices = []
    for dim in range(reference.heads * 2, reference.dim + 1, reference.heads):
        candidate = ModelConfig(**{**asdict(reference), "shared": False, "dim": dim})
        # Isolate temporary initialization from the caller's training RNG state.
        with torch.random.fork_rng(devices=[]):
            count = WorldModel(candidate).parameter_count()
        choices.append((abs(count - target), candidate))
    return min(choices, key=lambda pair: pair[0])[1]
