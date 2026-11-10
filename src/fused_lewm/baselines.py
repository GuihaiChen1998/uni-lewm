"""Toy-scale adapters, NOT full-paper reproductions. See docs/BENCHMARKS.md."""

import math
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from transformers import ViTConfig, ViTModel

from ._vendor.flexi_action import ARPrimitiveActor, VarKActionEncoder
from ._vendor.lewm import MLP, ARPredictor, Embedder, SIGReg
from ._vendor.subjepa import MultiSubspaceSIGReg


class ToyBaseline(nn.Module):
    def __init__(self, method, dim=48, depth=2, history=2, max_jump=4):
        super().__init__()
        if method not in {"lewm", "sub_lewm", "vlwm", "flexiworld"}:
            raise ValueError(method)
        self.method = method
        self.config = SimpleNamespace(dim=dim, history=history, image_size=32, num_actions=4)
        self.max_jump = max_jump
        self.stop_index = 4
        self.encoder = ViTModel(
            ViTConfig(
                image_size=32,
                patch_size=8,
                hidden_size=dim,
                num_hidden_layers=depth,
                num_attention_heads=3,
                intermediate_size=dim * 2,
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
            ),
            add_pooling_layer=False,
        )
        self.projector = MLP(dim, dim * 2, norm_fn=nn.BatchNorm1d)
        self.pred_proj = MLP(dim, dim * 2, norm_fn=nn.BatchNorm1d)
        if method == "vlwm":
            self.action_encoder = nn.Embedding(4, dim)
            self.types = nn.Embedding(2, dim)
            self.positions = nn.Parameter(torch.randn(1, history + max_jump, dim) * 0.02)
            layer = nn.TransformerEncoderLayer(
                dim,
                3,
                dim * 2,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.predictor = nn.TransformerEncoder(
                layer, depth, norm=nn.LayerNorm(dim), enable_nested_tensor=False
            )
        else:
            self.predictor = ARPredictor(
                num_frames=history,
                depth=depth,
                heads=3,
                dim_head=dim // 3,
                mlp_dim=dim * 2,
                input_dim=dim,
                hidden_dim=dim,
            )
            self.action_encoder = Embedder(4, 4, dim)
        if method == "flexiworld":
            self.action_encoder = VarKActionEncoder(
                action_dim=4,
                emb_dim=dim,
                d_tok=24,
                depth=2,
                heads=3,
                dim_head=8,
                mlp_dim=48,
                k_max=max_jump,
            )
            self.actor = ARPrimitiveActor(
                embed_dim=dim,
                action_emb_dim=dim,
                action_dim=4,
                d_model=dim,
                depth=2,
                heads=3,
                dim_head=dim // 3,
                mlp_dim=dim * 2,
                k_max=max_jump,
            )
        self.regularizer = (
            MultiSubspaceSIGReg(dim, num_subspaces=4, num_proj=32)
            if method == "sub_lewm"
            else SIGReg(num_proj=32)
        )

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def encode(self, images):
        return self.projector(self.encoder(images).last_hidden_state[:, 0])

    def encode_sequence(self, images):
        # Statistics are computed across independent episodes at each time.
        # Future frames never enter the BN statistics of an input frame.
        return torch.stack([self.encode(x) for x in images.unbind(1)], 1)

    def regularize(self, z):
        x = z if self.method == "sub_lewm" else z.transpose(0, 1)
        # Match the normalized Gaussian quadrature used by Uni-LeWM.
        return self.regularizer(x) / math.sqrt(2 * math.pi)

    def predict(self, history, actions):
        encoded_actions = self.action_encoder(F.one_hot(actions, 4).float())
        x = self.predictor(history, encoded_actions)[:, -1]
        return self.pred_proj(x)

    def jump(self, history, past_actions, future):
        """Predict from the last observed state after ALL provided future actions."""
        if not 1 <= future.shape[1] <= self.max_jump:
            raise ValueError("jump outside trained horizon")
        if self.method == "vlwm":
            length = history.shape[1]
            state = history + self.types.weight[0]
            actions = self.action_encoder(future) + self.types.weight[1]
            x = torch.cat([state, actions], 1)
            x = x + self.positions[:, : x.shape[1]]
            mask = torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool).triu(1)
            x = self.predictor(x, mask=mask, is_causal=True)
            return self.pred_proj(x[:, length:][:, -1])
        if self.method != "flexiworld":
            raise ValueError("jump only defined for variable-length models")
        previous = (
            self.action_encoder(F.one_hot(past_actions, 4).float())
            if past_actions.shape[1]
            else history[:, :0]
        )
        block = F.one_hot(future, 4).float().flatten(1)[:, None]
        block = self.action_encoder(block, k=future.shape[1])
        conditioning = torch.cat([previous, block], 1)
        return self.pred_proj(self.predictor(history, conditioning)[:, -1])

    def rollout(self, history, past_actions, future_actions):
        if self.method in {"vlwm", "flexiworld"}:
            # Each output is a direct jump from the same observed history.
            return torch.stack(
                [
                    self.jump(history, past_actions, future_actions[:, :k])
                    for k in range(1, future_actions.shape[1] + 1)
                ],
                1,
            )
        outputs = []
        for action in future_actions.unbind(1):
            actions = torch.cat([past_actions, action[:, None]], 1)
            z = self.predict(history, actions)
            outputs.append(z)
            history = torch.cat([history, z[:, None]], 1)[:, -self.config.history :]
            past_actions = (
                actions[:, -(history.shape[1] - 1) :] if history.shape[1] > 1 else actions[:, :0]
            )
        return torch.stack(outputs, 1)

    def loss(self, observations, actions, progress):
        z = self.encode_sequence(observations)
        b, length, dim = z.shape
        context = self.config.history
        extra = z.new_zeros(())
        if self.method in {"lewm", "sub_lewm"}:
            histories = torch.stack(
                [z[:, t - context + 1 : t + 1] for t in range(context - 1, length - 1)], 1
            )
            act = torch.stack(
                [actions[:, t - context + 1 : t + 1] for t in range(context - 1, length - 1)], 1
            )
            predicted = self.predict(histories.flatten(0, 1), act.flatten(0, 1))
            prediction = (
                (predicted.reshape(b, length - context, dim) - z[:, context:]).square().mean()
            )
        else:
            # VLWM's curriculum increases maximum jump; FlexiWorld samples mixed k.
            max_k = (
                min(self.max_jump, 1 + int(progress * self.max_jump))
                if self.method == "vlwm"
                else self.max_jump
            )
            k = int(torch.randint(1, max_k + 1, ()).item())
            future = actions[:, context - 1 : context - 1 + k]
            predicted = self.jump(z[:, :context], actions[:, : context - 1], future)
            prediction = (predicted - z[:, context - 1 + k]).square().mean()
            if self.method == "flexiworld":
                current = z[:, context - 1]
                previous = self.action_encoder(
                    F.one_hot(actions[:, context - 2 : context - 1], 4).float()
                )[:, 0]
                target = F.one_hot(future, 4).float()
                lengths = torch.full((b,), k, device=z.device, dtype=torch.long)
                local = self.actor.nll(
                    current,
                    z[:, context - 1 + k] - current,
                    previous,
                    target,
                    lengths,
                    student_p=0.5,
                )["loss"]
                goal = self.actor.nll(
                    current, z[:, -1].detach() - current, previous, target, lengths, student_p=0.5
                )["loss"]
                extra = 0.1 * local + 0.05 * goal
        reg = self.regularize(z)
        total = prediction + reg + extra
        return total, {
            "loss": total.detach(),
            "prediction": prediction.detach(),
            "sigreg": reg.detach(),
            "actor": extra.detach(),
            "latent_std": z.detach().std(dim=0).mean(),
        }

    @torch.no_grad()
    def direct_action(self, history, past_actions, goal):
        if past_actions.shape[1]:
            prev = F.one_hot(past_actions[:, -1:], 4).float()
        else:
            prev = history.new_zeros(history.shape[0], 1, 4)
        prev = self.action_encoder(prev)[:, 0]
        chunk = self.actor.decode(history[:, -1], goal - history[:, -1], prev, self.max_jump)
        # Upstream is Gaussian/continuous; discretization is an explicit adaptation.
        return int(chunk[0, 0].argmax())
