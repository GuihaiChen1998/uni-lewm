import time
from dataclasses import dataclass

import torch
from torch import Tensor

from .model import WorldModel


@dataclass(frozen=True)
class PlannerConfig:
    horizon: int = 12
    candidates: int = 256
    elites: int = 32
    iterations: int = 4
    smoothing: float = 0.2
    uniform_mix: float = 0.05
    chunk_size: int = 128

    def __post_init__(self):
        if min(self.horizon, self.candidates, self.elites, self.iterations, self.chunk_size) < 1:
            raise ValueError("planner counts must be positive")
        if self.elites > self.candidates:
            raise ValueError("elites cannot exceed candidates")
        if not 0 <= self.smoothing < 1 or not 0 <= self.uniform_mix <= 1:
            raise ValueError("invalid smoothing or uniform_mix")


@dataclass
class Plan:
    action: int
    sequence: Tensor
    distribution: Tensor
    cost: float
    seconds: float
    stopped: bool


class CEMPlanner:
    """Categorical MPC over physical actions; observed goal distance controls STOP."""

    def __init__(
        self,
        model: WorldModel,
        config: PlannerConfig,
        stop_threshold: float | None = None,
        seed: int = 0,
    ):
        self.model, self.config = model, config
        if stop_threshold is not None and (
            not torch.isfinite(torch.tensor(stop_threshold)) or stop_threshold < 0
        ):
            raise ValueError("stop threshold must be finite and nonnegative")
        self.stop_threshold = stop_threshold
        self.generator = torch.Generator(device=next(model.parameters()).device).manual_seed(seed)

    @staticmethod
    def goal_distance(states: Tensor, goals: Tensor) -> Tensor:
        # Mean square makes units comparable across width-matched controls.
        return (states - goals).float().square().mean(-1)

    @torch.inference_mode()
    def plan(self, history: Tensor, past_actions: Tensor, goal: Tensor) -> Plan:
        if self.model.training:
            raise ValueError("call model.eval() before planning")
        if history.shape[0] != 1 or goal.shape != (1, self.model.config.dim):
            raise ValueError("planner handles one environment, batched candidate trajectories")
        if history.device.type == "cuda":
            torch.cuda.synchronize(history.device)
        start = time.perf_counter()
        c, k = self.config, self.model.config.num_actions
        current = float(self.goal_distance(history[:, -1], goal).item())
        probabilities = history.new_full((c.horizon, k), 1 / k)
        if self.stop_threshold is not None and current <= self.stop_threshold:
            distribution = history.new_zeros(k + 1)
            distribution[-1] = 1
            return Plan(
                k,
                torch.empty(0, dtype=torch.long, device=history.device),
                distribution,
                current,
                time.perf_counter() - start,
                True,
            )
        best_cost, best_sequence = float("inf"), None
        for _ in range(c.iterations):
            sequences = torch.multinomial(
                probabilities, c.candidates, replacement=True, generator=self.generator
            ).T.contiguous()
            # Retain the best sequence from previous iterations.
            if best_sequence is not None:
                sequences[0] = best_sequence
            costs = []
            for chunk in sequences.split(c.chunk_size):
                rollout = self.model.rollout(
                    history.expand(len(chunk), -1, -1), past_actions.expand(len(chunk), -1), chunk
                )
                costs.append(self.goal_distance(rollout[:, -1], goal))
            costs = torch.cat(costs)
            if not torch.isfinite(costs).all():
                raise FloatingPointError("non-finite planning costs")
            elite_ids = costs.topk(c.elites, largest=False).indices
            if float(costs[elite_ids[0]]) < best_cost:
                best_cost = float(costs[elite_ids[0]])
                best_sequence = sequences[elite_ids[0]].clone()
            frequency = torch.nn.functional.one_hot(sequences[elite_ids], k).float().mean(0)
            probabilities = c.smoothing * probabilities + (1 - c.smoothing) * frequency
            probabilities = (1 - c.uniform_mix) * probabilities + c.uniform_mix / k
        if history.device.type == "cuda":
            torch.cuda.synchronize(history.device)
        outgoing = history.new_zeros(k + 1)
        outgoing[:k] = probabilities[0]
        return Plan(
            int(best_sequence[0]),
            best_sequence,
            outgoing,
            best_cost,
            time.perf_counter() - start,
            False,
        )
