from types import SimpleNamespace

import torch
from torch import nn

from fused_lewm.grid import TwoRoomGrid
from fused_lewm.planner import CEMPlanner, PlannerConfig


class KnownDynamics(nn.Module):
    """Exact 1D dynamics exercises CEM independently of learned model quality."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(num_actions=2, dim=1)

    def rollout(self, history, past_actions, future_actions):
        return history[:, -1:] + (2 * future_actions - 1).cumsum(1)[..., None]


def test_cem_reaches_known_goal_and_stops_only_from_observed_state():
    model = KnownDynamics().eval()
    planner = CEMPlanner(
        model,
        PlannerConfig(horizon=3, candidates=64, elites=8, iterations=3, chunk_size=32),
        1e-8,
        seed=4,
    )
    past = torch.empty(1, 0, dtype=torch.long)
    result = planner.plan(torch.zeros(1, 1, 1), past, torch.tensor([[3.0]]))
    assert result.action == 1 and result.cost == 0 and not result.stopped
    result = planner.plan(torch.tensor([[[3.0]]]), past, torch.tensor([[3.0]]))
    assert result.action == 2 and result.stopped
    assert result.sequence.numel() == 0


def test_environment_walls_and_doorway():
    env = TwoRoomGrid()
    assert env.transition((3, 2), 3) == (3, 2)
    assert env.transition((3, 4), 3) == (4, 4)
    assert env.transition((4, 4), 3) == (5, 4)
    assert env.shortest_distance((3, 2), (5, 2)) == 6
