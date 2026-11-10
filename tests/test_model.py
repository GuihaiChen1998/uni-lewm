from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from fused_lewm.model import WorldModel, matched_independent_config


def test_start_goal_equal_and_stop_not_part_of_state(model):
    model.eval()
    images = torch.rand(3, 3, 16, 16)
    start = model.encode(images, torch.zeros(3, 5))
    goal = model.encode_goal(images)
    torch.testing.assert_close(start, goal.state, rtol=0, atol=0)
    torch.testing.assert_close(goal.outgoing, F.one_hot(torch.full((3,), 4), 5).float())
    # The STOP embedding cannot affect either observation reading or goal reading.
    with torch.no_grad():
        model.action_embedding.weight[4].add_(1000)
    torch.testing.assert_close(start, model.encode_goal(images).state, rtol=0, atol=0)


def test_sharing_is_parameter_identity_not_copy(model):
    assert model.dynamics_trunk is model.trunk
    assert not any("independent_dynamics" in name for name in model.state_dict())
    independent = WorldModel(replace(model.config, shared=False))
    assert independent.dynamics_trunk is not independent.trunk
    assert independent.parameter_count() > model.parameter_count()


def test_dynamics_does_not_pollute_observation_normalization(model):
    model.train()
    before = model.readout[1].running_mean.clone()
    model.predict(torch.rand(4, 3, 24), torch.zeros(4, 3, dtype=torch.long))
    torch.testing.assert_close(before, model.readout[1].running_mean, rtol=0, atol=0)
    assert not torch.equal(before, model.dynamics_norm.running_mean)
    assert sum(p.numel() for p in model.dynamics_norm.parameters()) == 0


def test_both_paths_update_the_same_trunk(model):
    weight = model.trunk.layers[0].self_attn.in_proj_weight
    model.encode(torch.rand(4, 3, 16, 16)).square().mean().backward()
    assert weight.grad is not None and weight.grad.norm() > 0
    model.zero_grad(set_to_none=True)
    model.predict(
        torch.rand(4, 3, 24), torch.zeros(4, 3, dtype=torch.long)
    ).square().mean().backward()
    assert weight.grad is not None and weight.grad.norm() > 0


def test_one_hot_agrees_with_action_id(model):
    model.eval()
    history = torch.rand(2, 3, 24)
    actions = torch.tensor([[0, 1, 2], [3, 0, 2]])
    torch.testing.assert_close(
        model.predict(history, actions), model.predict(history, F.one_hot(actions, 5).float())
    )
    with pytest.raises(ValueError, match="particle_rollout"):
        model.predict(history, torch.full((2, 3, 5), 0.2))
    with pytest.raises(ValueError, match="STOP"):
        model.predict(history, torch.full((2, 3), 4))


def test_rollout_tracks_transition_actions_and_history_trim(model, monkeypatch):
    calls = []

    def transition(history, actions):
        calls.append((history.clone(), actions.clone()))
        return history[:, -1] + actions[:, -1, None]

    monkeypatch.setattr(model, "predict", transition)
    history = torch.zeros(1, 2, 24)
    output = model.rollout(history, torch.tensor([[1]]), torch.tensor([[2, 3, 0]]))
    torch.testing.assert_close(output[0, :, 0], torch.tensor([2.0, 5.0, 5.0]))
    assert [a.tolist() for _, a in calls] == [[[1, 2]], [[1, 2, 3]], [[2, 3, 0]]]
    assert [z.shape[1] for z, _ in calls] == [2, 3, 3]


def test_causal_prefix_does_not_change_with_future_token(model):
    model.eval()
    x = torch.rand(2, 7, 24)
    mask = torch.ones(7, 7, dtype=torch.bool).triu(1)
    with torch.no_grad():
        first = model.dynamics_trunk(x, mask=mask)
        changed = x.clone()
        changed[:, 4:] += torch.randn_like(changed[:, 4:]) * 10
        second = model.dynamics_trunk(changed, mask=mask)
    torch.testing.assert_close(first[:, :4], second[:, :4], rtol=1e-5, atol=1e-6)


def test_particles_preserve_multimodal_actions(model, monkeypatch):
    def transition(history, actions):
        return history[:, -1] + (2 * actions[:, -1, None] - 1)

    monkeypatch.setattr(model, "predict", transition)
    probabilities = torch.tensor([[[0.5, 0.5, 0.0, 0.0]]])
    output, actions = model.particle_rollout(
        torch.zeros(1, 1, 24),
        torch.empty(1, 0, dtype=torch.long),
        probabilities,
        1000,
        torch.Generator().manual_seed(2),
    )
    assert set(output[..., 0].flatten().tolist()) == {-1.0, 1.0}
    assert 0.4 < float((actions == 0).float().mean()) < 0.6


def test_predict_does_not_call_image_stem(model, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("latent rollout must not process pixels")

    monkeypatch.setattr(model.patch, "forward", fail)
    output = model.rollout(
        torch.rand(2, 1, 24), torch.empty(2, 0, dtype=torch.long), torch.tensor([[0, 1], [2, 3]])
    )
    assert output.shape == (2, 2, 24)


def test_matched_control_parameter_budget(model):
    control = WorldModel(matched_independent_config(model.config))
    assert not control.config.shared
    assert abs(control.parameter_count() / model.parameter_count() - 1) < 0.2
