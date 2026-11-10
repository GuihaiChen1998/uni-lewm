import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fused_lewm.data import TrajectoryDataset, generate_dataset
from fused_lewm.losses import SIGReg, world_model_loss
from fused_lewm.training import load_model, train


def test_sigreg_detects_collapse_and_non_gaussian_scale():
    reg = SIGReg(64)
    gaussian = torch.randn(1024, 12)
    torch.manual_seed(3)
    good = reg(gaussian)
    torch.manual_seed(3)
    collapsed = reg(torch.zeros_like(gaussian))
    torch.manual_seed(3)
    inflated = reg(gaussian * 5)
    assert good < collapsed and good < inflated
    gaussian.requires_grad_()
    reg(gaussian).backward()
    assert gaussian.grad.isfinite().all() and gaussian.grad.norm() > 0


def test_prediction_uses_future_only_as_target(model, monkeypatch):
    observations = torch.rand(3, 5, 3, 16, 16)
    actions = torch.randint(4, (3, 4))
    captured = []
    original = model.rollout

    def capture(history, past, future):
        captured.append((history.detach().clone(), past.clone(), future.clone()))
        return original(history, past, future)

    monkeypatch.setattr(model, "rollout", capture)
    loss, _ = world_model_loss(model, observations, actions, SIGReg(8), 0.1)
    loss.backward()
    changed = observations.clone()
    changed[:, 3:] = torch.rand_like(changed[:, 3:])
    world_model_loss(model, changed, actions, SIGReg(8), 0.1)
    for before, after in zip(captured[0], captured[1]):
        torch.testing.assert_close(before, after)


def test_episode_split_prevents_overlapping_window_leakage(tmp_path):
    path = tmp_path / "data.npz"
    generate_dataset(path, episodes=10, steps=8, image_size=16, seed=3)
    training = TrajectoryDataset(path, 2, 2, "train")
    validation = TrajectoryDataset(path, 2, 2, "validation")
    assert set(training.episode_ids).isdisjoint(validation.episode_ids)
    assert set(training.episode_ids) | set(validation.episode_ids) == set(range(10))
    observations, actions = training[0]
    assert observations.shape == (4, 3, 16, 16) and actions.shape == (3,)
    with pytest.raises(FileExistsError):
        generate_dataset(path)


def test_checkpoint_resume_reproduces_uninterrupted_training(tmp_path):
    config = json.loads((Path(__file__).parents[1] / "configs/smoke.json").read_text())
    path = tmp_path / "data.npz"
    generate_dataset(path, episodes=10, steps=8, image_size=16, seed=3)
    config["data"]["path"] = str(path)
    config["model"].update(image_size=16, dim=24, depth=1, history=2)
    config["training"].update(
        steps=4,
        batch_size=4,
        sigreg_projections=8,
        validation_batches=1,
        validate_every=2,
        save_every=2,
    )
    train(config, tmp_path / "full", "cpu")
    partial = copy.deepcopy(config)
    partial["training"]["steps"] = 2
    train(partial, tmp_path / "resumed", "cpu")
    train(config, tmp_path / "resumed", "cpu", tmp_path / "resumed/last.pt")
    full, _ = load_model(tmp_path / "full/last.pt")
    resumed, _ = load_model(tmp_path / "resumed/last.pt")
    for name, value in full.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)


def test_collapsed_data_fails_with_diagnostic_checkpoint(tmp_path):
    config = json.loads((Path(__file__).parents[1] / "configs/smoke.json").read_text())
    path = tmp_path / "constant.npz"
    np.savez_compressed(
        path,
        observations=np.zeros((6, 6, 3, 16, 16), dtype=np.uint8),
        actions=np.zeros((6, 5), dtype=np.int64),
    )
    config["data"]["path"] = str(path)
    config["model"].update(image_size=16, dim=24, depth=1, history=2)
    config["training"].update(
        steps=2,
        batch_size=4,
        sigreg_projections=8,
        validation_batches=1,
        validate_every=1,
        save_every=1,
        collapse_check_after=1,
    )
    with pytest.raises(RuntimeError, match="representation collapse"):
        train(config, tmp_path / "collapsed", "cpu")
    assert (tmp_path / "collapsed/last.pt").exists()
    assert json.loads((tmp_path / "collapsed/failure.json").read_text())["step"] == 1
